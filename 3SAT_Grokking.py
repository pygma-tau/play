# %% [markdown]
# # Grokking on Algorithmic Tasks

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import random
from typing import List, Optional
from pysat.solvers import Glucose4  

# %%
def generate_random_3sat_instance(
    n_vars: int,
    ratio: float = 4.2,
    seed: Optional[int] = None
):
    if seed is not None:
        random.seed(seed)

    n_clauses = int(ratio * n_vars)
    clauses = []
    for _ in range(n_clauses):
        vars_in_clause = random.sample(range(1, n_vars + 1), 3)
        clause = [v if random.random() < 0.5 else -v for v in vars_in_clause]
        clauses.append(clause)
    return clauses


def solve_3sat(clauses: List[List[int]], n_vars: int):

    solver = Glucose4()
    for clause in clauses:
        solver.add_clause(clause)
    sat = solver.solve()
    if not sat:
        solver.delete()
        return None
    model = solver.get_model()  # list of ints with sign indicating T/F
    solver.delete()
    assignment = [False] * n_vars
    for lit in model:
        idx = abs(lit) - 1
        if 0 <= idx < n_vars:
            assignment[idx] = (lit > 0)
    return assignment


class ThreeSatDataset(Dataset):


    def __init__(
        self,
        num_samples: int,
        n_vars: int,
        ratio: float = 4.2,
        max_tries: int = 1000,
        skip_unsat: bool = True
    ):
        super().__init__()
        self.data = []
        self.n_vars = n_vars

        count = 0
        while len(self.data) < num_samples and count < max_tries:
            clauses = generate_random_3sat_instance(n_vars, ratio)
            solution = solve_3sat(clauses, n_vars)
            if solution is not None:
                input_repr = self._encode_instance(clauses, n_vars)
                assignment = [1 if x else 0 for x in solution]
                self.data.append((input_repr, assignment))
            else:
                if not skip_unsat:
                    input_repr = self._encode_instance(clauses, n_vars)
                    assignment = [0] * n_vars
                    self.data.append((input_repr, assignment))
            count += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _encode_instance(self, clauses: List[List[int]], n_vars: int):
        tokens = [0]  # start token
        # We'll map literal i (±1..±n_vars) to int: +x -> x, -x -> n_vars + |x|
        # That is, pos i in [1..n_vars] = i, neg i = n_vars + |i|.
        for clause in clauses:
            for lit in clause:
                if lit > 0:
                    tokens.append(lit)  # e.g. 1..n_vars
                else:
                    tokens.append(n_vars + abs(lit))  # e.g. n_vars+1..2*n_vars
            tokens.append(2 * n_vars + 1)  # a <sep> token
        tokens.append(2 * n_vars + 2)  # <end> token
        return torch.tensor(tokens, dtype=torch.long)

# %%
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.attn_ln = nn.RMSNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ff_ln = nn.RMSNorm(d_model)

    def forward(self, x, mask=None):
        # x shape: (batch, seq_len, d_model)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.attn_ln(x + attn_out)
        ff_out = self.ff(x)
        x = self.ff_ln(x + ff_out)
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        d_ff=512,
        num_layers=4,
        n_vars=10
    ):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.RMSNorm(d_model)
        self.classifier = nn.Linear(d_model, n_vars * 2)

    def forward(self, x, mask=None):
        emb = self.embed(x)  # (batch, seq_len, d_model)
        hidden = emb
        for block in self.blocks:
            hidden = block(hidden, mask=mask)
        hidden = self.final_ln(hidden)
        start_token_hidden = hidden[:, 0, :]  # (batch, d_model)
        logits = self.classifier(start_token_hidden)  # (batch, n_vars*2)
        logits = logits.view(-1, self.n_vars, 2)      # (batch, n_vars, 2)
        return logits

# %%
def collate_fn(batch):
    inputs = [b[0] for b in batch]
    targets = [b[1] for b in batch]


    input_lens = [len(i) for i in inputs]
    max_len = max(input_lens)
    padded_inputs = []
    for inp in inputs:
        pad_len = max_len - len(inp)
        padded_inp = torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)], dim=0)
        padded_inputs.append(padded_inp)
    padded_inputs = torch.stack(padded_inputs, dim=0)


    targets = torch.tensor(targets, dtype=torch.long)

    return padded_inputs, targets


def main():

    N_VARS = 5      # number of variables per sample
    RATIO = 3.0      # clauses-to-variables ratio
    TRAIN_SAMPLES = 200000
    VAL_SAMPLES = 20000
    BATCH_SIZE = 128
    EPOCHS = 1000
    LR = 1e-3


    vocab_size = 2*N_VARS + 3  # 3 for [start], <sep>, <end> (some margin)


    train_ds = ThreeSatDataset(num_samples=TRAIN_SAMPLES, n_vars=N_VARS, ratio=RATIO)
    val_ds = ThreeSatDataset(num_samples=VAL_SAMPLES, n_vars=N_VARS, ratio=RATIO)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)


    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        num_layers=8,
        n_vars=N_VARS
    ).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []


    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_inp, batch_tgt in train_loader:
            batch_inp = batch_inp.to(device)
            batch_tgt = batch_tgt.to(device)  # shape: (batch, n_vars)

            optimizer.zero_grad()
            logits = model(batch_inp)        
            logits_flat = logits.view(-1, 2)
            targets_flat = batch_tgt.view(-1)
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inp, batch_tgt in val_loader:
                batch_inp = batch_inp.to(device)
                batch_tgt = batch_tgt.to(device)

                logits = model(batch_inp)
                logits_flat = logits.view(-1, 2)
                targets_flat = batch_tgt.view(-1)
                loss = criterion(logits_flat, targets_flat)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            model.eval()
            test_clauses = generate_random_3sat_instance(N_VARS, RATIO)
            test_solution = solve_3sat(test_clauses, N_VARS)
            encoded = train_ds._encode_instance(test_clauses, N_VARS).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_logits = model(encoded)
                pred = pred_logits.argmax(dim=-1).squeeze(0).cpu().tolist()
            
            print("\nSample test case:")
            print("Clauses:", test_clauses)
            print("SAT solver solution:", [1 if x else 0 for x in test_solution])
            print("Model prediction:", pred)
            print("Match:", pred == [1 if x else 0 for x in test_solution], "\n")

if __name__ == "__main__":
    main()


