import torch
import datetime
import sys
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    elif torch.has_mps:
        return torch.device('mps')
    return torch.device('cpu')


class Head(nn.Module):
    def __init__(self, block_size, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        weight = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        weight = F.softmax(weight, dim=-1)  # (B, T, T)
        weight = self.dropout(weight)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = weight @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, block_size, n_embd, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size, n_embd, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, block_size, n_embd, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(block_size, n_embd, n_head, head_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, device='cpu'):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(block_size, n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)
        self.device = device
        self.block_size = block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class InputDataset(Dataset):
    def __init__(self, data, block_size):
        data_len = len(data)
        self.block_size = block_size
        self.X = torch.stack([data[i:i + block_size] for i in range(data_len - block_size)])
        self.Y = torch.stack([data[i + 1:i + block_size + 1] for i in range(data_len - block_size)])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)
    print(str(info) + "\n")


if __name__ == '__main__':
    batch_size = 64
    block_size = 256
    # max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    device = try_gpu()
    print(device)
    eval_iters = 200
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2

    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(vocab_size)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # Train and test splits
    print(len(text), len(text[0]), text[0])
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(len(train_data))
    train_dataset = InputDataset(train_data, block_size)
    val_dataset = InputDataset(val_data, block_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout, device)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    ckpt_path = './GPT_mini_model'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    monitor = 'val_loss'
    mode = 'min'
    patience = 5
    history = {}
    for epoch in range(10):
        printlog("Epoch {0} / {1}".format(epoch, 10))
        ## train
        model.train()
        total_loss, step = 0, 0
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, batch in loop:
            X, Y = batch
            logits, loss = model(X, Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step_log = {'train_loss': loss.item()}
            total_loss += loss.item()

            step += 1
            if i != len(train_dataloader) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_log = {'train_loss': epoch_loss}
                loop.set_postfix(**epoch_log)
        for name, metric in epoch_log.items():
            history[name] = history.get(name, []) + [metric]
        ## validation
        model.eval()
        total_loss, step = 0, 0
        loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        with torch.no_grad():
            for i, batch in loop:
                X, Y = batch
                preds, loss = model(X, Y)

                step_log = {'val_loss': loss.item()}
                total_loss += loss.item()

                step += 1
                if i != len(val_dataloader) - 1:
                    loop.set_postfix(**step_log)
                else:
                    epoch_loss = total_loss / step
                    epoch_log = {'val_loss': epoch_loss}
                    loop.set_postfix(**epoch_log)

        epoch_log["epoch"] = epoch
        for name, metric in epoch_log.items():
            history[name] = history.get(name, []) + [metric]

        ## early-stopping
        arr_scores = history[monitor]
        best_score_idx = np.argmin(arr_scores) if mode == 'min' else np.argmax(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            torch.save(model.state_dict(), ckpt_path)
            print(f"<<<<<< reach best {monitor}: {arr_scores[best_score_idx]} >>>>>>", file=sys.stderr)
        if len(arr_scores) - best_score_idx > patience:
            print(f"<<<<<< {monitor} without improvement in {patience} epoch, early stopping >>>>>>", file=sys.stderr)
            break

        model.load_state_dict(torch.load(ckpt_path))
    dfhistory = pd.DataFrame(history)
