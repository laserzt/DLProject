import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from tokenization import *
from common import *
import matplotlib.pyplot as plt
import numpy as np

# hyperparameters
batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
begin_iters = 0
max_iters = 50
eval_interval = 10
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 1024
n_head = 32
n_layer = 12
dropout = 0.2
model = None
m = None
# ------------

torch.manual_seed(1)

# Load Data    
d = []


def add_to_data(f):
    global d
    _, channels = load_tokens(f)
    for k in channels:
        # songs are separated by silence, which is later removed
        d += channels[k] + [silence_token] * block_size


iterate_all_files(add_to_data, file_type='.tokens', run_in_threads=False)

data = torch.tensor(d, dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# get random batch
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


"""# The Attention Head
This is the most important block.
"""


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
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
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


"""Many Heads are combined into a Multi-Attention Head"""


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


"""# The Transformer
We combine  the Multi-Attention Head with a two-layer MLP to create a Transformer Block
"""


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
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

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


"""# The GPT Language Model"""


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

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
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
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
        # idx is (B, T) array of indices in the current context
        accuracy = torch.zeros((1, batch_size))
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # add sampled probability
            accuracy += probs[:, idx_next[:, 0]]
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx, accuracy / max_new_tokens


def calculate_accuracy(iterations=10, model_num=None):
    global model
    global m
    if model_num:
        model = GPTLanguageModel()
        model.load_state_dict(torch.load(os.path.join(out_dir, f'model_{model_num}')))
        model.eval()
        m = model.to(device)
    res = []
    for i in range(iterations):
        context = torch.randint(vocab_size, (2,)).unsqueeze(0)
        _, accuracy = m.generate(context, max_new_tokens=10)
        res += list(accuracy.detach().numpy()[0])
    return sum(res) / len(res)


def train_gpt():
    global model
    global m
    model = GPTLanguageModel()
    if begin_iters > 0:
        model.load_state_dict(torch.load(os.path.join(out_dir, f'{begin_iters}')))
        model.eval()
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_time = time.time()

    for iter in range(begin_iters + 1, max_iters + 1):
        # every once in a while evaluate the loss on train and val sets and save the model
        if iter % eval_interval == 0 or iter == max_iters:
            losses = estimate_loss()
            acc = calculate_accuracy()
            print(
                f"step {iter}: training loss {losses['train']:.4f}, validation loss {losses['val']:.4f}, accuracy {acc:.4f}, elapsed time: {time.time() - start_time:.1f} secs")
            torch.save(model.state_dict(), os.path.join(out_dir, f'model_{iter}'))

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def write_song(model_num=None):
    global model
    global m
    if model_num:
        model = GPTLanguageModel()
        model.load_state_dict(torch.load(os.path.join(out_dir, f'model_{model_num}')))
        model.eval()
        m = model.to(device)
    context = torch.randint(vocab_size, (2,)).unsqueeze(0)
    write_midi(detokenize(m.generate(context, max_new_tokens=250)[0][0].tolist()), 'gpt_song')


if __name__ == '__main__':
    train_gpt()
    write_song()
