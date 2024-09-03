import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F


k = 70000
input_file_path = './data/mr.txt'
output_file_path = f"./data/mr_{k}.txt"
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
batch_size = 32
# context_size = 9
context_size = 64
num_iterations = 100000
# eval_iters = 10
eval_iters = 100
# eval_interval = 1000 
eval_interval = 1000
embedding_dim = 516
num_layers = 8
num_heads = 4
dropout = 0.2
weights_path = '/Users/mayurb/src/open/marathiModels/weights/gpt2_marathi.pth'


# input => batch x context_size x vocab_size
# embedding => vocab_size x embedding_dim => results in => context_size x embedding_dim => 9 * 32
# position => context_size x embedding_dim => results in => context_size x embedding_dim => 9 * 32
# Num model parameters calculation
# Embedding table: vocab_size * embedding_dim => 332 * 32 => 10624
# Position table: context_size * embedding_dim => 9 * 32
# Blocks: num_layers * (3 * embedding_dim^2) => 8 * (3 * 32^2)  
# where, Block = MultiHeadAttention + FeedForward + 2 * LayerNorm
# where, MultiHeadAttention = num_heads * (3 * embedding_dim^2)
# where head = embedding_dim // num_heads
# Norm: 2 * embedding_dim
# lm_head: embedding_dim * vocab_size


# READ DATA
# Function to read the first k lines from the input file and write them to the output file
def read_and_write_first_k_lines(input_file, output_file, num_lines=1000):
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for i in range(num_lines):
                line = infile.readline()
                if not line:  # End of file reached before 1000 lines
                    break
                outfile.write(line)
        print(f"Successfully wrote the first {num_lines} lines to {output_file}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function
read_and_write_first_k_lines(input_file_path, output_file_path, k)

data_file = output_file_path
with open(data_file, 'r') as file:
    lines = file.readlines()


# GET VOCAB
vocab = set()
sos_char = '♣'
eos_char = '♦'
for line in lines:
    if line.strip() != "":
        line = sos_char + line.strip() + eos_char
        for ch in line:
            vocab.add(ch)
vocab = list(vocab)
vocab_size = len(vocab)

# CHARACTER ENCODER-DECODER
s_to_i = {char: i for i, char in enumerate(vocab)}
i_to_s = {i: char for i, char in enumerate(vocab)}
encode = lambda x: [s_to_i[char] for char in x]
decode = lambda x: "".join([i_to_s[num] for num in x])


# GENERATE DATASET
data = []
for line in lines:
    if line.strip() != "":
        line = sos_char + line.strip() + eos_char
        # Take each line and encode it
        data_local = []
        for ch in line:
            data_local.append(s_to_i[ch])
        data.append(data_local)
# We will discard the examples which are < context_size
print(f"Original length: {len(data)}")
all_data = [x for x in data if len(x) >= context_size+1]
print(f"Filtered length: {len(data)}")


# TRAIN-TEST split
n = math.floor(0.9 * len(all_data))
train_data = all_data[:n]
val_data = all_data[n:]
print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")

# Function to get a batch of the data
def get_batch(split, batch_size, context_size):
    data = train_data if split == 'train' else val_data
    random_indices = torch.randint(0, len(data), (batch_size,))
    data_filtered = [data[i] for i in random_indices]
    X = []
    Y = []
    for data_item in data_filtered:
        start_index = torch.randint(0, len(data_item) - context_size, (1,))[0].item()
        x = data_item[start_index:start_index+context_size]
        y = data_item[start_index+1:start_index+context_size+1]
        x, y = torch.tensor(x), torch.tensor(y)
        x, y = x.to(device), y.to(device)
        X.append(x)
        Y.append(y)
    # return torch.tensor(X), torch.tensor(Y)
    # return X, Y
    return torch.stack(X), torch.stack(Y)


@torch.no_grad()
def estimate_loss():
    loss_item = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, batch_size, context_size)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        loss_mean = losses.mean()
        loss_item[split] = loss_mean
    # save the weights
    torch.save(model.state_dict(), weights_path)
    model.train()
    return loss_item


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size)
        self.query = nn.Linear(embedding_dim, head_size)
        self.value = nn.Linear(embedding_dim, head_size)
        self.register_buffer('mask', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        batch_size, context_size, embedding_dim = x.shape
        
        key = self.key(x) # batch_size, context_size, head_size
        query = self.query(x) # batch_size, context_size, head_size
        wei = key @ query.transpose(-1, -2) * key.shape[-1]**-0.5 # (batch_size, context_size, head_size) @ (batch_size, head_size, context_size) = (batch_size, context_size, context_size)
        wei = wei.masked_fill(self.mask[:context_size, :context_size] == 0, float('-inf')) # batch_size, context_size, context_size
        wei = F.softmax(wei, dim=-1) # batch_size, context_size, context_size
        wei = self.dropout(wei)
        
        value = self.value(x) # batch_size, context_size, head_size
        out = wei @ value # (batch_size, context_size, context_size) @ (batch_size, context_size, head_size) = (batch_size, context_size, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, num_heads * head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        heads = [head(x) for head in self.heads] # num_heads, batch_size, context_size, head_size
        out = torch.cat(heads, dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out
        
class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, num_heads, embedding_dim):
        super().__init__()
        self.sa_heads = MultiHeadAttention(num_heads, embedding_dim//num_heads)
        self.feed_forward = FeedForward(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        x = x + self.sa_heads(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class BiagramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_table = nn.Embedding(context_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(num_heads, embedding_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, y):
        
        batch_size, context_size = x.shape

        embeddings = self.embedding_table(x) # batch_size, context_size, embedding_dim
        position_embeddings = self.position_table(torch.arange(context_size, device=device)) # context_size, embedding_dim
        x = embeddings + position_embeddings
        x = self.blocks(x) # batch_size, context_size, embedding_dim
        x = self.norm(x) # batch_size, context_size, embedding_dim
        logits = self.lm_head(x) #batch_size, context_size, vocab_size
        if y is not None:
            batch, context, embedding = logits.shape
            assert(batch == batch_size)
            assert(context == context_size)
            assert(embedding == vocab_size)
            logits = logits.view(batch * context, embedding)
            batch, context = y.shape
            assert(batch == batch_size)
            assert(context == context_size)
            targets = y.view(batch * context)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss


    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            x_trimmed = x[:, -context_size:]
            logits, loss = self(x_trimmed, None)
            logits_filtered = logits[:, -1,:]
            probs = F.softmax(logits_filtered, dim=1)
            selected = torch.multinomial(probs, 1)
            assert(selected.shape == (1, 1))
            x = torch.cat((x, selected), dim=1)
        return x

model = BiagramLanguageModel()
m = model.to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_model():
    for i in range(num_iterations):
        x, y = get_batch('train', batch_size, context_size)
        logits, loss = model(x, y)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        if i % eval_interval == 0:
            loss_item = estimate_loss()
            print(f"Iteration: {i}, Train loss: {loss_item['train']}, Validation loss: {loss_item['val']}")
        optimiser.step()


def generate_sentences(max_tokens):
    feed = s_to_i[sos_char]
    inp = torch.zeros(1, 1, dtype=torch.long, device=device)
    inp[0][0] = feed
    generarted_text = model.generate(inp, max_tokens)
    print(decode(generarted_text.cpu().numpy()[0]))

# train_model()
# generate_sentences(1000)
    
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
print("Vocab size: ", vocab_size)