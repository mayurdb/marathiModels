import numpy as np
import math
import torch
from torch import nn

k = 50000
input_file_path = './data/mr.txt'
output_file_path = f"./data/mr_{k}.txt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
context_size = 9
num_iterations = 10000
eval_iters = 10
eval_interval = 1000 


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
n = math.floor(0.9 * len(data))
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
        X.append(x)
        Y.append(y)
        x, y = torch.tensor(X), torch.tensor(Y)
        x, y = x.to(device), y.to(device)
    return torch.tensor(X), torch.tensor(Y)


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
    model.train()
    return loss_item

from torch.nn import functional as F

class BiagramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, y):
        logits = self.embedding_table(x) #batch_size, context_size, vocab_size
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
            logits, loss = self(x, None)
            logits_filtered = logits[:, -1,:]
            probs = F.softmax(logits_filtered, dim=1)
            selected = torch.multinomial(probs, 1)
            assert(selected.shape == (1, 1))
            x = torch.cat((x, selected), dim=1)
        return x


model = BiagramLanguageModel(vocab_size)
m = model.to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)


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
    print(decode(generarted_text.numpy()[0]))

generate_sentences(1000)