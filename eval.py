"""
Evaluate a trained model
"""
import os
import re
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import numpy as np

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-syn_smiles_mapped' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
block_size = 1024
batch_size = 32
max_new_tokens = 200 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
dataset = 'syn_smiles_mapped'
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'exp13.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in tokenise_smiles(s)]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
data_dir = os.path.join('data', dataset)
test_data = np.load(os.path.join(data_dir, 'test.npy'), mmap_mode='r')

def get_tree():
    ix = torch.randint(len(test_data), (batch_size,))
    x = torch.stack([torch.from_numpy((test_data[i][:block_size - 1]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((test_data[i][1:block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

num_batches = int(len(test_data)/batch_size)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(num_batches)
    for batch_num in range(num_batches):
        X, Y = get_tree()
        with ctx:
            logits, loss = model(X, Y)
        losses[batch_num] = loss.item()
    avg_loss = losses.mean()
    print(f"Average loss : {avg_loss}")
    return avg_loss

def tokenise_smiles(smi):
    pattern =  "(\[[^\]]+]|ERXN?|ETREE?|STREE?|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|!|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return tokens

def exact_match(pred, gt):
    return pred == gt

def get_generated_mols(predictions):
    targets = []
    for pred in predictions:
        try:
            rxns = pred.split('[ERXN]')
            tgt = rxns[0].split('>>')[1]
            targets.append(tgt)
        except:
            targets.append('NOMATCH')
    return targets

def top_k_accuracy(results_dict, top_k):
    correct_predictions = 0
    for result in results_dict.values():
        if True in result[:top_k+1]:
            correct_predictions += 1

    print(f'Accuracy for k = {top_k} is : {correct_predictions/len(results_dict)}')


def eval_exact_match():
    tree_dict = {}
    results_dict = {}
    for tree in test_data:
        tree_str = decode(tree)
        mols = tree_str.split('>>')
        root = mols[0]
        tgt = mols[1].split('[ERXN]')[0]
        tree_dict[root] = tgt

        ids = encode(root)
        inp = x = (torch.tensor(ids, dtype=torch.long, device=device)[None, ...])
        with torch.no_grad():
            with ctx:
                res = []
                for k in range(5):
                    y = model.generate(inp, max_new_tokens, temperature=temperature, top_k=200)
                    res.append(decode(y[0].tolist()))

        preds = get_generated_mols(res)
        top_ks = [exact_match(pred, root) for pred in preds]
        results_dict[root] = top_ks

    

    for k in range(top_k):
        top_k_accuracy(results_dict, k)
    


# estimate_loss()
eval_exact_match()


