out_dir = 'out-syn_smiles'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 250 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'syn_smiles'
wandb_run_name = 'mini-gpt'

dataset = 'syn_smiles'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 1024 # context of up to 1024 previous characters

# baby GPT model :)
n_layer = 10
n_head = 8
n_embd = 512
dropout = 0.2

learning_rate = 8e-4 # with baby networks can afford to go a bit higher
max_iters = 120000000
lr_decay_iters = 120000000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 2000 # not super necessary potentially
