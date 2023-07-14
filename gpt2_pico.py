import numpy as np


def lm_loss(inputs: list[int], params) -> float:
    # the labels y are just the input shifted 1 to the left
    #
    # inputs = [not,     all,   heros,   wear,   capes]
    #      x = [not,     all,   heroes,  wear]
    #      y = [all,  heroes,     wear,  capes]
    #
    # of course, we don't have a label for inputs[-1], so we exclude it from x
    #
    # as such, for N inputs, we have N - 1 langauge modeling example pairs
    x, y = inputs[:-1], inputs[1:]

    # forward pass
    # all the predicted next token probability distributions at each position
    output = gpt(x, params)

    # cross entropy loss
    # we take the average over all N-1 examples
    loss = np.mean(-np.log(output[y]))

    return loss

def train(texts: list[list[str]], params) -> float:
    for text in texts:
        inputs = tokenizer.encode(text) # converts a str -> list[int] using the BPE tokenizer
        loss = lm_loss(inputs, params)
        gradients = compute_gradients_via_backpropagation(loss, params)
        params = gradient_descent_update_step(gradients, params)
    return params

# The non-linearity (activation function) of choice for GPT-2 is GELU (Gaussian Error Linear Units), an alternative for ReLU:
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Layer normalization standardizes values to have a mean of 0 and a variance of 1
def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

# Matrix multiplication + bias:
def linear(x, w, b):
    return x @ w + b

# Position-wise Feed Forward Network, simple multi-layer perceptron with 2 layers
def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)

# q:    query matrix
# k:    key matrix
# v:    value matrix
# mask: causal_mask matrix
def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

# x: embedding vectors
# c_attn: weight matrix for query, key, value in multi-head attention
# c_proj: weight matrix to convert output size to n_embd size
# n_head: head size
def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)

    # Query, Key, Value 로 split 하고 n_head(12)로 split
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1)))

    # n_seq=11
    #
    #[[-0.e+00 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10]
    # [-0.e+00 -0.e+00 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10]
    # [-0.e+00 -0.e+00 -0.e+00 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10]
    # [-0.e+00 -0.e+00 -0.e+00 -0.e+00 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10]
    # [-0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10]
    # [-0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -1.e+10 -1.e+10 -1.e+10 -1.e+10 -1.e+10]
    # [-0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -1.e+10 -1.e+10 -1.e+10 -1.e+10]
    # [-0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -1.e+10 -1.e+10 -1.e+10]
    # [-0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -1.e+10 -1.e+10]
    # [-0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -1.e+10]
    # [-0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00]]

    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = linear(np.hstack(out_heads), **c_proj)
    return x

    # Multi-head attention mechanism: http://jalammar.github.io/illustrated-transformer/

# x: embedding vectors
# mlp: weights for ffn
# attn: weighted attention
# ln_1: layer normalization params for mha
# ln_2: layer normalization params for ffn
# n_head: head size
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x

# inputs: encoded tokens
# wte: word token embedding
# wpe: positioning embedding
# blocks: transformer params (size: 12)
# ln_f: layer normalization params
# n_head: sublayer size: 12
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T

    # GPT Model: https://i.imgur.com/c4Z6PG8.png


# inputs: encoded tokens
# params: params
# n_head: sublayer size: 12
# n_tokens_to_generate: 40
def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm
    for _ in tqdm(range(n_tokens_to_generate), "generating"): # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head) # model forward pass
        next_id = np.argmax(logits[-1]) # greedy sampling
        inputs.append(int(next_id)) # append prediction to input
    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids

# prompt: input string
# n_tokens_to_generate
def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    input_ids = encoder.encode(prompt) # converts a str -> list[int] using the BPE tokenizer
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    output_text = encoder.decode(output_ids)
    return output_text

# python gpt2_pico.py "Alan Turing theorized that computers would one day become"
#
#generating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:12<00:00,  3.21it/s]
# the most powerful machines on the planet.
#
#The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.

if __name__ == "__main__":
    import fire
    fire.Fire(main)


