"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - stable_softmax
import numpy as np

def stable_softmax(logits):
    # Subtract the maximum along the last axis for numerical stability
    shift_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shift_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# Step 2 - apply_temperature
import numpy as np

def apply_temperature(logits, temperature):
    if temperature <= 0:
        return logits.copy()
    return logits / temperature

# Step 3 - top_k_filter
import numpy as np

def top_k_filter(logits, k):
    """Mask logits outside the top-k per row to -inf."""
    vocab_size = logits.shape[-1]
    if k >= vocab_size:
        return logits.copy()
    
    # Sort descending along the last axis and find the k-th largest value per row
    kth_vals = np.sort(logits, axis=-1)[..., -k][..., None]
    
    # Keep values >= kth_vals, set the rest to -inf
    return np.where(logits >= kth_vals, logits, -np.inf).astype(logits.dtype)

# Step 4 - top_p_filter
import numpy as np

def top_p_filter(logits, p):
    # Compute probabilities using stable_softmax from step 1
    probs = stable_softmax(logits)
    
    # Sort indices descending along the last axis
    sorted_indices = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
    
    # Compute cumulative probabilities and build mask in sorted order
    cum_probs = np.cumsum(sorted_probs, axis=-1)
    sorted_mask = cum_probs >= p
    
    # Shift right by 1 so the token that crosses the threshold p is included
    sorted_mask[..., 1:] = sorted_mask[..., :-1]
    sorted_mask[..., 0] = False
    
    # Scatter the mask back to the original vocabulary positions
    remove_mask = np.zeros_like(sorted_mask, dtype=bool)
    np.put_along_axis(remove_mask, sorted_indices, sorted_mask, axis=-1)
    
    return np.where(remove_mask, -np.inf, logits).astype(logits.dtype)

# Step 5 - sample_from_probs
import numpy as np

def sample_from_probs(probs, rng):
    # Sample a single token index using the provided numpy.random.Generator
    idx = rng.choice(len(probs), p=probs)
    return int(idx)

# Step 6 - greedy_select
import numpy as np

def greedy_select(logits):
    # np.argmax automatically returns the first (lowest) index in case of ties
    return int(np.argmax(logits))

# Step 7 - build_vocab
def build_vocab(corpus, special_tokens):
    # Initialize id_to_token with special tokens in order
    id_to_token = list(special_tokens)
    
    # Collect all unique characters from the corpus
    corpus_chars = set()
    for text in corpus:
        for char in text:
            corpus_chars.add(char)
            
    # Sort remaining unique characters and append them
    sorted_chars = sorted(list(corpus_chars))
    for char in sorted_chars:
        if char not in id_to_token:
            id_to_token.append(char)
            
    # Build token_to_id mapping
    token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
    
    return {
        'token_to_id': token_to_id,
        'id_to_token': id_to_token
    }

# Step 8 - encode_prompt
def encode_prompt(text, vocab, add_bos=True):
    token_to_id = vocab['token_to_id']
    ids = []
    
    # Prepend <bos> if requested and present in vocab
    if add_bos and '<bos>' in token_to_id:
        ids.append(token_to_id['<bos>'])
        
    unk_id = token_to_id.get('<unk>')
    
    # Map each character to its token ID, falling back to <unk> or dropping
    for char in text:
        if char in token_to_id:
            ids.append(token_to_id[char])
        elif unk_id is not None:
            ids.append(unk_id)
            
    return ids

# Step 9 - decode_tokens
def decode_tokens(token_ids, vocab, skip_special=True):
    id_to_token = vocab['id_to_token']
    tokens = []
    
    for idx in token_ids:
        token = id_to_token[int(idx)]
        if skip_special and token.startswith('<') and token.endswith('>'):
            continue
        tokens.append(token)
        
    return ''.join(tokens)

# Step 10 - embed_tokens
import numpy as np

def embed_tokens(token_ids, embedding_matrix):
    # Index into the (V, D) embedding matrix using the 1D sequence of token_ids
    return embedding_matrix[np.asarray(token_ids, dtype=int)]

# Step 11 - linear_projection
import numpy as np

def linear_projection(x, weight, bias=None):
    y = x @ weight
    if bias is not None:
        y = y + bias
    return y

# Step 12 - init_kv_cache
import numpy as np

def init_kv_cache(max_seq_len, d_model):
    return {
        'K': np.zeros((max_seq_len, d_model), dtype=np.float32),
        'V': np.zeros((max_seq_len, d_model), dtype=np.float32),
        'length': 0
    }

# Step 13 - append_kv
import numpy as np

def append_kv(cache, k_new, v_new):
    start = cache['length']
    t = k_new.shape[0]
    
    cache['K'][start : start + t] = k_new
    cache['V'][start : start + t] = v_new
    cache['length'] += t
    
    return cache

# Step 14 - causal_attention
import numpy as np

def causal_attention(q, k, v, is_causal=True):
    Tq, D = q.shape
    Tk = k.shape[0]
    
    # Compute scaled dot-product attention scores: (Tq, Tk)
    scores = (q @ k.T) / np.sqrt(D)
    
    # Apply causal mask: position i cannot attend to keys at j > i + (Tk - Tq)
    if is_causal:
        i_idx = np.arange(Tq)[:, None]
        j_idx = np.arange(Tk)[None, :]
        mask = j_idx > (i_idx + (Tk - Tq))
        scores = np.where(mask, -np.inf, scores)
        
    # Normalize row-wise using stable_softmax from step 1
    weights = stable_softmax(scores)
    
    # Weighted sum of values: (Tq, D)
    return weights @ v

# Step 15 - model_prefill
import numpy as np

def model_prefill(token_ids, params):
    # 1. Embed input tokens -> (T, D)
    x = embed_tokens(token_ids, params['embedding'])
    d_model = x.shape[-1]
    
    # 2. Project to Q, K, V -> each (T, D)
    q = linear_projection(x, params['Wq'])
    k = linear_projection(x, params['Wk'])
    v = linear_projection(x, params['Wv'])
    
    # 3. Initialize KV cache and populate with K and V
    cache = init_kv_cache(params['max_seq_len'], d_model)
    cache = append_kv(cache, k, v)
    
    # 4. Run causal self-attention and output projection -> (T, D)
    attn_out = causal_attention(q, k, v, is_causal=True)
    hidden = linear_projection(attn_out, params['Wo'])
    
    # 5. Project last token's hidden state to vocabulary logits -> (V,)
    logits = linear_projection(hidden[-1], params['W_out'])
    
    return logits, cache

# Step 16 - model_decode_step
import numpy as np

def model_decode_step(token_id, cache, params):
    """Advance generation by one token using the existing KV cache."""
    # 1. Embed the single incoming token -> (1, D)
    emb_matrix = params.get('embedding', params.get('E'))
    x = embed_tokens([token_id], emb_matrix)
    
    # 2. Project to Q, K, V -> each (1, D)
    q = linear_projection(x, params['Wq'])
    k_new = linear_projection(x, params['Wk'])
    v_new = linear_projection(x, params['Wv'])
    
    # 3. Update the KV cache in place with the new K/V row
    cache = append_kv(cache, k_new, v_new)
    
    # 4. Attend over every stored entry in the cache -> (1, D)
    L = cache['length']
    k_all = cache['K'][:L]
    v_all = cache['V'][:L]
    attn_out = causal_attention(q, k_all, v_all, is_causal=True)
    
    # 5. Output projection and vocabulary logits -> (V,)
    hidden = linear_projection(attn_out, params['Wo'])
    logits = linear_projection(hidden[0], params['W_out'])
    
    return logits, cache

# Step 17 - blocks_needed
def blocks_needed(num_tokens, block_size):
    return (num_tokens + block_size - 1) // block_size

# Step 18 - init_block_allocator
import numpy as np

def init_block_allocator(num_blocks, block_size, d_model):
    return {
        'K_blocks': np.zeros((num_blocks, block_size, d_model), dtype=np.float32),
        'V_blocks': np.zeros((num_blocks, block_size, d_model), dtype=np.float32),
        'free_list': list(range(num_blocks)),
        'block_size': block_size,
        'num_blocks': num_blocks,
        'd_model': d_model,
        'seq_tables': {}
    }

# Step 19 - allocate_block
def allocate_block(allocator, seq_id):
    # Check if any free blocks remain
    if not allocator['free_list']:
        raise RuntimeError("Out of memory: no free blocks remaining in the allocator.")
    
    # Pop from the end of the free list (LIFO)
    block_id = allocator['free_list'].pop()
    
    # Ensure the sequence has an entry in seq_tables, then append the block id
    if seq_id not in allocator['seq_tables']:
        allocator['seq_tables'][seq_id] = []
    allocator['seq_tables'][seq_id].append(block_id)
    
    return block_id

# Step 20 - free_block
def free_block(allocator, block_id):
    # Append the block_id back to the free list
    allocator['free_list'].append(block_id)

# Step 21 - append_to_paged_cache
import numpy as np

def append_to_paged_cache(allocator, seq_id, k_new, v_new):
    """Write t new K/V rows into the sequence's paged blocks, allocating as needed."""
    if 'seq_lengths' not in allocator:
        allocator['seq_lengths'] = {}
        
    current_len = allocator['seq_lengths'].get(seq_id, 0)
    t = k_new.shape[0]
    new_len = current_len + t
    
    block_size = allocator['block_size']
    needed_blocks = blocks_needed(new_len, block_size)
    
    # Ensure seq_tables has an entry
    if seq_id not in allocator['seq_tables']:
        allocator['seq_tables'][seq_id] = []
        
    current_blocks = len(allocator['seq_tables'][seq_id])
    
    # Allocate new blocks as needed
    for _ in range(needed_blocks - current_blocks):
        allocate_block(allocator, seq_id)
        
    # Write rows into the physical blocks
    for i in range(t):
        logical_pos = current_len + i
        block_idx_in_table = logical_pos // block_size
        offset_in_block = logical_pos % block_size
        
        block_id = allocator['seq_tables'][seq_id][block_idx_in_table]
        
        allocator['K_blocks'][block_id, offset_in_block] = k_new[i]
        allocator['V_blocks'][block_id, offset_in_block] = v_new[i]
        
    # Update running length
    allocator['seq_lengths'][seq_id] = new_len

