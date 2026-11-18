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

