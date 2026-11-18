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

