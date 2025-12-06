import torch
import triton
import triton.language as tl

@triton.jit
def fused_softmax_dropout_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols, p, seed,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    # Pointers
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load data with masking
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # --- TRAP: Numerical Stability (Safe Softmax) ---
    # We KEEP this trap. It prevents the 60% pass rate.
    # Naive exp(row) overflows if row has values > 88.
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max) 
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Generate dropout mask
    # We REMOVED the float scaling trap. Just use seed directly.
    rng_offset = row_idx * n_cols + col_offsets
    rand = tl.rand(seed, rng_offset)
    
    keep_mask = rand > p
    
    # Apply dropout and store
    output = softmax_output * keep_mask / (1 - p)
    
    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, output, mask=mask)

def fused_softmax_dropout(x, p, seed):
    """
    Args:
        x: Input tensor (2D)
        p: Dropout probability
        seed: Int seed (Standard integer seed)
    """
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    grid = (n_rows,)
    
    fused_softmax_dropout_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_cols, p, seed,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
