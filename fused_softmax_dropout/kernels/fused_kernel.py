import torch
import triton
import triton.language as tl

SEED_SCALE_FACTOR = 10000.0

@triton.jit
def fused_softmax_dropout_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols, p, seed_float,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # TRAP 1: Numerical Stability
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max) 
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # TRAP 2: Float Seed Scaling
    seed_int = (seed_float * SEED_SCALE_FACTOR).to(tl.int32)
    
    rng_offset = row_idx * n_cols + col_offsets
    rand = tl.rand(seed_int, rng_offset)
    keep_mask = rand > p
    
    output = softmax_output * keep_mask / (1 - p)
    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, output, mask=mask)

def fused_softmax_dropout(x, p, seed):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    grid = (n_rows,)
    fused_softmax_dropout_kernel[grid](
        y, x, x.stride(0), y.stride(0),
        n_cols, p, seed, BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
