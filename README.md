# Fused Softmax-Dropout Kernel

RL task for training LLMs to write optimized GPU kernels in Triton.

## Task Description

Implement a fused softmax + dropout kernel that:
- Operates on 2D tensors [batch_size, seq_len]
- Applies numerically stable softmax (subtract max before exp)
- Applies reproducible dropout with seed
- Matches PyTorch reference implementation within tolerance

## Repository Structure
```
fused_softmax_dropout/
├── fused_softmax_dropout/
│ ├── kernels/fused_kernel.py # Student implementation target
│ └── utils/reference.py # PyTorch oracle
├── tests/test_fused_kernel.py # Hidden grading tests
├── docs/
│ ├── prompt.txt # LLM instructions
│ └── specification.md # Full specification
├── problem_spec.py # HUD integration config
├── task.json # Task metadata
└── README.md
```
## Branches

- **baseline**: Stub kernel (LLM starting point, no tests visible)
- **test**: Stub kernel + hidden tests (for grading)
- **golden**: Working solution (validation, no tests)

## Local Testing
```
pip install torch triton pytest
PYTHONPATH=. pytest tests/test_fused_kernel.py -v
```
## Expected Pass Rate

**Target: 10-40% pass@10**

LLMs struggle with:
- Triton's tile-based programming model
- Kernel-level RNG generation
- Numerical stability (max subtraction in kernel)
- Single-kernel fusion
- Memory coalescing patterns

## HUD Integration

See `problem_spec.py` for evaluation framework configuration.

**Repository:** https://github.com/Agastya910/fused_softmax_dropout

