# Fused Softmax + Dropout Triton Kernel

## Task Spec
Implement a fused, numerically-stable softmax + dropout kernel in Triton.

## Requirements
- Numerically stable softmax
- Seeded dropout
- Fully fused kernel
- Must match PyTorch reference output
- Deterministic for given seed
- Must handle large shapes
- Must be tile-based

## Grading
All grading is numeric-only:
- small correctness test
- large correctness test
- numerical stability test
- reproducibility test
