# Fused Softmax-Dropout Task - Submission

## Repository
https://github.com/Agastya910/fused_softmax_dropout

## Branch Structure

### `baseline`
- **Purpose:** LLM starting point
- **Kernel:** Stub (raises NotImplementedError)
- **Tests:** None visible
- **Status:** ✅ Verified

### `test`
- **Purpose:** Grading branch
- **Kernel:** Stub (raises NotImplementedError)
- **Tests:** 4 hidden tests present
- **Status:** ✅ All tests FAIL on stub (expected)

### `golden`
- **Purpose:** Validation (proves task is solvable)
- **Kernel:** Working oracle implementation
- **Tests:** None (removed for validation branch)
- **Status:** ✅ Verified

## Task Specifications

**Target Difficulty:** 10-40% pass@10

**Why LLMs Struggle:**
1. Triton's tile-based programming model
2. Kernel-level RNG for dropout (not PyTorch's dropout)
3. Numerical stability (max subtraction in kernel)
4. Single-kernel fusion constraint
5. Memory coalescing requirements

**Grading Method:**
- Numerical comparison: torch.allclose(output, reference, atol=1e-5, rtol=1e-4)
- Deterministic (seeded RNG)
- 4 tests: small/large tensors, numerical stability, reproducibility

## Files

- problem_spec.py - HUD integration configuration
- docs/prompt.txt - LLM instructions
- docs/specification.md - Full technical spec
- task.json - Task metadata
- README.md - Documentation
- SUBMISSION.md - This verification summary

## Verification Checklist

✅ baseline: No tests, stub kernel  
✅ test: Has tests, stub kernel (all 4 tests fail)  
✅ golden: No tests, working kernel  
✅ main: Complete documentation  
✅ No gold/ directory in public branches  
✅ .gitignore properly configured

## Contact
**Author:** Agastya  
**GitHub:** @Agastya910
