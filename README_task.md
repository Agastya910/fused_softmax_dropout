Fused Softmax + Dropout - Task Package
--------------------------------------

Visible to the agent:
- docs/prompt.txt         # short prompt that the LLM sees
- docs/specification.md   # full technical spec (copied from uploaded file)
- fused_softmax_dropout/  # package (contains stub kernel + utils)
- tests/test_fused_kernel.py  # automatic grader harness
- utils/seed_utils.py     # RNG helper exposed to agent

Hidden to the agent (kept in gold/):
- gold/oracle_fused_kernel.py  # gold/oracle implementation (PyTorch) - used by grader
- gold/utils/reference.py      # PyTorch reference implementation (numerical ground truth)

Evaluation:
1. Agent writes `fused_softmax_dropout/kernels/fused_kernel.py`.
2. Evaluator runs `pytest -q` (or run_tests_and_score.py).
3. Tests compare agent output to gold reference numerically.

To run locally:
- Activate your Python venv and ensure CUDA/Triton/PyTorch installed.
- Run `pytest -q` to see pass/fail.
- Use `./package_task.sh` to produce a task zip (includes gold for reviewers).
