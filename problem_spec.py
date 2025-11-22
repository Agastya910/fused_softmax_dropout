"""
HUD Problem Specification for Fused Softmax-Dropout Task
"""

from dataclasses import dataclass
from typing import List

@dataclass
class ProblemSpec:
    name: str = "fused_softmax_dropout"
    display_name: str = "Fused Softmax-Dropout Kernel"
    description: str = "Implement a fused softmax + dropout GPU kernel using Triton"
    
    # Repository
    repo_url: str = "https://github.com/Agastya910/fused_softmax_dropout.git"
    
    # Branches
    baseline_branch: str = "baseline"
    test_branch: str = "test"
    golden_branch: str = "golden"
    
    # Evaluation
    test_command: str = "PYTHONPATH=. pytest tests/test_fused_kernel.py -v"
    test_timeout: int = 300
    
    # Files
    prompt_file: str = "docs/prompt.txt"
    main_file: str = "fused_softmax_dropout/kernels/fused_kernel.py"
    
    # Metadata
    tags: List[str] = None
    estimated_pass_at_10: float = 0.25
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = ["gpu", "triton", "kernel-optimization", "ml-systems"]

FUSED_SOFTMAX_DROPOUT = ProblemSpec()
