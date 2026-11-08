"""
AGRI-FUSE Hybrid Ensemble (AF-HE) Package.
Adaptive Gated Multimodal Deep Learning with Tree-Based Ensemble Learning for Agricultural Intelligence.
"""

import os
import sys

# Ensure serialized execution environment on Apple Silicon M2 and multi-runtime OpenMP
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

__version__ = "2.1.0"
__author__ = "AGRI-FUSE Research Team"
