"""lerobot-curate: CPU-only curation/selection of LeRobot v3 robot datasets.

Public API is kept import-light and torch-free. Heavy/optional backends
(onnxruntime embedder, PyAV frame decode, lerobot, fiftyone) are imported lazily
inside the modules that need them, so ``import lerobot_curate`` never pulls a GPU
stack.
"""

from ._version import __version__

__all__ = ["__version__"]
