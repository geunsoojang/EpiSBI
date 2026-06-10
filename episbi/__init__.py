from .inference import SBIEngine
from .embedding import LSTMembedding
from .utils import plot_prediction_windows
from .simulation import sample_prior, simulate_for_sbi


__version__ = "0.1.0"

def __getattr__(name):
    if name == "SBIEngine":
        from .inference import SBIEngine
        return SBIEngine
    if name == "LSTMembedding":
        from .embedding import LSTMembedding
        return LSTMembedding
    raise AttributeError(f"module 'episbi' has no attribute {name!r}")

__all__ = ["SBIEngine", "LSTMembedding","plot_prediction_windows", "simulate_for_sbi"]