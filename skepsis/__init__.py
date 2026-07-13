"""skepsis — is your backtest result real, or overfit?"""

__version__ = "0.1.0.dev0"

from skepsis.evaluate import DsrResult, PsrResult, Result, evaluate  # noqa: E402
from skepsis.verdict import Thresholds, Verdict  # noqa: E402

__all__ = [
    "DsrResult",
    "PsrResult",
    "Result",
    "Thresholds",
    "Verdict",
    "__version__",
    "evaluate",
]
