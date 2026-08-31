"""
Models and architectural components for LeJEPA with SIGReg.
"""

from .sigreg import SIGReg, sigreg_loss
from .lejepa_module import LeJEPA, LeJEPAEncoder, MLPProjector, MLPPredictor

__all__ = [
    "SIGReg",
    "sigreg_loss",
    "LeJEPA",
    "LeJEPAEncoder",
    "MLPProjector",
    "MLPPredictor",
]
