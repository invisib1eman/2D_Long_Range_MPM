"""
2D Long Range MPM Simulations for Plasmonics Assembly

This package provides tools for simulating plasmonic assembly using
2D Long Range MPM (Multipole Particle Method) simulations.
"""

__version__ = '0.1.0'

# Import main functionality
try:
    from mean_field_model import *
    from .utils import *
except ImportError as e:
    print(f"Warning: Could not import all modules - {e}")

# Package-level documentation
__doc__ = """
Plasmonics Assembly Simulation Package

This package contains:
- Core simulation models in model.py
- Utility functions in utils/
- Visualization tools in display.ipynb
"""