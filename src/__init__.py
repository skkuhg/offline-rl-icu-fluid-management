"""
Offline RL ICU Fluid Management Package

This package provides modular utilities for offline reinforcement learning
applied to ICU fluid management decisions.

Modules:
- data_utils: Synthetic dataset generation and data processing
- offline_rl: Training utilities for BC, CQL, and IQL models
- visualization: Plotting and analysis functions
"""

__version__ = "1.0.0"
__author__ = "Generated from Jupyter Notebook"

# Import main classes and functions for easy access
try:
    from .data_utils import generate_synthetic_icu_episode
    from .offline_rl import OfflineRLTrainer
    from .visualization import plot_policy_heatmap, generate_all_visualizations
except ImportError:
    # Handle import errors gracefully during package installation
    pass

__all__ = [
    "generate_synthetic_icu_episode",
    "OfflineRLTrainer", 
    "plot_policy_heatmap",
    "generate_all_visualizations"
]