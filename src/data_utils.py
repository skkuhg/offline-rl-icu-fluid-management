"""
Data utilities for ICU fluid management offline RL.

This module provides functions for generating synthetic ICU datasets
with realistic clinical patterns and trajectories.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def generate_synthetic_icu_episode(
    episode_id: int,
    severity: str = "moderate",
    min_len: int = 10,
    max_len: int = 40,
    seed: Optional[int] = None
) -> List[Dict]:
    """
    Generate a single synthetic ICU episode with realistic clinical patterns.
    
    Args:
        episode_id: Unique identifier for the episode
        severity: Patient severity level ('stable', 'moderate', 'critical')
        min_len: Minimum episode length in hours
        max_len: Maximum episode length in hours
        seed: Random seed for reproducibility
        
    Returns:
        List of dictionaries containing patient states, actions, and rewards
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Determine episode characteristics based on severity
    severity_params = {
        'stable': {'length_bonus': 0, 'survival_prob': 0.95, 'map_factor': 1.0, 'hr_factor': 1.0},
        'moderate': {'length_bonus': 2, 'survival_prob': 0.85, 'map_factor': 0.90, 'hr_factor': 1.10},
        'critical': {'length_bonus': 5, 'survival_prob': 0.70, 'map_factor': 0.82, 'hr_factor': 1.18}
    }
    
    params = severity_params.get(severity, severity_params['moderate'])
    
    # Episode length correlates with severity
    T = int(rng.integers(
        min_len + params['length_bonus'], 
        max_len - (8 if severity == 'stable' else 0) + 1
    ))
    
    survived = rng.random() < params['survival_prob']
    
    # Baseline vitals with individual variation
    base_map = (70 + rng.normal(0, 8)) * params['map_factor']
    base_hr = (85 + rng.normal(0, 10)) * params['hr_factor']
    
    episode_data = []
    cumulative_fluid = 0.0
    
    for t in range(T):
        # Clinical trajectory modeling
        if survived:
            recovery_factor = 1 + (t / T) * 0.15  # Gradual improvement
        else:
            recovery_factor = 1 - (t / T) * 0.30  # Gradual deterioration
        
        # Generate correlated vital signs
        vitals = generate_correlated_vitals(
            base_map, base_hr, recovery_factor, rng
        )
        
        # Determine clinician action based on clinical decision rules
        action = determine_clinician_action(vitals, rng)
        
        # Calculate fluid administration
        fluid_amounts = [0, 200, 400, 750]  # mL per action
        fluid_given = fluid_amounts[action]
        cumulative_fluid += fluid_given
        
        # Calculate reward
        reward = calculate_reward(
            vitals['MAP'], cumulative_fluid, t == T - 1, survived
        )
        
        # Store episode step
        episode_data.append({
            'subject_id': 100000 + episode_id,
            'hadm_id': 200000 + episode_id,
            'stay_id': 300000 + episode_id,
            'episode_id': episode_id,
            't': t,
            'action': action,
            'action_name': ['0', '0-250', '250-500', '>500'][action],
            'reward': reward,
            'terminal': (t == T - 1),
            'cumulative_fluid': cumulative_fluid,
            **{f'state_{key}': float(value) for key, value in vitals.items()}
        })
    
    return episode_data


def generate_correlated_vitals(
    base_map: float, 
    base_hr: float, 
    recovery_factor: float, 
    rng: np.random.Generator
) -> Dict[str, float]:
    """
    Generate correlated vital signs with realistic clinical relationships.
    
    Args:
        base_map: Baseline mean arterial pressure
        base_hr: Baseline heart rate
        recovery_factor: Recovery/deterioration factor (1.0 = stable)
        rng: Random number generator
        
    Returns:
        Dictionary of vital signs with clinical correlations
    """
    # Primary vitals with recovery trajectory
    MAP = np.clip(
        base_map * recovery_factor + rng.normal(0, 6), 
        35, 130
    )
    
    HR = np.clip(
        base_hr * (2 - recovery_factor + 0.1) + rng.normal(0, 8), 
        50, 180
    )
    
    # Laboratory values correlated with recovery
    Lactate = np.clip(
        1.5 * (2 - recovery_factor) + rng.normal(0, 0.4), 
        0.4, 8.0
    )
    
    Creatinine = np.clip(
        1.0 * (2 - recovery_factor) + rng.normal(0, 0.3), 
        0.3, 6.0
    )
    
    # Oxygen saturation and temperature
    SpO2 = np.clip(
        96 * recovery_factor + rng.normal(0, 2.5), 
        75, 100
    )
    
    Temp = np.clip(
        37.0 + (2 - recovery_factor - 1) * 1.5 + rng.normal(0, 0.6), 
        34, 42
    )
    
    return {
        'MAP': MAP,
        'HR': HR,
        'Lactate': Lactate,
        'Creatinine': Creatinine,
        'SpO2': SpO2,
        'Temp': Temp,
        'HourIdx': float(rng.integers(0, 24))  # Hour of day
    }


def determine_clinician_action(
    vitals: Dict[str, float], 
    rng: np.random.Generator
) -> int:
    """
    Determine clinician fluid management action based on clinical decision rules.
    
    Args:
        vitals: Dictionary of patient vital signs
        rng: Random number generator
        
    Returns:
        Action index (0: no fluid, 1: 0-250mL, 2: 250-500mL, 3: >500mL)
    """
    MAP = vitals['MAP']
    Lactate = vitals['Lactate']
    
    # Clinical decision logic
    shock = MAP < 60 or Lactate > 4.0
    hypotensive = MAP < 65
    hypertensive = MAP > 85
    
    # Action probabilities based on clinical guidelines
    if shock:
        # Aggressive fluid resuscitation for shock
        prob_bins = np.array([0.10, 0.25, 0.40, 0.25])
    elif hypotensive:
        # Moderate fluid administration for hypotension
        prob_bins = np.array([0.20, 0.35, 0.30, 0.15])
    elif hypertensive:
        # Conservative approach for high MAP
        prob_bins = np.array([0.70, 0.20, 0.08, 0.02])
    else:
        # Maintenance for normal MAP
        prob_bins = np.array([0.45, 0.35, 0.15, 0.05])
    
    return int(rng.choice(4, p=prob_bins))


def calculate_reward(
    MAP: float, 
    cumulative_fluid: float, 
    is_terminal: bool, 
    survived: bool
) -> float:
    """
    Calculate reward for the current state and action.
    
    Args:
        MAP: Mean arterial pressure
        cumulative_fluid: Total fluid given (mL)
        is_terminal: Whether this is the final step
        survived: Whether patient survived (for terminal reward)
        
    Returns:
        Reward value clipped to [-1.0, 1.0]
    """
    reward = 0.0
    
    # Penalty for hypotension
    if MAP < 65:
        reward -= 0.1
    
    # Penalty for fluid overload
    if cumulative_fluid > 3500:
        reward -= 0.06
    
    # Terminal reward based on survival
    if is_terminal:
        reward += 1.0 if survived else -1.0
    
    return float(np.clip(reward, -1.0, 1.0))


def generate_full_dataset(
    n_episodes: int = 80,
    min_len: int = 10,
    max_len: int = 40,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a complete synthetic ICU dataset.
    
    Args:
        n_episodes: Number of ICU episodes to generate
        min_len: Minimum episode length in hours
        max_len: Maximum episode length in hours
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing all episode data
    """
    rng = np.random.default_rng(seed)
    all_data = []
    
    for episode_id in range(n_episodes):
        # Sample severity distribution
        severity = rng.choice(
            ['stable', 'moderate', 'critical'], 
            p=[0.4, 0.4, 0.2]
        )
        
        episode_data = generate_synthetic_icu_episode(
            episode_id=episode_id,
            severity=severity,
            min_len=min_len,
            max_len=max_len,
            seed=seed + episode_id  # Unique seed per episode
        )
        
        all_data.extend(episode_data)
    
    return pd.DataFrame(all_data)


def preprocess_dataset(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Preprocess dataset for offline RL training.
    
    Args:
        df: Raw dataset DataFrame
        feature_columns: List of feature column names (auto-detect if None)
        
    Returns:
        Tuple of (features, actions, feature_column_names)
    """
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col.startswith('state_')]
    
    X = df[feature_columns].values.astype(np.float32)
    y = df['action'].values.astype(np.int64)
    
    return X, y, feature_columns


def validate_dataset(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate dataset quality and return summary statistics.
    
    Args:
        df: Dataset DataFrame
        
    Returns:
        Dictionary of validation metrics and statistics
    """
    validation_results = {
        'total_steps': len(df),
        'num_episodes': df['episode_id'].nunique(),
        'avg_episode_length': df.groupby('episode_id').size().mean(),
        'action_distribution': df['action'].value_counts().to_dict(),
        'survival_rate': df[df['terminal']]['reward'].apply(lambda x: x > 0).mean(),
        'feature_ranges': {}
    }
    
    # Feature range validation
    feature_cols = [col for col in df.columns if col.startswith('state_')]
    for col in feature_cols:
        validation_results['feature_ranges'][col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'std': float(df[col].std())
        }
    
    return validation_results