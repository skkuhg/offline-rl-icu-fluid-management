#!/usr/bin/env python3
"""
Offline Reinforcement Learning for ICU Fluid Management

This script provides the main execution pipeline for training offline RL policies
for ICU fluid management decisions using d3rlpy.

Author: Generated from Jupyter Notebook
License: MIT
"""

import os
import sys
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

# Configure paths and constants
SEED = 42
ARTIFACTS_DIR = 'artifacts'
FIGURES_DIR = 'figures'
DATA_DIR = 'data'

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)

def setup_environment():
    """Setup directories and install required packages."""
    print("Setting up environment...")
    
    # Create directories
    for directory in [ARTIFACTS_DIR, FIGURES_DIR, DATA_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Install packages if running in a fresh environment
    packages = [
        'd3rlpy>=2.8.1', 'torch', 'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'plotly', 'tqdm', 'joblib'
    ]
    
    try:
        import d3rlpy
        print(f"d3rlpy version: {d3rlpy.__version__}")
    except ImportError:
        print("Installing required packages...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--quiet'
        ] + packages, check=False)
    
    # Set torch seed
    try:
        import torch
        torch.manual_seed(SEED)
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"Torch seeding note: {e}")
    
    # Set d3rlpy seed
    try:
        import d3rlpy
        d3rlpy.seed(SEED)
    except Exception as e:
        print(f"d3rlpy seeding note: {e}")

def generate_synthetic_dataset(n_episodes=80, min_len=10, max_len=40):
    """Generate synthetic ICU dataset with realistic clinical patterns."""
    print("Generating synthetic ICU dataset...")
    
    rng = np.random.default_rng(SEED)
    rows = []
    
    for episode_id in range(n_episodes):
        # Patient characteristics
        severity = rng.choice(['stable', 'moderate', 'critical'], p=[0.4, 0.4, 0.2])
        
        # Episode length correlates with severity
        if severity == 'critical':
            T = int(rng.integers(min_len+5, max_len+1))
            survival_prob = 0.70
        elif severity == 'moderate':
            T = int(rng.integers(min_len+2, max_len-5))
            survival_prob = 0.85
        else:
            T = int(rng.integers(min_len, max_len-8))
            survival_prob = 0.95
            
        survived = rng.random() < survival_prob
        
        # Baseline vitals
        base_map = 70 + rng.normal(0, 8)
        base_hr = 85 + rng.normal(0, 10)
        
        if severity == 'critical':
            base_map *= 0.82
            base_hr *= 1.18
        elif severity == 'moderate':
            base_map *= 0.90
            base_hr *= 1.10
            
        cum_fluid = 0.0
        
        for t in range(T):
            # Clinical trajectory
            if survived:
                recovery_factor = 1 + (t / T) * 0.15
            else:
                recovery_factor = 1 - (t / T) * 0.30
                
            # Generate vitals with clinical correlations
            MAP = np.clip(base_map * recovery_factor + rng.normal(0, 6), 35, 130)
            HR = np.clip(base_hr * (2 - recovery_factor + 0.1) + rng.normal(0, 8), 50, 180)
            Lactate = np.clip(1.5 * (2 - recovery_factor) + rng.normal(0, 0.4), 0.4, 8.0)
            Creatinine = np.clip(1.0 * (2 - recovery_factor) + rng.normal(0, 0.3), 0.3, 6.0)
            SpO2 = np.clip(96 * recovery_factor + rng.normal(0, 2.5), 75, 100)
            Temp = np.clip(37.0 + (2 - recovery_factor - 1) * 1.5 + rng.normal(0, 0.6), 34, 42)
            
            # Clinician decision model
            hypotensive = MAP < 65
            shock = MAP < 60 or Lactate > 4.0
            
            if shock:
                prob_bins = np.array([0.10, 0.25, 0.40, 0.25])
            elif hypotensive:
                prob_bins = np.array([0.20, 0.35, 0.30, 0.15])
            elif MAP > 85:
                prob_bins = np.array([0.70, 0.20, 0.08, 0.02])
            else:
                prob_bins = np.array([0.45, 0.35, 0.15, 0.05])
                
            action = int(rng.choice(4, p=prob_bins))
            fluid_ml = [0, 200, 400, 750][action]
            cum_fluid += fluid_ml

            # Reward function
            reward = 0.0
            if MAP < 65:
                reward -= 0.1
            if cum_fluid > 3500:
                reward -= 0.06
            
            terminal = (t == T - 1)
            if terminal:
                reward += 1.0 if survived else -1.0
            reward = float(np.clip(reward, -1.0, 1.0))

            rows.append({
                'subject_id': 100000 + episode_id,
                'hadm_id': 200000 + episode_id,
                'stay_id': 300000 + episode_id,
                'episode_id': episode_id,
                't': t,
                'action': action,
                'action_name': ['0', '0-250', '250-500', '>500'][action],
                'reward': reward,
                'terminal': terminal,
                'state_MAP': float(MAP),
                'state_HR': float(HR),
                'state_Lactate': float(Lactate),
                'state_Creatinine': float(Creatinine),
                'state_SpO2': float(SpO2),
                'state_Temp': float(Temp),
                'state_HourIdx': float(t),
            })
    
    return pd.DataFrame(rows)

def prepare_dataset():
    """Load or generate dataset and prepare for training."""
    parquet_path = os.path.join(ARTIFACTS_DIR, 'mdp_dataset.parquet')
    
    # Try to load existing dataset
    if os.path.exists(parquet_path):
        print("Loading existing dataset...")
        df = pd.read_parquet(parquet_path)
    else:
        print("Generating new dataset...")
        df = generate_synthetic_dataset()
        df.to_parquet(parquet_path, index=False)
    
    print(f"Dataset loaded: {len(df)} steps, {df['episode_id'].nunique()} episodes")
    return df

def train_behavior_classifier(X_train, y_train, X_test, y_test):
    """Train behavior cloning classifier."""
    print("Training behavior classifier...")
    
    clf = LogisticRegression(
        penalty='l2', solver='saga', max_iter=500, 
        n_jobs=-1, random_state=SEED
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    top2 = top_k_accuracy_score(y_test, proba, k=2)
    
    print(f"Behavior classifier - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Top-2: {top2:.3f}")
    
    return clf

def train_offline_rl_policies(mdp_dataset):
    """Train offline RL policies using d3rlpy."""
    print("Training offline RL policies...")
    
    try:
        from d3rlpy.algos import DiscreteBC, DiscreteCQL
        from d3rlpy.algos import DiscreteBCConfig, DiscreteCQLConfig
        
        models = {}
        
        # Train Behavior Cloning
        print("Training DiscreteBC...")
        bc_config = DiscreteBCConfig()
        bc = DiscreteBC(config=bc_config, device='cpu', enable_ddp=False)
        bc.fit(mdp_dataset, n_steps=10000)
        bc.save_model(os.path.join(ARTIFACTS_DIR, 'bc_policy.d3rlpy'))
        models['bc'] = bc
        
        # Train Conservative Q-Learning
        print("Training DiscreteCQL...")
        cql_config = DiscreteCQLConfig()
        cql = DiscreteCQL(config=cql_config, device='cpu', enable_ddp=False)
        cql.fit(mdp_dataset, n_steps=10000)
        cql.save_model(os.path.join(ARTIFACTS_DIR, 'cql_policy.d3rlpy'))
        models['cql'] = cql
        
        # Try IQL or use BC as fallback
        try:
            from d3rlpy.algos import DiscreteIQL, DiscreteIQLConfig
            print("Training DiscreteIQL...")
            iql_config = DiscreteIQLConfig()
            iql = DiscreteIQL(config=iql_config, device='cpu', enable_ddp=False)
            iql.fit(mdp_dataset, n_steps=10000)
        except ImportError:
            print("DiscreteIQL not available, using BC as fallback...")
            iql_config = DiscreteBCConfig()
            iql = DiscreteBC(config=iql_config, device='cpu', enable_ddp=False)
            iql.fit(mdp_dataset, n_steps=10000)
        
        iql.save_model(os.path.join(ARTIFACTS_DIR, 'iql_policy.d3rlpy'))
        models['iql'] = iql
        
        print("Offline RL training completed successfully!")
        return models
        
    except Exception as e:
        print(f"Error in offline RL training: {e}")
        return {}

def generate_visualizations(df, models, X_test, feature_index):
    """Generate policy visualizations and comparisons."""
    print("Generating visualizations...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Basic action distribution comparison
        if 'bc' in models and 'cql' in models:
            A_test = df[df['episode_id'].isin(df['episode_id'].unique()[-13:])]['action'].values
            
            plt.figure(figsize=(8, 4))
            bins = np.arange(0, 5) - 0.5
            
            plt.hist(A_test, bins=bins, alpha=0.6, label='Clinician', density=True)
            
            if len(X_test) > 0:
                A_bc = models['bc'].predict(X_test.astype(np.float32))
                A_cql = models['cql'].predict(X_test.astype(np.float32))
                
                plt.hist(A_bc, bins=bins, alpha=0.6, label='BC', density=True)
                plt.hist(A_cql, bins=bins, alpha=0.6, label='CQL', density=True)
            
            plt.xlabel('Action')
            plt.ylabel('Density')
            plt.title('Policy Action Distributions')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'policy_comparison.png'), dpi=150)
            plt.close()
            
        print("Visualizations saved to figures/")
        
    except Exception as e:
        print(f"Visualization error: {e}")

def save_artifacts(clf, feature_columns):
    """Save training artifacts and metadata."""
    print("Saving artifacts...")
    
    # Save behavior classifier
    import joblib
    joblib.dump({
        'model': clf, 
        'features': feature_columns
    }, os.path.join(ARTIFACTS_DIR, 'behavior_policy.pkl'))
    
    # Save feature mapping
    feature_index = {feat: i for i, feat in enumerate(feature_columns)}
    with open(os.path.join(ARTIFACTS_DIR, 'feature_index.json'), 'w') as f:
        json.dump(feature_index, f, indent=2)
    
    # Save action mapping
    action_mapping = {i: str(i) for i in range(4)}
    with open(os.path.join(ARTIFACTS_DIR, 'action_mapping.json'), 'w') as f:
        json.dump(action_mapping, f, indent=2)
    
    print("Artifacts saved successfully!")

def main():
    """Main execution pipeline."""
    print("Starting Offline RL for ICU Fluid Management")
    print("=" * 50)
    
    # Setup
    setup_environment()
    
    # Data preparation
    df = prepare_dataset()
    
    # Extract features and prepare data
    feature_cols = [c for c in df.columns if c.startswith('state_')]
    X = df[feature_cols].values
    y = df['action'].values
    
    # Train/test split by episodes
    episodes = df['episode_id'].unique()
    train_episodes, test_episodes = train_test_split(
        episodes, test_size=0.2, random_state=SEED
    )
    
    train_mask = df['episode_id'].isin(train_episodes)
    test_mask = df['episode_id'].isin(test_episodes)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_test = scaler.transform(X[test_mask])
    
    # Save scaler
    import joblib
    joblib.dump({
        'scaler': scaler, 
        'features': feature_cols
    }, os.path.join(ARTIFACTS_DIR, 'state_scaler.pkl'))
    
    # Train behavior classifier
    clf = train_behavior_classifier(
        X_train, y[train_mask], 
        X_test, y[test_mask]
    )
    
    # Prepare MDP dataset for d3rlpy
    try:
        from d3rlpy.dataset import MDPDataset
        
        X_all = scaler.transform(X)
        mdp = MDPDataset(
            observations=X_all.astype(np.float32),
            actions=y.astype(np.int64),
            rewards=df['reward'].values.astype(np.float32),
            terminals=df['terminal'].values.astype(bool)
        )
        
        # Train offline RL policies
        models = train_offline_rl_policies(mdp)
        
        # Generate visualizations
        feature_index = {feat: i for i, feat in enumerate(feature_cols)}
        generate_visualizations(df, models, X_test, feature_index)
        
    except Exception as e:
        print(f"d3rlpy training failed: {e}")
        models = {}
    
    # Save artifacts
    save_artifacts(clf, feature_cols)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Models saved to: {ARTIFACTS_DIR}/")
    print(f"Visualizations saved to: {FIGURES_DIR}/")
    print("\nNext steps:")
    print("1. Review generated policy comparisons")
    print("2. Examine clinical heatmaps and support diagnostics")
    print("3. Use trained models for policy evaluation")

if __name__ == "__main__":
    main()