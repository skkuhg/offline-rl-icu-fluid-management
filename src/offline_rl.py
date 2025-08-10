"""
Offline RL training utilities for ICU fluid management.

This module provides the OfflineRLTrainer class for training BC, CQL, and IQL models
using d3rlpy with fallback strategies and model management.
"""

import os
import json
import warnings
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class OfflineRLTrainer:
    """
    Trainer class for offline RL algorithms applied to ICU fluid management.
    
    Supports Behavior Cloning (BC), Conservative Q-Learning (CQL), and 
    Implicit Q-Learning (IQL) using d3rlpy library.
    """
    
    def __init__(
        self, 
        artifacts_dir: str = "artifacts",
        device: str = "cpu",
        seed: int = 42
    ):
        """
        Initialize the OfflineRLTrainer.
        
        Args:
            artifacts_dir: Directory to save trained models and artifacts
            device: Device for training ('cpu' or 'cuda')
            seed: Random seed for reproducibility
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)
        self.device = device
        self.seed = seed
        
        # Model storage
        self.models = {}
        self.scaler = None
        self.feature_columns = None
        
        # Set seeds
        np.random.seed(seed)
        self._set_torch_seed()
        self._set_d3rlpy_seed()
    
    def _set_torch_seed(self):
        """Set PyTorch random seeds if available."""
        try:
            import torch
            torch.manual_seed(self.seed)
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True)
        except ImportError:
            pass
    
    def _set_d3rlpy_seed(self):
        """Set d3rlpy random seed if available."""
        try:
            import d3rlpy
            d3rlpy.seed(self.seed)
        except ImportError:
            pass
    
    def prepare_data(
        self, 
        df: pd.DataFrame,
        test_size: float = 0.2,
        scale_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare dataset for training by extracting features and splitting data.
        
        Args:
            df: Dataset DataFrame
            test_size: Fraction of data for testing
            scale_features: Whether to standardize features
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Extract feature columns
        self.feature_columns = [col for col in df.columns if col.startswith('state_')]
        X = df[self.feature_columns].values
        y = df['action'].values
        
        # Split by episodes to avoid data leakage
        episodes = df['episode_id'].unique()
        train_episodes, test_episodes = train_test_split(
            episodes, test_size=test_size, random_state=self.seed
        )
        
        train_mask = df['episode_id'].isin(train_episodes)
        test_mask = df['episode_id'].isin(test_episodes)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Feature scaling
        if scale_features:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            # Save scaler
            scaler_path = self.artifacts_dir / "state_scaler.pkl"
            joblib.dump({
                'scaler': self.scaler,
                'features': self.feature_columns
            }, scaler_path)
        
        return X_train, X_test, y_train, y_test
    
    def create_mdp_dataset(self, df: pd.DataFrame, X: np.ndarray) -> Any:
        """
        Create d3rlpy MDPDataset from preprocessed data.
        
        Args:
            df: Original DataFrame with episodes and rewards
            X: Preprocessed feature matrix
            
        Returns:
            d3rlpy MDPDataset object
        """
        try:
            from d3rlpy.dataset import MDPDataset
            
            mdp_dataset = MDPDataset(
                observations=X.astype(np.float32),
                actions=df['action'].values.astype(np.int64),
                rewards=df['reward'].values.astype(np.float32),
                terminals=df['terminal'].values.astype(bool)
            )
            
            return mdp_dataset
            
        except ImportError as e:
            raise ImportError(f"d3rlpy not available: {e}")
    
    def train_behavior_cloning(
        self, 
        mdp_dataset: Any, 
        n_steps: int = 10000,
        **kwargs
    ) -> Any:
        """
        Train Behavior Cloning (BC) model.
        
        Args:
            mdp_dataset: d3rlpy MDPDataset
            n_steps: Number of training steps
            **kwargs: Additional arguments for BC configuration
            
        Returns:
            Trained BC model
        """
        try:
            from d3rlpy.algos import DiscreteBC, DiscreteBCConfig
            
            print("Training Behavior Cloning (BC)...")
            
            config = DiscreteBCConfig(**kwargs)
            bc = DiscreteBC(
                config=config, 
                device=self.device, 
                enable_ddp=False
            )
            
            bc.fit(mdp_dataset, n_steps=n_steps)
            
            # Save model
            model_path = self.artifacts_dir / "bc_policy.d3rlpy"
            bc.save_model(str(model_path))
            
            self.models['bc'] = bc
            return bc
            
        except Exception as e:
            print(f"BC training failed: {e}")
            return None
    
    def train_conservative_ql(
        self, 
        mdp_dataset: Any, 
        n_steps: int = 10000,
        **kwargs
    ) -> Any:
        """
        Train Conservative Q-Learning (CQL) model.
        
        Args:
            mdp_dataset: d3rlpy MDPDataset
            n_steps: Number of training steps
            **kwargs: Additional arguments for CQL configuration
            
        Returns:
            Trained CQL model
        """
        try:
            from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig
            
            print("Training Conservative Q-Learning (CQL)...")
            
            config = DiscreteCQLConfig(**kwargs)
            cql = DiscreteCQL(
                config=config, 
                device=self.device, 
                enable_ddp=False
            )
            
            cql.fit(mdp_dataset, n_steps=n_steps)
            
            # Save model
            model_path = self.artifacts_dir / "cql_policy.d3rlpy"
            cql.save_model(str(model_path))
            
            self.models['cql'] = cql
            return cql
            
        except Exception as e:
            print(f"CQL training failed: {e}")
            return None
    
    def train_implicit_ql(
        self, 
        mdp_dataset: Any, 
        n_steps: int = 10000,
        use_bc_fallback: bool = True,
        **kwargs
    ) -> Any:
        """
        Train Implicit Q-Learning (IQL) model with BC fallback.
        
        Args:
            mdp_dataset: d3rlpy MDPDataset
            n_steps: Number of training steps
            use_bc_fallback: Use BC if IQL is not available
            **kwargs: Additional arguments for IQL configuration
            
        Returns:
            Trained IQL model (or BC if fallback used)
        """
        try:
            from d3rlpy.algos import DiscreteIQL, DiscreteIQLConfig
            
            print("Training Implicit Q-Learning (IQL)...")
            
            config = DiscreteIQLConfig(**kwargs)
            iql = DiscreteIQL(
                config=config, 
                device=self.device, 
                enable_ddp=False
            )
            
            iql.fit(mdp_dataset, n_steps=n_steps)
            
            # Save model
            model_path = self.artifacts_dir / "iql_policy.d3rlpy"
            iql.save_model(str(model_path))
            
            self.models['iql'] = iql
            return iql
            
        except ImportError:
            if use_bc_fallback:
                print("IQL not available, using BC as fallback...")
                return self.train_behavior_cloning(mdp_dataset, n_steps, **kwargs)
            else:
                print("IQL not available and fallback disabled")
                return None
        except Exception as e:
            print(f"IQL training failed: {e}")
            return None
    
    def train_all_algorithms(
        self, 
        df: pd.DataFrame,
        n_steps: int = 10000,
        algorithms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train all specified offline RL algorithms.
        
        Args:
            df: Dataset DataFrame
            n_steps: Number of training steps per algorithm
            algorithms: List of algorithms to train ['bc', 'cql', 'iql']
            
        Returns:
            Dictionary of trained models
        """
        if algorithms is None:
            algorithms = ['bc', 'cql', 'iql']
        
        print(f"Training offline RL algorithms: {algorithms}")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Create full dataset for d3rlpy
        X_full = self.scaler.transform(df[self.feature_columns].values) if self.scaler else df[self.feature_columns].values
        mdp_dataset = self.create_mdp_dataset(df, X_full)
        
        # Train algorithms
        trained_models = {}
        
        if 'bc' in algorithms:
            bc_model = self.train_behavior_cloning(mdp_dataset, n_steps)
            if bc_model:
                trained_models['bc'] = bc_model
        
        if 'cql' in algorithms:
            cql_model = self.train_conservative_ql(mdp_dataset, n_steps)
            if cql_model:
                trained_models['cql'] = cql_model
        
        if 'iql' in algorithms:
            iql_model = self.train_implicit_ql(mdp_dataset, n_steps)
            if iql_model:
                trained_models['iql'] = iql_model
        
        self.models.update(trained_models)
        return trained_models
    
    def evaluate_models(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test actions
            
        Returns:
            Dictionary of evaluation metrics per model
        """
        from sklearn.metrics import accuracy_score, f1_score
        
        results = {}
        
        for model_name, model in self.models.items():
            try:
                y_pred = model.predict(X_test.astype(np.float32))
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro')
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'f1_macro': f1
                }
                
                print(f"{model_name.upper()} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
                
            except Exception as e:
                print(f"Evaluation failed for {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def create_prediction_helper(self) -> None:
        """
        Create a standalone prediction helper script.
        """
        helper_code = '''
"""Prediction helper for trained offline RL models."""

import os
import json
import numpy as np
import joblib

def load_models(artifacts_dir="artifacts"):
    """Load all trained models and preprocessing components."""
    models = {}
    
    # Load scaler
    scaler_path = os.path.join(artifacts_dir, "state_scaler.pkl")
    if os.path.exists(scaler_path):
        scaler_data = joblib.load(scaler_path)
        scaler = scaler_data['scaler']
        features = scaler_data['features']
    else:
        print("Warning: Scaler not found")
        scaler, features = None, None
    
    # Load d3rlpy models
    for model_name in ['bc', 'cql', 'iql']:
        model_path = os.path.join(artifacts_dir, f"{model_name}_policy.d3rlpy")
        if os.path.exists(model_path):
            try:
                if model_name == 'bc':
                    from d3rlpy.algos import DiscreteBC
                    model = DiscreteBC.from_pretrained(model_path)
                elif model_name == 'cql':
                    from d3rlpy.algos import DiscreteCQL
                    model = DiscreteCQL.from_pretrained(model_path)
                elif model_name == 'iql':
                    from d3rlpy.algos import DiscreteIQL
                    model = DiscreteIQL.from_pretrained(model_path)
                    
                models[model_name] = model
                print(f"Loaded {model_name.upper()} model")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
    
    return models, scaler, features

def predict_action(state_dict, models, scaler, features, model_name='bc'):
    """Predict action for a given state."""
    if model_name not in models:
        raise ValueError(f"Model {model_name} not available")
    
    # Prepare state vector
    state_vector = np.array([state_dict[f'state_{feat.split("_", 1)[1]}'] for feat in features])
    
    if scaler:
        state_vector = scaler.transform(state_vector.reshape(1, -1))
    else:
        state_vector = state_vector.reshape(1, -1)
    
    # Predict
    action = models[model_name].predict(state_vector.astype(np.float32))[0]
    return int(action)

def predict_all_models(state_dict, models, scaler, features):
    """Get predictions from all available models."""
    predictions = {}
    for model_name in models.keys():
        try:
            predictions[model_name] = predict_action(
                state_dict, models, scaler, features, model_name
            )
        except Exception as e:
            predictions[model_name] = f"Error: {e}"
    
    return predictions

if __name__ == "__main__":
    # Example usage
    models, scaler, features = load_models()
    
    # Example state
    example_state = {
        'state_MAP': 75.0,
        'state_HR': 95.0,
        'state_Lactate': 2.1,
        'state_Creatinine': 1.2,
        'state_SpO2': 96.0,
        'state_Temp': 37.1,
        'state_HourIdx': 8.0
    }
    
    predictions = predict_all_models(example_state, models, scaler, features)
    print("Predictions:", predictions)
'''
        
        helper_path = self.artifacts_dir / "predict.py"
        with open(helper_path, 'w') as f:
            f.write(helper_code)
        
        print(f"Prediction helper saved to {helper_path}")
    
    def save_metadata(self, validation_results: Dict) -> None:
        """
        Save training metadata and model information.
        
        Args:
            validation_results: Dataset validation results
        """
        metadata = {
            'models_trained': list(self.models.keys()),
            'feature_columns': self.feature_columns,
            'artifacts_dir': str(self.artifacts_dir),
            'device': self.device,
            'seed': self.seed,
            'dataset_info': validation_results
        }
        
        metadata_path = self.artifacts_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Training metadata saved to {metadata_path}")


def load_pretrained_models(artifacts_dir: str = "artifacts") -> Dict[str, Any]:
    """
    Load pretrained models from artifacts directory.
    
    Args:
        artifacts_dir: Directory containing saved models
        
    Returns:
        Dictionary of loaded models
    """
    artifacts_path = Path(artifacts_dir)
    models = {}
    
    for model_name in ['bc', 'cql', 'iql']:
        model_path = artifacts_path / f"{model_name}_policy.d3rlpy"
        if model_path.exists():
            try:
                if model_name == 'bc':
                    from d3rlpy.algos import DiscreteBC
                    model = DiscreteBC.from_pretrained(str(model_path))
                elif model_name == 'cql':
                    from d3rlpy.algos import DiscreteCQL
                    model = DiscreteCQL.from_pretrained(str(model_path))
                elif model_name == 'iql':
                    from d3rlpy.algos import DiscreteIQL
                    model = DiscreteIQL.from_pretrained(str(model_path))
                
                models[model_name] = model
                print(f"Loaded {model_name.upper()} model from {model_path}")
                
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
    
    return models