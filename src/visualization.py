"""
Visualization utilities for offline RL ICU fluid management analysis.

This module provides comprehensive plotting functions for policy analysis,
clinical correlations, and model comparisons.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# Optional plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class PolicyVisualizer:
    """
    Comprehensive visualization suite for offline RL policy analysis.
    """
    
    def __init__(self, figures_dir: str = "figures"):
        """
        Initialize the PolicyVisualizer.
        
        Args:
            figures_dir: Directory to save generated figures
        """
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_policy_heatmap(
        self, 
        models: Dict[str, Any], 
        feature_ranges: Dict[str, Dict], 
        primary_features: Tuple[str, str] = ('state_MAP', 'state_Lactate'),
        resolution: int = 50
    ) -> None:
        """
        Generate policy heatmaps showing action distributions across state space.
        
        Args:
            models: Dictionary of trained models
            feature_ranges: Feature value ranges from dataset
            primary_features: Two features to plot on x and y axes
            resolution: Grid resolution for heatmap
        """
        if not models:
            print("No models available for heatmap generation")
            return
        
        feature_x, feature_y = primary_features
        
        # Create state grid
        x_range = np.linspace(
            feature_ranges[feature_x]['min'],
            feature_ranges[feature_x]['max'],
            resolution
        )
        y_range = np.linspace(
            feature_ranges[feature_y]['min'],
            feature_ranges[feature_y]['max'],
            resolution
        )
        
        XX, YY = np.meshgrid(x_range, y_range)
        
        # Create figure
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(models.items()):
            try:
                # Create state vectors for prediction
                grid_states = []
                for i in range(resolution):
                    for j in range(resolution):
                        state = np.zeros(len(feature_ranges))
                        # Set primary features
                        state[0] = XX[i, j]  # Assuming first feature is MAP
                        state[1] = YY[i, j]  # Assuming second feature is Lactate
                        # Set other features to mean values
                        for k, (feat, ranges) in enumerate(feature_ranges.items()):
                            if k > 1:  # Skip first two features
                                state[k] = ranges['mean']
                        grid_states.append(state)
                
                grid_states = np.array(grid_states)
                
                # Predict actions
                actions = model.predict(grid_states.astype(np.float32))
                action_grid = actions.reshape(resolution, resolution)
                
                # Plot heatmap
                im = axes[idx].imshow(
                    action_grid, 
                    extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
                    origin='lower',
                    aspect='auto',
                    cmap='viridis',
                    alpha=0.8
                )
                
                axes[idx].set_xlabel(feature_x.replace('state_', ''))
                axes[idx].set_ylabel(feature_y.replace('state_', ''))
                axes[idx].set_title(f'{model_name.upper()} Policy')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[idx])
                cbar.set_label('Action')
                
                # Add clinical decision boundaries
                if feature_x == 'state_MAP':
                    axes[idx].axvline(x=65, color='red', linestyle='--', alpha=0.7, label='Hypotension')
                    axes[idx].axvline(x=85, color='orange', linestyle='--', alpha=0.7, label='High MAP')
                
                if feature_y == 'state_Lactate':
                    axes[idx].axhline(y=4.0, color='red', linestyle='--', alpha=0.7, label='High Lactate')
                
                axes[idx].legend()
                
            except Exception as e:
                print(f"Error generating heatmap for {model_name}: {e}")
                axes[idx].text(0.5, 0.5, f'Error: {str(e)[:50]}...', 
                             transform=axes[idx].transAxes, ha='center')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'policy_heatmaps.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Policy heatmaps saved to {self.figures_dir / 'policy_heatmaps.png'}")
    
    def plot_action_distributions(
        self, 
        df: pd.DataFrame, 
        models: Optional[Dict[str, Any]] = None,
        X_test: Optional[np.ndarray] = None
    ) -> None:
        """
        Plot action distribution comparisons between clinicians and models.
        
        Args:
            df: Dataset DataFrame
            models: Dictionary of trained models
            X_test: Test features for model predictions
        """
        plt.figure(figsize=(10, 6))
        
        # Clinician actions
        clinician_actions = df['action'].values
        bins = np.arange(0, 5) - 0.5
        
        plt.hist(clinician_actions, bins=bins, alpha=0.6, 
                label='Clinician', density=True, color=self.colors[0])
        
        # Model predictions
        if models and X_test is not None:
            for idx, (model_name, model) in enumerate(models.items()):
                try:
                    model_actions = model.predict(X_test.astype(np.float32))
                    plt.hist(model_actions, bins=bins, alpha=0.6, 
                            label=model_name.upper(), density=True, 
                            color=self.colors[idx + 1])
                except Exception as e:
                    print(f"Error plotting {model_name} actions: {e}")
        
        plt.xlabel('Action (Fluid Category)')
        plt.ylabel('Density')
        plt.title('Action Distribution Comparison')
        plt.legend()
        plt.xticks([0, 1, 2, 3], ['0 mL', '0-250 mL', '250-500 mL', '>500 mL'])
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'action_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Action distributions saved to {self.figures_dir / 'action_distributions.png'}")
    
    def plot_clinical_correlations(
        self, 
        df: pd.DataFrame
    ) -> None:
        """
        Plot clinical correlations between vitals and actions.
        
        Args:
            df: Dataset DataFrame
        """
        # Vital signs correlations
        vital_cols = ['state_MAP', 'state_HR', 'state_Lactate', 'state_Creatinine']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, vital in enumerate(vital_cols):
            for action in range(4):
                action_data = df[df['action'] == action][vital]
                axes[idx].hist(action_data, alpha=0.6, label=f'Action {action}', 
                             bins=20, density=True)
            
            axes[idx].set_xlabel(vital.replace('state_', ''))
            axes[idx].set_ylabel('Density')
            axes[idx].set_title(f'{vital.replace("state_", "")} Distribution by Action')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'clinical_correlations.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Clinical correlations saved to {self.figures_dir / 'clinical_correlations.png'}")
    
    def plot_episode_trajectories(
        self, 
        df: pd.DataFrame, 
        episode_ids: Optional[List[int]] = None,
        max_episodes: int = 5
    ) -> None:
        """
        Plot trajectories for selected episodes.
        
        Args:
            df: Dataset DataFrame
            episode_ids: Specific episodes to plot (random selection if None)
            max_episodes: Maximum number of episodes to plot
        """
        if episode_ids is None:
            episode_ids = np.random.choice(
                df['episode_id'].unique(), 
                min(max_episodes, df['episode_id'].nunique()), 
                replace=False
            )
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot MAP and HR trajectories
        for episode_id in episode_ids:
            episode_data = df[df['episode_id'] == episode_id].sort_values('t')
            
            axes[0, 0].plot(episode_data['t'], episode_data['state_MAP'], 
                           label=f'Episode {episode_id}', alpha=0.7)
            axes[0, 1].plot(episode_data['t'], episode_data['state_HR'], 
                           label=f'Episode {episode_id}', alpha=0.7)
            axes[1, 0].plot(episode_data['t'], episode_data['state_Lactate'], 
                           label=f'Episode {episode_id}', alpha=0.7)
            axes[1, 1].scatter(episode_data['t'], episode_data['action'], 
                              label=f'Episode {episode_id}', alpha=0.7)
        
        # Add clinical thresholds
        axes[0, 0].axhline(y=65, color='red', linestyle='--', alpha=0.5, label='Hypotension')
        axes[1, 0].axhline(y=4.0, color='red', linestyle='--', alpha=0.5, label='High Lactate')
        
        axes[0, 0].set_title('MAP Trajectories')
        axes[0, 0].set_ylabel('MAP (mmHg)')
        axes[0, 1].set_title('Heart Rate Trajectories')
        axes[0, 1].set_ylabel('HR (bpm)')
        axes[1, 0].set_title('Lactate Trajectories')
        axes[1, 0].set_ylabel('Lactate (mmol/L)')
        axes[1, 1].set_title('Action Patterns')
        axes[1, 1].set_ylabel('Action')
        
        for ax in axes.ravel():
            ax.set_xlabel('Time (hours)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'episode_trajectories.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Episode trajectories saved to {self.figures_dir / 'episode_trajectories.png'}")
    
    def plot_reward_analysis(
        self, 
        df: pd.DataFrame
    ) -> None:
        """
        Analyze reward patterns and survival outcomes.
        
        Args:
            df: Dataset DataFrame
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Reward distribution by action
        for action in range(4):
            action_rewards = df[df['action'] == action]['reward']
            axes[0].hist(action_rewards, alpha=0.6, label=f'Action {action}', 
                        bins=30, density=True)
        
        axes[0].set_xlabel('Reward')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Reward Distribution by Action')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Survival analysis
        terminal_df = df[df['terminal']]
        survival_rate = (terminal_df['reward'] > 0).mean()
        
        axes[1].pie([survival_rate, 1-survival_rate], 
                   labels=['Survived', 'Died'], 
                   autopct='%1.1f%%',
                   colors=['#2ca02c', '#d62728'])
        axes[1].set_title(f'Overall Survival Rate: {survival_rate:.1%}')
        
        # Cumulative reward by episode
        episode_rewards = df.groupby('episode_id')['reward'].sum()
        axes[2].hist(episode_rewards, bins=20, alpha=0.7, color='skyblue')
        axes[2].set_xlabel('Cumulative Episode Reward')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Episode Reward Distribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'reward_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Reward analysis saved to {self.figures_dir / 'reward_analysis.png'}")
    
    def create_interactive_dashboard(
        self, 
        df: pd.DataFrame, 
        models: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create interactive Plotly dashboard (if available).
        
        Args:
            df: Dataset DataFrame
            models: Dictionary of trained models
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available for interactive dashboard")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Action Distribution', 'MAP vs Lactate', 
                          'Episode Trajectories', 'Survival Analysis'],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )
        
        # Action distribution
        action_counts = df['action'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=action_counts.index, y=action_counts.values, 
                  name='Actions', marker_color='lightblue'),
            row=1, col=1
        )
        
        # MAP vs Lactate scatter
        fig.add_trace(
            go.Scatter(
                x=df['state_MAP'], y=df['state_Lactate'], 
                mode='markers', 
                marker=dict(color=df['action'], colorscale='viridis'),
                text=df['action'], name='Patient States'
            ),
            row=1, col=2
        )
        
        # Sample episode trajectory
        sample_episode = df[df['episode_id'] == df['episode_id'].iloc[0]]
        fig.add_trace(
            go.Scatter(
                x=sample_episode['t'], y=sample_episode['state_MAP'],
                mode='lines+markers', name='MAP Trajectory'
            ),
            row=2, col=1
        )
        
        # Survival pie chart
        terminal_df = df[df['terminal']]
        survival_counts = (terminal_df['reward'] > 0).value_counts()
        fig.add_trace(
            go.Pie(
                labels=['Died', 'Survived'], 
                values=[survival_counts.get(False, 0), survival_counts.get(True, 0)],
                marker_colors=['red', 'green']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="ICU Fluid Management Dashboard",
            height=800,
            showlegend=False
        )
        
        dashboard_path = self.figures_dir / 'interactive_dashboard.html'
        fig.write_html(str(dashboard_path))
        print(f"Interactive dashboard saved to {dashboard_path}")


# Convenience functions for backward compatibility
def plot_policy_heatmap(
    models: Dict[str, Any], 
    feature_ranges: Dict[str, Dict], 
    figures_dir: str = "figures",
    **kwargs
) -> None:
    """
    Generate policy heatmaps (convenience function).
    
    Args:
        models: Dictionary of trained models
        feature_ranges: Feature value ranges from dataset
        figures_dir: Directory to save figures
        **kwargs: Additional arguments for heatmap generation
    """
    visualizer = PolicyVisualizer(figures_dir)
    visualizer.plot_policy_heatmap(models, feature_ranges, **kwargs)


def generate_all_visualizations(
    df: pd.DataFrame,
    models: Optional[Dict[str, Any]] = None,
    X_test: Optional[np.ndarray] = None,
    feature_ranges: Optional[Dict[str, Dict]] = None,
    figures_dir: str = "figures"
) -> None:
    """
    Generate comprehensive visualization suite.
    
    Args:
        df: Dataset DataFrame
        models: Dictionary of trained models
        X_test: Test features for model predictions
        feature_ranges: Feature value ranges from dataset
        figures_dir: Directory to save figures
    """
    print("Generating comprehensive visualizations...")
    
    visualizer = PolicyVisualizer(figures_dir)
    
    # Generate all standard visualizations
    visualizer.plot_action_distributions(df, models, X_test)
    visualizer.plot_clinical_correlations(df)
    visualizer.plot_episode_trajectories(df)
    visualizer.plot_reward_analysis(df)
    
    # Generate policy heatmaps if models and feature ranges available
    if models and feature_ranges:
        visualizer.plot_policy_heatmap(models, feature_ranges)
    
    # Create interactive dashboard if plotly available
    visualizer.create_interactive_dashboard(df, models)
    
    print(f"All visualizations saved to {figures_dir}/")