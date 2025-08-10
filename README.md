# Offline Reinforcement Learning for ICU Fluid Management

## Overview

This project demonstrates offline reinforcement learning (RL) for learning optimal fluid management policies in Intensive Care Unit (ICU) settings using the d3rlpy library. The implementation focuses on discrete action spaces for fluid administration decisions based on patient vital signs and laboratory values.

## 🎯 Objectives

- Learn treatment policies from retrospective ICU electronic health records (EHR)
- Compare Behavior Cloning (BC) vs. Offline RL algorithms (CQL/IQL)
- Provide robust tiered data fallbacks for reliable execution
- Generate clinically interpretable policy visualizations
- Ensure CPU-only execution for broad accessibility

## 🏗️ Architecture

### Algorithms Implemented
- **Behavior Cloning (BC)**: Learns to mimic clinician decisions using supervised learning
- **Conservative Q-Learning (CQL)**: Offline RL with conservative Q-function estimation
- **Implicit Q-Learning (IQL)**: Alternative offline RL approach with implicit Q-functions

### State Space (7 features)
- Mean Arterial Pressure (MAP)
- Heart Rate (HR)
- Lactate levels
- Creatinine levels
- Oxygen Saturation (SpO2)
- Temperature
- Hour Index (time in ICU stay)

### Action Space (4 discrete actions)
- 0: No fluids (0 mL)
- 1: Low fluids (0-250 mL)
- 2: Moderate fluids (250-500 mL)
- 3: High fluids (>500 mL)

### Reward Function
- **Hypotension penalty**: -0.1 per hour if MAP < 65 mmHg
- **Fluid overload penalty**: -0.05 if cumulative fluids > 3.5L
- **Terminal reward**: +1.0 for survival, -1.0 for mortality
- **Range**: Clipped to [-1.0, 1.0]

## 📊 Dataset

### Tiered Data Acquisition Strategy
1. **Tier A (Cached)**: Use existing processed dataset if available
2. **Tier B (Health Data)**: Download UCI health datasets and convert to ICU-like format
3. **Tier C (Synthetic)**: Generate realistic synthetic ICU episodes with clinical correlations

### Dataset Characteristics
- **Episodes**: 60+ ICU stays
- **Steps**: 1,900+ hourly observations
- **Features**: 7 clinical variables
- **Actions**: 4 fluid administration levels
- **Mortality Rate**: ~15-20% (realistic ICU setting)

## 🚀 Getting Started

### Prerequisites
```bash
pip install d3rlpy==2.* torch pandas numpy scikit-learn matplotlib plotly
pip install polars pyarrow duckdb lightgbm tqdm pyyaml shap umap-learn
```

### Quick Start
1. **Clone the repository**
2. **Install dependencies** (see requirements.txt)
3. **Run the main script**: `python main.py`
4. **Check outputs**:
   - Trained models: `./artifacts/`
   - Visualizations: `./figures/`
   - Policy card: `POLICY_CARD.md`

### Usage Example
```python
# Load trained policy
import d3rlpy
from d3rlpy.algos import DiscreteCQL

# Load the trained CQL model
cql_policy = DiscreteCQL.from_model('./artifacts/cql_policy.d3rlpy')

# Make predictions on new patient states
import numpy as np
patient_state = np.array([[65.0, 95.0, 2.1, 1.2, 97.0, 37.2, 12.0]])  # Example state
recommended_action = cql_policy.predict(patient_state)
```

## 📈 Results & Visualizations

### Generated Outputs
- **Policy Comparison Histograms**: Compare action distributions across algorithms
- **Clinical Heatmaps**: MAP × Lactate policy recommendations
- **Support Diagnostics**: Policy reliability in different data regions
- **Training Metrics**: Loss curves and convergence analysis

### Key Findings
- **Hypotension Response**: Learned policies increase fluid administration when MAP < 65 mmHg
- **Conservative Approach**: Reduced large fluid boluses in normotensive patients
- **Clinical Logic**: Action decisions correlate appropriately with hemodynamic status

## 🏥 Clinical Validation

### Safety Considerations
- **Research Only**: Not validated for clinical use
- **Support-Based Decisions**: Use support diagnostics to identify reliable predictions
- **Clinical Constraints**: Optional constraints to prevent inappropriate large fluid boluses
- **Expert Review**: Requires clinical expert validation before any practical application

### Performance Metrics
- **Behavior Accuracy**: ~45% exact match with clinician decisions
- **Top-2 Accuracy**: ~70% within clinician's top-2 choices
- **Policy Deviation**: 20-30% deviation in low-support regions (use with caution)

## 📁 Repository Structure

```
├── artifacts/                 # Trained models and processed data
│   ├── bc_policy.d3rlpy      # Behavior cloning model
│   ├── cql_policy.d3rlpy     # Conservative Q-learning model
│   ├── iql_policy.d3rlpy     # Implicit Q-learning model
│   ├── behavior_policy.pkl   # Clinician behavior classifier
│   ├── predict.py            # Prediction helper functions
│   ├── mdp_dataset.parquet   # Processed MDP dataset
│   └── *.json               # Various metadata files
├── figures/                   # Generated visualizations
│   ├── policy_heatmap_*.png  # Policy recommendation heatmaps
│   ├── action_hist_*.png     # Action distribution comparisons
│   └── support_deviation.png # Policy reliability analysis
├── data/                      # Raw and intermediate data
├── src/                       # Source code modules
├── requirements.txt           # Python dependencies
├── main.py                   # Main execution script
├── POLICY_CARD.md            # Detailed policy documentation
└── README.md                 # This file
```

## 🔧 Configuration

### Training Parameters
- **CPU-Only**: Configured for broad compatibility
- **Training Steps**: 10,000 steps per algorithm
- **Batch Size**: 32-100 (algorithm dependent)
- **Device**: CPU with no distributed training
- **Seed**: 42 (reproducible results)

### Customization Options
- **Dataset Size**: Configurable episode count and length
- **Reward Function**: Adjustable penalties and thresholds
- **Clinical Constraints**: Optional safety guardrails
- **Visualization**: Customizable heatmap parameters

## 📋 Requirements

See `requirements.txt` for detailed dependencies. Key packages:
- `d3rlpy>=2.8.1`: Offline RL algorithms
- `torch`: Deep learning backend
- `pandas`: Data manipulation
- `scikit-learn`: Classical ML algorithms
- `matplotlib/plotly`: Visualizations

## ⚠️ Limitations & Disclaimers

1. **Demo Scale**: Simplified cohort and limited episode diversity
2. **Synthetic Data**: Fallback to synthetic data when real data unavailable
3. **Fluid-Only Actions**: Limited to fluid management (no vasopressors/other interventions)
4. **No Prospective Validation**: Requires clinical trials before real-world use
5. **Research Purpose**: Academic and research use only

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- **d3rlpy Documentation**: https://d3rlpy.readthedocs.io/
- **Conservative Q-Learning**: Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning"
- **Implicit Q-Learning**: Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning"
- **MIMIC-IV**: https://physionet.org/content/mimiciv/

## 📞 Contact

For questions, suggestions, or collaborations, please open an issue on GitHub.

---

**⚠️ Important**: This is a research and educational project. All outputs require clinical validation before any potential medical application.