
# POLICY CARD: Offline RL Policy (Demo)

Task: Learn fluids policy from retrospective ICU EHR (demo).

State Space: ['state_MAP', 'state_HR', 'state_Lactate', 'state_Creatinine', 'state_SpO2', 'state_Temp', 'state_HourIdx']

Action Space: 4-bin fluids (0, 0-250, 250-500, >500 mL)

Reward: -0.1 if MAP<65 per hour; -0.05 if cumulative fluids >3L; terminal +1 survival / -1 death; clipped [-1,1].

Data: MIMIC-IV Demo (downloaded if available) or synthetic fallback. Split 80/20 by episode.

Models: Behavior classifier (LogReg), DiscreteBC, CQL, IQL (CPU-only, small MLP).

Safety Notes: Research-only. Defer when state is out-of-support (use support diagnostics). Optional constraint to avoid large boluses when MAP>=65.

Usage: Load d3rlpy models from ./artifacts and use predict helper in ./artifacts/predict.py.

Limitations: Demo-scale, simplified cohort and rewards, fluids-only actions, no prospective validation.

