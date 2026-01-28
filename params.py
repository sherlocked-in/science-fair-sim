
### **File 2: `params.py`**
```python
"""NP parameters from Hersh 2022 review[file:11]"""
NP_DESIGNS = {
    "AMT_Gold": {"size": 15, "zeta": 25, "tf_ligand": 0, "color": "red"},
    "RMT_PBCA": {"size": 80, "zeta": 0, "tf_ligand": 1, "color": "blue"}, 
    "DUAL_GoldTf": {"size": 15, "zeta": 15, "tf_ligand": 1, "color": "gold"}
}

RATES = {
    "amt_base": 1e-4,      # Cationic charge[file:11]
    "rmt_base": 5e-5,      # Tf receptor[file:11]  
    "synergy_factor": 0.4, # Literature synergy
    "diffusion": 1e-6      # Brownian
}
