import pandas as pd

def add_trial(nct_id, size_nm, zeta_mv, ligand, tumor, phase3_success):
    """ADD YOUR CLINICALTRIALS.GOV TRIALS HERE"""
    data = {
        'NCT_ID': [nct_id],
        'Size_nm': [size_nm],
        'Zeta_mV': [zeta_mv],
        'Ligand': [ligand],
        'Tumor': [tumor],
        'Phase3_Success': [phase3_success]  # 1=success, 0=fail
    }
    df = pd.DataFrame(data)
    df.to_csv('trials.csv', mode='a', header=False, index=False)
    print(f"Added {nct_id}")

# === YOUR TRIALS FROM CLINICALTRIALS.GOV ===
add_trial('NCT04553133', 15, 15, 'TfR', 'GBM', 0)
add_trial('NCT03742713', 80, 0, 'None', 'Breast', 1)
add_trial('NCT02962295', 25, -10, 'PEG', 'Lung', 0)
add_trial('NCT03609544', 110, 5, 'None', 'Pancreatic', 0)
add_trial('NCT02484391', 20, 20, 'Folate', 'Ovarian', 1)
