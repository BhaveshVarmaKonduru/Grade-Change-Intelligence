import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path

def discover_correlations():
    csv_path = Path(__file__).resolve().parent.parent / "data" / "historical_data.csv"
    if not csv_path.exists():
        return []
    
    df = pd.read_csv(csv_path)
    
    features = ['speed', 'stock_flow', 'steam', 'filler', 'basis_weight', 'moisture', 'ash', 'caliper']
    if 'off_spec' not in df.columns:
        return []
        
    df_clean = df[features + ['off_spec']].dropna()
    if len(df_clean) == 0:
        return []
        
    X = df_clean[features]
    y = df_clean['off_spec']
    
    # Compute Pearson Correlation
    correlations = X.corrwith(y).to_dict()
    
    # Compute Mutual Information
    try:
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_dict = dict(zip(features, mi_scores))
    except Exception:
        mi_dict = {f: 0.0 for f in features}
        
    findings = []
    for feat in features:
        findings.append({
            "feature": feat,
            "correlation": correlations.get(feat, 0.0),
            "mutual_information": mi_dict.get(feat, 0.0)
        })
        
    # Sort findings by absolute correlation / mutual information
    findings = sorted(findings, key=lambda x: abs(x["correlation"]), reverse=True)
    return findings
