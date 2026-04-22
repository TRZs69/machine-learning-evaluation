import pandas as pd
import numpy as np
from irrCAC.raw import CAC

try:
    df_results = pd.read_csv('evaluation_results_clean.csv', sep=';')
except:
    df_results = pd.read_csv('evaluation_results_clean.csv')

df_results['human_score'] = pd.to_numeric(df_results['human_score'], errors='coerce')
df_results['llm_score'] = pd.to_numeric(df_results['llm_score'], errors='coerce')

df_clean = df_results.dropna(subset=['human_score', 'llm_score'])

print(f"Analyzing {len(df_clean)} valid samples...")

df_rater = df_clean[['human_score', 'llm_score']]

cac = CAC(df_rater, weights='quadratic')
gwet_result = cac.gwet()

ac2_score = gwet_result['est']['coefficient_value']
coef_name = gwet_result['est']['coefficient_name']
p_value = gwet_result['est']['p_value']

print("\n" + "="*30)
print("--- Gwet Analysis (Human vs LLM) ---")
print(f"Coefficient: {coef_name}")
print(f"Score: {ac2_score:.4f}")
print(f"P-Value: {p_value:.4f}")
print("="*30)

labels = [1, 2, 3, 4, 5]
cm = pd.crosstab(
    pd.Categorical(df_clean['human_score'], categories=labels),
    pd.Categorical(df_clean['llm_score'], categories=labels),
    rownames=['Human'],
    colnames=['LLM'],
    dropna=False
)

print("\n--- Confusion Matrix ---")
print(cm)