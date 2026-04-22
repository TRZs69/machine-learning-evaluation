import pandas as pd
from irrCAC.raw import CAC

# Data rater
rater1 = [4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 3, 4, 3, 3, 4, 4, 3, 3, 4, 4, 4, 3, 3, 4, 4, 4]
rater2 = [5, 5, 4, 4, 5, 5, 5, 5, 1, 1, 5, 5, 5, 4, 5, 5, 3, 5, 5, 4, 4, 4, 3, 5, 5, 5, 3, 4, 3, 5]
rater3 = [4, 5, 5, 5, 4, 5, 3, 5, 5, 3, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 4, 5, 5, 5, 4, 3, 3, 5, 5]

# DataFrame
df = pd.DataFrame({
    'Rater1': rater1,
    'Rater2': rater2,
    'Rater3': rater3
})

# =========================
# Gwet's AC2 (3 rater)
# =========================
cac = CAC(df, weights='quadratic')
gwet_result = cac.gwet()

ac2_score = gwet_result['est']['coefficient_value']
coef_name = gwet_result['est']['coefficient_name']
p_value = gwet_result['est']['p_value']

print("--- Gwet Analysis ---")
print(f"Coefficient: {coef_name}")
print(f"Score: {ac2_score:.4f}")
print(f"P-Value: {p_value:.4f}")

# =========================
# Confusion Matrix Pairwise
# =========================
labels = [1, 2, 3, 4, 5]  # supaya semua kategori selalu muncul

cm_12 = pd.crosstab(
    pd.Categorical(df['Rater1'], categories=labels),
    pd.Categorical(df['Rater2'], categories=labels),
    rownames=['Rater1'],
    colnames=['Rater2'],
    dropna=False
)

cm_13 = pd.crosstab(
    pd.Categorical(df['Rater1'], categories=labels),
    pd.Categorical(df['Rater3'], categories=labels),
    rownames=['Rater1'],
    colnames=['Rater3'],
    dropna=False
)

cm_23 = pd.crosstab(
    pd.Categorical(df['Rater2'], categories=labels),
    pd.Categorical(df['Rater3'], categories=labels),
    rownames=['Rater2'],
    colnames=['Rater3'],
    dropna=False
)

print("\n--- Confusion Matrix: Rater1 vs Rater2 ---")
print(cm_12)

print("\n--- Confusion Matrix: Rater1 vs Rater3 ---")
print(cm_13)

print("\n--- Confusion Matrix: Rater2 vs Rater3 ---")
print(cm_23)