

# ── 1. IMPORTS ──────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

# ── 2. LOAD DATA ─────────────────────────────────────────
print("=" * 55)
print("  STEP 1 — Loading Data")
print("=" * 55)

df = pd.read_csv('Gaming_Academic_Performance_DS_Project.csv')
print(f"  Shape     : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Columns   : {list(df.columns)}")
print(f"\n  Target (grades) stats:")
print(f"    Mean    : {df['grades'].mean():.2f}")
print(f"    Std     : {df['grades'].std():.2f}")
print(f"    Min/Max : {df['grades'].min():.2f} / {df['grades'].max():.2f}")
print(f"\n  Missing values: {df.isnull().sum().sum()}")

# ── 3. EDA — QUICK PLOTS ─────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 2 — Exploratory Data Analysis (EDA)")
print("=" * 55)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Gaming vs Academic Performance — EDA', fontsize=15, fontweight='bold')

# Grade distribution
axes[0, 0].hist(df['grades'].clip(0, 100), bins=20, color='steelblue', edgecolor='white')
axes[0, 0].set_title('Grade Distribution')
axes[0, 0].set_xlabel('Grade')
axes[0, 0].set_ylabel('Count')

# Gaming hours vs Grades (scatter)
colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
for level, grp in df.groupby('stress_level'):
    axes[0, 1].scatter(grp['gaming_hours'], grp['grades'], alpha=0.15,
                       s=8, label=level, color=colors[level])
axes[0, 1].set_title('Gaming Hours vs Grades (by Stress)')
axes[0, 1].set_xlabel('Gaming Hours/day')
axes[0, 1].set_ylabel('Grade')
axes[0, 1].legend(fontsize=9)

# Study hours vs Grades
axes[0, 2].scatter(df['study_hours'], df['grades'], alpha=0.15, s=8, color='teal')
axes[0, 2].set_title('Study Hours vs Grades')
axes[0, 2].set_xlabel('Study Hours/day')
axes[0, 2].set_ylabel('Grade')

# Avg grade by gaming bucket
df['gaming_bucket'] = pd.cut(df['gaming_hours'], bins=[0,2,4,6,8,15],
                              labels=['0-2h','2-4h','4-6h','6-8h','8+h'])
bucket_means = df.groupby('gaming_bucket', observed=True)['grades'].mean()
bars = axes[1, 0].bar(bucket_means.index, bucket_means.values,
                       color=['#2d8c56','#5aad7e','#f0a64b','#e05c2e','#c0392b'])
axes[1, 0].set_title('Avg Grade by Gaming Hours')
axes[1, 0].set_xlabel('Daily Gaming')
axes[1, 0].set_ylabel('Avg Grade')
for bar, val in zip(bars, bucket_means.values):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', fontsize=9)

# Correlation heatmap
num_cols = ['gaming_hours','study_hours','sleep_hours','attendance',
            'reaction_time_ms','addiction_score','social_activity',
            'device_usage','grades']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=axes[1, 1], cmap='RdYlGn',
            center=0, annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5)
axes[1, 1].set_title('Correlation Heatmap')
axes[1, 1].tick_params(axis='x', rotation=45, labelsize=8)
axes[1, 1].tick_params(axis='y', rotation=0, labelsize=8)

# Avg grade by stress level
stress_means = df.groupby('stress_level')['grades'].mean().reindex(['Low','Medium','High'])
axes[1, 2].bar(stress_means.index, stress_means.values,
               color=['#e05c2e','#f0a64b','#2d8c56'])
axes[1, 2].set_title('Avg Grade by Stress Level')
axes[1, 2].set_xlabel('Stress Level')
axes[1, 2].set_ylabel('Avg Grade')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("  EDA plots saved → eda_plots.png")

# ── 4. PREPROCESSING ─────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 3 — Preprocessing & Feature Engineering")
print("=" * 55)

df_model = df.drop(columns=['student_id', 'gaming_bucket'])

# Encode categoricals
le_gender = LabelEncoder()
le_genre  = LabelEncoder()
le_stress = LabelEncoder()

df_model['gender']       = le_gender.fit_transform(df_model['gender'])
df_model['gaming_genre'] = le_genre.fit_transform(df_model['gaming_genre'])
df_model['stress_level'] = le_stress.fit_transform(df_model['stress_level'])

print(f"  gender classes   : {list(le_gender.classes_)}")
print(f"  gaming_genre     : {list(le_genre.classes_)}")
print(f"  stress_level     : {list(le_stress.classes_)}")

# Feature matrix & target
FEATURES = ['age','gender','gaming_hours','study_hours','sleep_hours',
            'attendance','gaming_genre','social_activity','device_usage',
            'reaction_time_ms','addiction_score','stress_level']

X = df_model[FEATURES]
y = df_model['grades'].clip(0, 100)   # cap at 0-100

# Train / test split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\n  Train size : {len(X_train)}")
print(f"  Test size  : {len(X_test)}")

# ── 5. TRAIN MODELS ──────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 4 — Training Models")
print("=" * 55)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    'Linear Regression' : LinearRegression(),
    'Ridge Regression'  : Ridge(alpha=1.0),
    'Random Forest'     : RandomForestRegressor(n_estimators=200, max_depth=12,
                                                 random_state=42, n_jobs=-1),
    'Gradient Boosting' : GradientBoostingRegressor(n_estimators=200,
                                                     learning_rate=0.05,
                                                     max_depth=5, random_state=42),
}

results = {}
print(f"\n  {'Model':<22}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}  {'CV R²':>9}")
print("  " + "-" * 58)

for name, model in models.items():
    # Scaled data for linear models, raw for tree models
    use_sc = 'Regression' in name
    Xtr = X_train_sc if use_sc else X_train
    Xte = X_test_sc  if use_sc else X_test

    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    cv = cross_val_score(model, Xtr, y_train, cv=5, scoring='r2', n_jobs=-1)

    results[name] = {'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2,
                     'cv_r2': cv.mean(), 'scaled': use_sc, 'preds': preds}
    print(f"  {name:<22}  {rmse:>7.3f}  {mae:>7.3f}  {r2:>7.4f}  {cv.mean():>7.4f} ± {cv.std():.4f}")

# ── 6. BEST MODEL ────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]['r2'])
best      = results[best_name]
print(f"\n  Best model → {best_name}  (R² = {best['r2']:.4f})")

# ── 7. FEATURE IMPORTANCE ────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 5 — Feature Importance (Random Forest)")
print("=" * 55)

rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=FEATURES) \
                .sort_values(ascending=True)

print("\n  Feature importances:")
for feat, imp in importances.sort_values(ascending=False).items():
    bar = '█' * int(imp * 50)
    print(f"    {feat:<22} {imp:.4f}  {bar}")

# ── 8. PREDICTION PLOTS ──────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('Model Evaluation', fontsize=13, fontweight='bold')

# Actual vs Predicted — best model
axes2[0].scatter(y_test, best['preds'], alpha=0.2, s=8, color='steelblue')
mn, mx = y_test.min(), y_test.max()
axes2[0].plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect')
axes2[0].set_title(f'{best_name}\nActual vs Predicted')
axes2[0].set_xlabel('Actual Grade')
axes2[0].set_ylabel('Predicted Grade')
axes2[0].legend(fontsize=9)

# Residuals
residuals = y_test.values - best['preds']
axes2[1].scatter(best['preds'], residuals, alpha=0.2, s=8, color='coral')
axes2[1].axhline(0, color='black', lw=1)
axes2[1].set_title('Residuals')
axes2[1].set_xlabel('Predicted Grade')
axes2[1].set_ylabel('Residual')

# Feature importance bar
importances_sorted = importances.sort_values(ascending=True)
axes2[2].barh(importances_sorted.index, importances_sorted.values, color='teal')
axes2[2].set_title('Feature Importance\n(Random Forest)')
axes2[2].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Model plots saved → model_results.png")

# ── 9. SAVE THE BEST MODEL ───────────────────────────────
print("\n" + "=" * 55)
print("  STEP 6 — Save Model")
print("=" * 55)

# Save full pipeline: scaler is only needed for linear models;
# Random Forest works on raw features, so we just save the RF model.
joblib.dump(rf_model, 'grade_predictor_rf.pkl')
joblib.dump({'le_gender': le_gender, 'le_genre': le_genre, 'le_stress': le_stress},
            'label_encoders.pkl')
print("  Saved: grade_predictor_rf.pkl")
print("  Saved: label_encoders.pkl")


# ══════════════════════════════════════════════════════════
#  HOW TO MAKE A PREDICTION  ←← EDIT THIS SECTION
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  STEP 7 — Making a Prediction")
print("=" * 55)

# ─── Edit these values for any new student ───────────────
my_student = {
    'age'              : 20,        # 16–24
    'gender'           : 'Male',    # 'Male', 'Female', 'Other'
    'gaming_hours'     : 4.0,       # hours/day
    'study_hours'      : 6.0,       # hours/day
    'sleep_hours'      : 7.5,       # hours/night
    'attendance'       : 85.0,      # percentage 0–100
    'gaming_genre'     : 'FPS',     # 'FPS', 'RPG', 'Casual'
    'social_activity'  : 3.0,       # 0–10
    'device_usage'     : 8.0,       # hours/day on devices
    'reaction_time_ms' : 260.0,     # milliseconds (lower = faster)
    'addiction_score'  : 9.0,       # 0–23
    'stress_level'     : 'Medium',  # 'Low', 'Medium', 'High'
}
# ─────────────────────────────────────────────────────────

def predict_grade(student_dict):
    """
    Predict a student's grade from a plain Python dict.
    Automatically handles encoding of categorical fields.

    Parameters
    ----------
    student_dict : dict
        Keys matching `my_student` above.

    Returns
    -------
    float  — predicted grade (0-100)
    """
    encoders = joblib.load('label_encoders.pkl')
    model    = joblib.load('grade_predictor_rf.pkl')

    row = pd.DataFrame([student_dict])

    # Encode categoricals
    row['gender']       = encoders['le_gender'].transform(row['gender'])
    row['gaming_genre'] = encoders['le_genre'].transform(row['gaming_genre'])
    row['stress_level'] = encoders['le_stress'].transform(row['stress_level'])

    row = row[FEATURES]          # enforce column order
    grade = model.predict(row)[0]
    return float(np.clip(grade, 0, 100))


predicted = predict_grade(my_student)

# Letter grade mapping
def letter_grade(g):
    if g >= 90: return 'A'
    if g >= 80: return 'B'
    if g >= 70: return 'C'
    if g >= 60: return 'D'
    return 'F'

print(f"\n  Student profile:")
for k, v in my_student.items():
    print(f"    {k:<22} : {v}")

print(f"\n  ┌────────────────────────────────┐")
print(f"  │  Predicted Grade  :  {predicted:5.1f}/100  │")
print(f"  │  Letter Grade     :  {letter_grade(predicted):<10}  │")
print(f"  └────────────────────────────────┘")

