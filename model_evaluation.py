import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

print("Loading data...")
# Read the Excel file from the specific folder
file_path = r'd:\GIS_works\Aaa\B\Akka_data\ahp_model_dept.xlsx'
df = pd.read_excel(file_path)

print("Checking and filling missing values...")
# Handle missing values: median for numeric, mode for categorical
for col in df.columns:
    if df[col].isnull().any():
        print(f"Filling missing values in column '{col}'")
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

print("Preparing target variable and fixing data alignment...")
if 'points' in df.columns:
    # ground truth (1) vs random points (0)
    df['Target'] = df['points'].astype(str).str.lower().apply(lambda x: 1 if 'gw' in x else 0)
    
    # Fix corrupted data in backend: specific columns for 'gw' points got shifted in the dataset
    gw_mask = df['points'].str.contains('gw', case=False, na=False)
    gw_data = df[gw_mask].copy()
    
    if gw_mask.any():
        df.loc[gw_mask, 'rainfall'] = gw_data['curvature']
        df.loc[gw_mask, 'curvature'] = gw_data['rainfall']
        df.loc[gw_mask, 'soil'] = gw_data['slope']
        df.loc[gw_mask, 'tpi'] = gw_data['soil']
        df.loc[gw_mask, 'twi'] = gw_data['tpi']
        df.loc[gw_mask, 'slope'] = gw_data['twi']
        
    X = df.drop(columns=['points', 'Target'])
else:
    print("Column 'points' not found. Please verify the dataset structure.")
    exit(1)

# Encode categorical variables if any exist
for col in X.columns:
    if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
        print(f"Encoding categorical column: {col}")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

y = df['Target']

print(f"Data shape after preprocessing - Features: {X.shape}, Target: {y.shape}")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

roc_data = {}
trained_models = {}

print("Training models and calculating ROC-AUC...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    trained_models[name] = model
    print(f"{name} AUC: {roc_auc:.4f}")

# Plot 1: ROC-AUC Curves
print("Generating ROC Curves plot...")
plt.figure(figsize=(10, 8))
for name, metrics in roc_data.items():
    plt.plot(metrics['fpr'], metrics['tpr'], lw=2, label=f"{name} (AUC = {metrics['auc']:.3f})")

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
roc_plot_path = r'd:\GIS_works\Aaa\B\Akka_data\roc_curves.png'
plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
print(f"Saved {roc_plot_path}")

# Plot 2: Variable Importance
print("Generating Variable Importance plots...")
# Set up a figure with 3 subplots side by side
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for i, (name, model) in enumerate(trained_models.items()):
    if name in ['Random Forest', 'Decision Tree']:
        importances = model.feature_importances_
    else:
        # For SVM (RBF kernel), we use permutation importance on the test set
        print(f"Calculating permutation importance for {name}...")
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importances = result.importances_mean
        
    # Normalize importance to percentages
    importances = 100.0 * (importances / importances.sum())
    
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=axes[i], palette='viridis', hue='Feature', legend=False)
    axes[i].set_title(f'{name} Variable Importance')
    axes[i].set_xlabel('Relative Importance (%)')
    if i > 0:
        axes[i].set_ylabel('')

plt.tight_layout()
var_imp_path = r'd:\GIS_works\Aaa\B\Akka_data\variable_importance.png'
plt.savefig(var_imp_path, dpi=300, bbox_inches='tight')
print(f"Saved {var_imp_path}")

print("Evaluation complete.")
