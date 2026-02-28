import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load enhanced features
with open("data/link_data_enhanced.pkl", "rb") as f:
    X, y = pickle.load(f)

print("=" * 60)
print("ENHANCED LINK PREDICTION MODEL TRAINING")
print("=" * 60)
print(f"Total samples: {len(y)}")
print(f"Features per sample: {X.shape[1]}")
print(f"Positive samples: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
print(f"Negative samples: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
print("=" * 60)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# MODEL 1: Logistic Regression with GridSearch
# ============================================================
print("\n[1/3] Training Logistic Regression with hyperparameter tuning...")

lr_params = {
    'C': [0.1, 1, 10, 100],
    'max_iter': [500, 1000]
}

lr = LogisticRegression(random_state=42)
lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='roc_auc', n_jobs=-1)
lr_grid.fit(X_train_scaled, y_train)

lr_best = lr_grid.best_estimator_
lr_pred = lr_best.predict(X_test_scaled)
lr_prob = lr_best.predict_proba(X_test_scaled)[:, 1]

print(f"  Best params: {lr_grid.best_params_}")
print(f"  Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(f"  ROC-AUC: {roc_auc_score(y_test, lr_prob):.4f}")

# ============================================================
# MODEL 2: Random Forest with GridSearch
# ============================================================
print("\n[2/3] Training Random Forest with hyperparameter tuning...")

rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)
rf_prob = rf_best.predict_proba(X_test)[:, 1]

print(f"  Best params: {rf_grid.best_params_}")
print(f"  Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"  ROC-AUC: {roc_auc_score(y_test, rf_prob):.4f}")

# ============================================================
# MODEL 3: XGBoost with GridSearch
# ============================================================
print("\n[3/3] Training XGBoost with hyperparameter tuning...")

xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0]
}

xgb = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb, xgb_params, cv=5, scoring='roc_auc', n_jobs=-1)
xgb_grid.fit(X_train, y_train)

xgb_best = xgb_grid.best_estimator_
xgb_pred = xgb_best.predict(X_test)
xgb_prob = xgb_best.predict_proba(X_test)[:, 1]

print(f"  Best params: {xgb_grid.best_params_}")
print(f"  Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
print(f"  ROC-AUC: {roc_auc_score(y_test, xgb_prob):.4f}")

# ============================================================
# COMPARISON & BEST MODEL SELECTION
# ============================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results = [
    ("Logistic Regression", accuracy_score(y_test, lr_pred), roc_auc_score(y_test, lr_prob)),
    ("Random Forest", accuracy_score(y_test, rf_pred), roc_auc_score(y_test, rf_prob)),
    ("XGBoost", accuracy_score(y_test, xgb_pred), roc_auc_score(y_test, xgb_prob))
]

print(f"{'Model':<25} {'Accuracy':<12} {'ROC-AUC':<12}")
print("-" * 60)
for model_name, acc, auc in results:
    print(f"{model_name:<25} {acc:<12.4f} {auc:<12.4f}")

# Select best model based on ROC-AUC
best_idx = np.argmax([r[2] for r in results])
best_model_name = results[best_idx][0]
best_model = [lr_best, rf_best, xgb_best][best_idx]
best_pred = [lr_pred, rf_pred, xgb_pred][best_idx]

print("\n" + "=" * 60)
print(f"🏆 BEST MODEL: {best_model_name}")
print("=" * 60)
print("\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=['No Link', 'Link']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# Save best model and scaler
with open("data/link_model_best.pkl", "wb") as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'model_name': best_model_name
    }, f)

print(f"\n✅ Best model ({best_model_name}) saved to data/link_model_best.pkl")
print("=" * 60)
