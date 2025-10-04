# ==============================================================================
# PERCEPTRON + ACP AUTOMATIQUE (≥95% VARIANCE) - ADAPTÉ À bcw_data.csv
# ==============================================================================
"""
Pipeline :
1. Chargement et préparation des données
2. Séparation train/test manuelle
3. Standardisation (avant ACP)
4. ACP avec choix automatique du nombre de composantes pour atteindre ≥95% de variance
5. Perceptron from scratch sur les données ACP
6. Évaluation des performances
"""

import numpy as np
import pandas as pd
from pathlib import Path

print("🎯 PERCEPTRON + ACP AUTOMATIQUE - DATASET bcw_data.csv")
print("=" * 60)

# ------------------------------------------------------------------------------
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ------------------------------------------------------------------------------
file_path = Path("Data") / "bcw_data.csv"
df = pd.read_csv(file_path)

print("📊 Chargement des données...")
print(f"   Shape: {df.shape}")
print(f"   Classes: {df['diagnosis'].value_counts().to_dict()}")

# Features et cible
X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).values
y = np.where(df['diagnosis'].values == 'M', 1, 0)  # M→1, B→0
print(f"   Features shape: {X.shape}")
print(f"   Distribution [B=0, M=1]: {np.bincount(y)}")

# ------------------------------------------------------------------------------
# 2. SÉPARATION TRAIN/TEST (À LA MAIN)
# ------------------------------------------------------------------------------
print("\n🔄 Séparation train/test...")
np.random.seed(42)
indices = np.random.permutation(len(X))
split_idx = int(0.8 * len(X))  # TU PEUX MODIFIER
train_idx, test_idx = indices[:split_idx], indices[split_idx:]

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]
print(f"   Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# ------------------------------------------------------------------------------
# 3. STANDARDISATION (OBLIGATOIRE POUR ACP + Perceptron)
# ------------------------------------------------------------------------------
print("\n🔧 Standardisation...")
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-8
X_train_scaled = (X_train - mean) / std
X_test_scaled  = (X_test  - mean) / std
print("   ✅ Standardisation terminée")

# ------------------------------------------------------------------------------
# 4. ACP – CHOIX AUTOMATIQUE DES COMPOSANTES (≥95% VARIANCE)
# ------------------------------------------------------------------------------
print("\n🔍 Calcul ACP...")
cov_mat = np.cov(X_train_scaled.T)
eigs, vecs = np.linalg.eig(cov_mat)
idx = np.argsort(eigs)[::-1]
eigs, vecs = eigs[idx], vecs[:, idx]
cum_var = np.cumsum(eigs) / np.sum(eigs)
n_comp  = np.searchsorted(cum_var, 0.95) + 1  # au moins 95% variance

print(f"   Composantes retenues: {n_comp} / {X_train.shape[1]}")
print(f"   Variance expliquée cumulée: {cum_var[n_comp-1]:.4f}")

P = vecs[:, :n_comp]
X_train_pca = X_train_scaled.dot(P)
X_test_pca  = X_test_scaled.dot(P)
print(f"   Nouvelle dimension: {X_train_pca.shape}")

# ------------------------------------------------------------------------------
# 5. INITIALISATION DU PERCEPTRON FROM SCRATCH
# ------------------------------------------------------------------------------
print("\n🧠 Initialisation du Perceptron...")
np.random.seed(42)
w = np.random.normal(0, 0.01, n_comp)
b = 0.0
learning_rate = 0.01  # TU PEUX MODIFIER
n_epochs      = 1000  # TU PEUX MODIFIER
print(f"   Features PCA: {n_comp}, LR: {learning_rate}, Epochs: {n_epochs}")

# ------------------------------------------------------------------------------
# 6. ENTRAÎNEMENT DU PERCEPTRON
# ------------------------------------------------------------------------------
print("\n🚀 Début de l'entraînement...")
errors_history = []
for epoch in range(n_epochs):
    errors = 0
    for xi, yi in zip(X_train_pca, y_train):
        out = xi.dot(w) + b
        y_pred = 1 if out >= 0 else 0
        e = yi - y_pred
        if e != 0:
            w += learning_rate * e * xi
            b += learning_rate * e
            errors += 1
    errors_history.append(errors)
    if epoch < 10 or (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d}: erreurs = {errors}")
    if errors == 0:
        print(f"✅ Convergence atteinte à l'époque {epoch+1}!")
        break
print(f"🏁 Entraînement terminé après {len(errors_history)} époques")

# ------------------------------------------------------------------------------
# 7. PRÉDICTION ET ÉVALUATION (À LA MAIN)
# ------------------------------------------------------------------------------
print("\n🎯 Prédictions sur le test set...")
y_pred = np.array([1 if xi.dot(w)+b >= 0 else 0 for xi in X_test_pca])

acc = np.mean(y_pred == y_test)
tp  = np.sum((y_pred==1)&(y_test==1))
tn  = np.sum((y_pred==0)&(y_test==0))
fp  = np.sum((y_pred==1)&(y_test==0))
fn  = np.sum((y_pred==0)&(y_test==1))
prec = tp/(tp+fp) if tp+fp>0 else 0
rec  = tp/(tp+fn) if tp+fn>0 else 0
f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0

print("\n⚡ RÉSULTATS FINAUX:")
print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"   Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
print(f"   Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"   Features réduites: {X_train.shape[1]}→{n_comp}")

# ------------------------------------------------------------------------------
# 8. PARAMÈTRES FINAUX
# ------------------------------------------------------------------------------
print("\n🧠 Paramètres appris:")
print(f"   Biais: {b:.6f}")
print(f"   Poids (extraits): {w[:5]} ... {w[-5:]}")
print("=" * 60)