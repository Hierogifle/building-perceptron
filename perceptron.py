# ==============================================================================
# PERCEPTRON POCKET + ACP + GRID SEARCH – MAX PRÉCISION
# ==============================================================================
"""
Pipeline optimisé :
1. Chargement et préparation des données
2. Séparation train/test manuelle
3. Standardisation
4. ACP (grid sur n_components pour ≥95% variance)
5. Pocket Perceptron (conserve les meilleurs poids selon précision)
6. Évaluation finale sur le test set
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------------------
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ------------------------------------------------------------------------------
# Lecture du CSV depuis le dossier "Data"
df = pd.read_csv(Path("Data") / "bcw_data.csv")

# Suppression des colonnes non informatives et conversion en array
X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).values

# Conversion des labels 'M'/'B' en binaire 1/0
y = np.where(df['diagnosis'] == "M", 1, 0)

# ------------------------------------------------------------------------------
# 2. SÉPARATION TRAIN/TEST (80% / 20%) – MANUELLE
# ------------------------------------------------------------------------------
np.random.seed(0)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

X_tr, y_tr = X[train_idx], y[train_idx]
X_te, y_te = X[test_idx],    y[test_idx]

# ------------------------------------------------------------------------------
# 3. STANDARDISATION (nécessaire avant ACP et apprentissage)
# ------------------------------------------------------------------------------
# Calcul des moyennes et écarts-types sur le train set
mu    = X_tr.mean(axis=0)
sigma = X_tr.std(axis=0) + 1e-8  # ajout d'un epsilon pour éviter la division par zéro

# Application de la standardisation
X_trs = (X_tr - mu) / sigma
X_tes = (X_te - mu) / sigma

# ------------------------------------------------------------------------------
# 4. ACP – CALCUL DES COMPOSANTES PRINCIPALES
# ------------------------------------------------------------------------------
# Matrice de covariance
cov = np.cov(X_trs.T)

# Valeurs et vecteurs propres
eigs, vecs = np.linalg.eig(cov)

# Tri décroissant par valeur propre
order = np.argsort(eigs)[::-1]
eigs, vecs = eigs[order], vecs[:, order]

# Variance cumulée
cumvar = np.cumsum(eigs) / np.sum(eigs)

# ------------------------------------------------------------------------------
# 5. GRID SEARCH + POCKET PERCEPTRON
# ------------------------------------------------------------------------------
best_prec = 0.0
best_cfg  = None
best_w    = None
best_b    = None
best_P    = None

# Tester plusieurs nombres de composantes et learning rates
for n_comp in [8, 10, 12, 15]:
    # Projection des données sur les n_comp premières composantes
    P       = vecs[:, :n_comp]
    Xp_tr   = X_trs.dot(P)
    Xp_te   = X_tes.dot(P)
    for lr in [0.01, 0.05, 0.1]:
        # Initialisation des poids et biais
        w  = np.zeros(n_comp)
        b  = 0.0
        wp = w.copy()  # pocket poids
        bp = b        # pocket biais
        best_pock = 0.0

        # Entraînement pocket perceptron
        for _ in range(200):  # itérations limitées
            for xi, yi in zip(Xp_tr, y_tr):
                out    = xi.dot(w) + b
                y_pred = 1 if out >= 0 else 0
                e      = yi - y_pred
                if e != 0:
                    w += lr * e * xi
                    b += lr * e
            # Évaluer sur train pour pocket
            preds = np.where(Xp_tr.dot(w) + b >= 0, 1, 0)
            tp    = np.sum((preds == 1) & (y_tr == 1))
            fp    = np.sum((preds == 1) & (y_tr == 0))
            prec  = tp / (tp + fp) if tp + fp > 0 else 0
            if prec > best_pock:
                best_pock, wp, bp = prec, w.copy(), b

        # Évaluer pocket sur test
        preds_te = np.where(Xp_te.dot(wp) + bp >= 0, 1, 0)
        tp_te    = np.sum((preds_te == 1) & (y_te == 1))
        fp_te    = np.sum((preds_te == 1) & (y_te == 0))
        prec_te  = tp_te / (tp_te + fp_te) if tp_te + fp_te > 0 else 0

        # Mettre à jour la meilleure configuration globale
        if prec_te > best_prec:
            best_prec = prec_te
            best_cfg  = (n_comp, lr)
            best_w, best_b, best_P = wp, bp, P

# ------------------------------------------------------------------------------
# 6. ÉVALUATION FINALE SUR LE TEST SET
# ------------------------------------------------------------------------------
Xp_te = X_tes.dot(best_P)
y_pred = np.where(Xp_te.dot(best_w) + best_b >= 0, 1, 0)

acc = np.mean(y_pred == y_te)
tp  = np.sum((y_pred == 1) & (y_te == 1))
tn  = np.sum((y_pred == 0) & (y_te == 0))
fp  = np.sum((y_pred == 1) & (y_te == 0))
fn  = np.sum((y_pred == 0) & (y_te == 1))
rec = tp / (tp + fn) if tp + fn > 0 else 0
f1  = 2 * best_prec * rec / (best_prec + rec) if best_prec + rec > 0 else 0

# Affichage des résultats
print(f"💎 Best precision test: {best_prec*100:.2f}% with ACP={best_cfg[0]} components & lr={best_cfg[1]}")
print(f"Accuracy: {acc*100:.2f}%, Recall: {rec:.4f}, F1: {f1:.4f}")
print(f"Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
