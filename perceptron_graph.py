import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------------------
# PERCEPTRON POCKET + ACP + GRID SEARCH + GRAPHIQUE
# ------------------------------------------------------------------------------
"""
Ajout de graphiques :
- Nombre d'erreurs d'entraînement par itération
- Précision test par configuration pendant le grid-search
"""

# 1. Chargement
df = pd.read_csv(Path("Data") / "bcw_data.csv")
X = df.drop(['id','diagnosis','Unnamed: 32'],axis=1).values
y = np.where(df['diagnosis']=="M",1,0)

# 2. Train/test
np.random.seed(0)
idx = np.random.permutation(len(X))
split = int(0.8*len(X))
X_tr, y_tr = X[idx[:split]], y[idx[:split]]
X_te, y_te = X[idx[split:]], y[idx[split:]]

# 3. Standardisation
mu = X_tr.mean(0); sigma = X_tr.std(0)+1e-8
X_trs = (X_tr-mu)/sigma; X_tes = (X_te-mu)/sigma

# 4. ACP
cov = np.cov(X_trs.T)
eigs, vecs = np.linalg.eig(cov)
order = np.argsort(eigs)[::-1]
eigs, vecs = eigs[order], vecs[:,order]
cumvar = np.cumsum(eigs)/eigs.sum()

# 5. Grid-search + pocket + collecte des métriques
best_prec = 0.0
best_cfg = None
best_w = best_b = best_P = None

# Pour tracer performance test
results_grid = []

for n_comp in [8,10,12,15]:
    P = vecs[:,:n_comp]
    Xp_tr = X_trs.dot(P); Xp_te = X_tes.dot(P)

    for lr in [0.01,0.05,0.1]:
        # Initialisation
        w = np.zeros(n_comp); b = 0.0
        wp, bp = w.copy(), b
        best_pock = 0.0
        errors_history = []

        # Pocket perceptron
        for epoch in range(200):
            errors = 0
            for xi, yi in zip(Xp_tr, y_tr):
                out = xi.dot(w) + b
                y_pred = 1 if out >= 0 else 0
                e = yi - y_pred
                if e != 0:
                    w += lr*e*xi; b += lr*e
                    errors += 1
            errors_history.append(errors)

            # pocket update
            preds_tr = np.where(Xp_tr.dot(w)+b>=0,1,0)
            tp = np.sum((preds_tr==1)&(y_tr==1))
            fp = np.sum((preds_tr==1)&(y_tr==0))
            prec_tr = tp/(tp+fp) if tp+fp>0 else 0
            if prec_tr > best_pock:
                best_pock, wp, bp = prec_tr, w.copy(), b

        # Évaluation pocket sur test
        preds_te = np.where(Xp_te.dot(wp)+bp>=0,1,0)
        tp_te = np.sum((preds_te==1)&(y_te==1))
        fp_te = np.sum((preds_te==1)&(y_te==0))
        prec_te = tp_te/(tp_te+fp_te) if tp_te+fp_te>0 else 0

        results_grid.append((n_comp, lr, prec_te, errors_history))

        # Mise à jour du meilleur
        if prec_te > best_prec:
            best_prec, best_cfg = prec_te, (n_comp, lr)
            best_w, best_b, best_P = wp, bp, P
            best_errors_history = errors_history.copy()

# 6. Tracer apprentissage sur la meilleure configuration
plt.figure(figsize=(10,4))
plt.plot(best_errors_history, marker='o', label='Erreurs par époque')
plt.xlabel('Époque')
plt.ylabel('Nombre d\'erreurs')
plt.title(f'Convergence (ACP={best_cfg[0]}, lr={best_cfg[1]})')
plt.grid(True)
plt.legend()
plt.show()

# 7. Tracer précision test pour chaque configuration
cfg_labels = [f"{nc}c,{lr}lr" for nc,lr,_,_ in results_grid]
precisions = [res for _,_,res,_ in results_grid]
plt.figure(figsize=(10,4))
plt.bar(cfg_labels, precisions)
plt.xticks(rotation=45)
plt.ylabel('Précision test')
plt.title('Comparaison des configurations ACP vs lr')
plt.ylim(0,1)
plt.grid(axis='y')
plt.show()

# 8. Évaluation finale
Xp_te = X_tes.dot(best_P)
y_pred = np.where(Xp_te.dot(best_w)+best_b>=0,1,0)
acc = np.mean(y_pred==y_te)
tp = np.sum((y_pred==1)&(y_te==1))
tn = np.sum((y_pred==0)&(y_te==0))
fp = np.sum((y_pred==1)&(y_te==0))
fn = np.sum((y_pred==0)&(y_te==1))
rec = tp/(tp+fn) if tp+fn>0 else 0
f1 = 2*best_prec*rec/(best_prec+rec) if best_prec+rec>0 else 0

print(f"\n💎 Best precision test: {best_prec*100:.2f}% with ACP={best_cfg[0]} & lr={best_cfg[1]}")
print(f"Accuracy: {acc*100:.2f}%, Recall: {rec:.4f}, F1: {f1:.4f}")
print(f"Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
