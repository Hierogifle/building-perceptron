# import numpy as np
# import matplotlib.pyplot as plt


# class Perceptron:
#     """
#     Implémentation de la class du Perceptron :
#     - Early stopping
#     - Shuffle des données
#     - Test de plusieurs fonctions d'activation
#     """

#     def __init__(self, learning_rate=0.01, n_iter=1000, patience=10, shuffle=True, verbose=False):
#         self.lr = learning_rate
#         self.n_iter = n_iter
#         self.patience = patience
#         self.shuffle = shuffle
#         self.verbose = verbose

#         self.weights_ = None
#         self.bias_ = 0
#         self.best_error_ = np.inf
#         self.best_weights_ = None
#         self.best_bias_ = None

#         self.errors_dict_ = {}  # erreurs par activation
#         self.activations_ = {
#             'sign': lambda z: np.where(z >= 0, 1, -1),
#             'sigmoid': lambda z: 1 / (1 + np.exp(-z)),
#             'tanh': lambda z: np.tanh(z),
#             'relu': lambda z: np.maximum(0, z),
#         }
        
#     # Prédit la classe pour un seul échantillon
#     def _predict_single(self, xi, activation_fn):
#         z = np.dot(xi, self.weights_) + self.bias_
#         if activation_fn == 'sign':
#             return 1 if z >= 0 else -1
#         else:
#             a = self.activations_[activation_fn](z)
#             return 1 if a >= 0.5 else -1

#     # Fit le modèle pour une seule fonction d'activation
#     def _fit_single_activation(self, X, y, act_name, use_early_stopping=True):
#         n_samples, n_features = X.shape
#         self.weights_ = np.zeros(n_features)
#         self.bias_ = 0

#         best_error = np.inf
#         best_weights = None
#         best_bias = None
#         errors = []
#         no_improve_count = 0

#         for epoch in range(self.n_iter):
#             epoch_errors = 0

#             if self.shuffle:
#                 idx = np.random.permutation(n_samples)
#                 X, y = X[idx], y[idx]

#             for xi, target in zip(X, y):
#                 prediction = self._predict_single(xi, act_name)
#                 update = self.lr * (target - prediction)
#                 if update != 0.0:
#                     self.weights_ += update * xi
#                     self.bias_ += update
#                     epoch_errors += 1

#             errors.append(epoch_errors)

#             if epoch_errors < best_error:
#                 best_error = epoch_errors
#                 best_weights = self.weights_.copy()
#                 best_bias = self.bias_
#                 no_improve_count = 0
#             else:
#                 no_improve_count += 1

#             if use_early_stopping and no_improve_count >= self.patience:
#                 if self.verbose:
#                     print(f"[{act_name}] Early stopping à époque {epoch+1}")
#                 break

#         label = f'{act_name}_early' if use_early_stopping else f'{act_name}_noearly'
#         self.errors_dict_[label] = errors

#     # Compare chaque activation avec et sans early stopping
#     def compare_early_vs_noearly(self, X, y):
#         """
#         Compare chaque activation avec et sans early stopping
#         """
#         self.errors_dict_.clear()
#         X = np.asarray(X)
#         y = np.where(y == 0, -1, 1)
#         for act in self.activations_.keys():
#             self._fit_single_activation(X, y, act, use_early_stopping=True)
#             self._fit_single_activation(X, y, act, use_early_stopping=False)

#     # Compare les différentes fonctions d'activation (early stopping ON ou OFF)
#     def compare_activations(self, X, y, use_early_stopping=True):
#         """
#         Compare les différentes fonctions d'activation (early stopping ON ou OFF)
#         """
#         self.errors_dict_.clear()
#         X = np.asarray(X)
#         y = np.where(y == 0, -1, 1)
#         for act in self.activations_.keys():
#             self._fit_single_activation(X, y, act, use_early_stopping=use_early_stopping)

#     # Affiche toutes les courbes enregistrées dans self.errors_dict_
#     def plot_all_convergences(self, title="Courbes de convergence"):
#         """
#         Affiche toutes les courbes enregistrées dans self.errors_dict_
#         """
#         plt.figure(figsize=(10, 6))
#         for label, errors in self.errors_dict_.items():
#             plt.plot(range(1, len(errors)+1), errors, marker='o', label=label)
#         plt.xlabel("Époques")
#         plt.ylabel("Nombre d'erreurs")
#         plt.title(title)
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()


import numpy as np
import matplotlib.pyplot as plt
from math import ceil


class Perceptron:
    """
    Implémentation du Perceptron (comparatif pédagogique) :
    - Early stopping (patience)
    - Shuffle des données (apprentissage en ligne)
    - Catalogue d'activations (sign, sigmoid, tanh, relu)
    - LOG des courbes d'erreurs/époque
    - Matrices de confusion + métriques pour chaque run (activation × early ON/OFF)

    Notes :
    - Les seuils décisionnels suivent ton implémentation d'origine :
        sign -> seuil z>=0 ; sigmoid/tanh/relu -> seuil a>=0.5
      (tu peux les surcharger via 'thresholds' si tu veux aligner tous les seuils sur z=0)
    """

    def __init__(self, learning_rate=0.01, n_iter=1000, patience=10, shuffle=True, verbose=False,
                 thresholds=None):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.patience = patience
        self.shuffle = shuffle
        self.verbose = verbose

        self.weights_ = None
        self.bias_ = 0.0
        self.best_error_ = np.inf
        self.best_weights_ = None
        self.best_bias_ = None

        # Logs & résultats
        self.errors_dict_ = {}   # label -> liste des erreurs par époque
        self.results_ = {}       # label -> dict(weights, bias, cm, metrics)

        # Activations
        self.activations_ = {
            'sign': lambda z: np.where(z >= 0, 1, -1),
            'sigmoid': lambda z: 1 / (1 + np.exp(-z)),
            'tanh': lambda z: np.tanh(z),
            'relu': lambda z: np.maximum(0, z),
        }
        # Seuils (décision) : par défaut, on respecte ton code d’origine
        self.thresholds_ = thresholds or {'sign': 0.0, 'sigmoid': 0.5, 'tanh': 0.5, 'relu': 0.5}

    # ---------- Utils de décision / prédiction ----------
    def _decision_from_activation(self, z, act_name):
        """
        Applique l'activation et renvoie y_pred dans {-1, +1} selon le seuil configuré.
        """
        if act_name == 'sign':
            return np.where(z >= self.thresholds_['sign'], 1, -1)
        a = self.activations_[act_name](z)
        thr = self.thresholds_[act_name]
        return np.where(a >= thr, 1, -1)

    def _predict_batch(self, X, weights, bias, act_name):
        z = X @ weights + bias
        return self._decision_from_activation(z, act_name)

    @staticmethod
    def _to01(y_pm):
        """Map {-1,+1} -> {0,1}"""
        return (y_pm == 1).astype(int)

    @staticmethod
    def _confusion_and_metrics(y_true01, y_pred01, eps=1e-12):
        """
        Calcule cm (2x2) et métriques. Retourne (cm, metrics_dict).
        cm = [[TN, FP],
              [FN, TP]]
        """
        # Comptages
        TN = np.sum((y_true01 == 0) & (y_pred01 == 0))
        FP = np.sum((y_true01 == 0) & (y_pred01 == 1))
        FN = np.sum((y_true01 == 1) & (y_pred01 == 0))
        TP = np.sum((y_true01 == 1) & (y_pred01 == 1))
        cm = np.array([[TN, FP],
                       [FN, TP]], dtype=int)

        acc = (TP + TN) / max(TP + TN + FP + FN, eps)
        prec = TP / max(TP + FP, eps)
        rec = TP / max(TP + FN, eps)
        f1 = 2 * prec * rec / max(prec + rec, eps)

        metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
        return cm, metrics

    # ---------- Entraînement d'un run ----------
    def _fit_single_activation(self, X, y_pm, act_name, use_early_stopping=True,
                               X_val=None, y_val_pm=None):
        """
        Entraîne pour une activation donnée.
        - y_pm est dans {-1,+1}
        - Si X_val/y_val_pm non fournis -> évalue sur train.
        - On évalue sur les *meilleurs* poids (best_weights/bias) et on restaure cet état.
        """
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0

        best_error = np.inf
        best_weights = self.weights_.copy()
        best_bias = float(self.bias_)
        errors = []
        no_improve_count = 0

        for epoch in range(self.n_iter):
            epoch_errors = 0

            if self.shuffle:
                idx = np.random.permutation(n_samples)
                X, y_pm = X[idx], y_pm[idx]

            for xi, target in zip(X, y_pm):
                # prédiction actuelle
                z = np.dot(xi, self.weights_) + self.bias_
                if act_name == 'sign':
                    pred = 1 if z >= self.thresholds_['sign'] else -1
                else:
                    a = self.activations_[act_name](z)
                    pred = 1 if a >= self.thresholds_[act_name] else -1

                update = self.lr * (target - pred)
                if update != 0.0:
                    self.weights_ += update * xi
                    self.bias_ += update
                    epoch_errors += 1

            errors.append(epoch_errors)

            if epoch_errors < best_error:
                best_error = epoch_errors
                best_weights = self.weights_.copy()
                best_bias = float(self.bias_)
                no_improve_count = 0
            else:
                no_improve_count += 1

            if use_early_stopping and no_improve_count >= self.patience:
                if self.verbose:
                    print(f"[{act_name}] Early stopping à époque {epoch+1}")
                break

        # --- RESTAURATION des meilleurs poids (cohérence d'évaluation) ---
        self.weights_ = best_weights
        self.bias_ = best_bias

        # --- Évaluation (train ou val) ---
        X_eval = X_val if X_val is not None else X
        y_eval_pm = y_val_pm if y_val_pm is not None else y_pm
        y_pred_pm = self._predict_batch(X_eval, self.weights_, self.bias_, act_name)

        y_true01 = self._to01(y_eval_pm)
        y_pred01 = self._to01(y_pred_pm)
        cm, metrics = self._confusion_and_metrics(y_true01, y_pred01)

        label = f'{act_name}_early' if use_early_stopping else f'{act_name}_noearly'
        self.errors_dict_[label] = errors
        self.results_[label] = {
            'weights': best_weights, 'bias': best_bias,
            'cm': cm, 'metrics': metrics,
            'n_epochs': len(errors)
        }

    # ---------- Protocoles comparatifs ----------
    def compare_early_vs_noearly(self, X, y, X_val=None, y_val=None):
        """
        Pour chaque activation : (early ON) + (early OFF).
        y peut être {0,1} ou {-1,+1}; on mappe en {-1,+1}.
        """
        self.errors_dict_.clear()
        self.results_.clear()

        X = np.asarray(X)
        y_pm = np.where(y == 0, -1, 1).astype(int)

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val)
            y_val_pm = np.where(y_val == 0, -1, 1).astype(int)
        else:
            y_val_pm = None

        for act in self.activations_.keys():
            self._fit_single_activation(X, y_pm, act, use_early_stopping=True,
                                        X_val=X_val, y_val_pm=y_val_pm)
            self._fit_single_activation(X, y_pm, act, use_early_stopping=False,
                                        X_val=X_val, y_val_pm=y_val_pm)

    def compare_activations(self, X, y, use_early_stopping=True, X_val=None, y_val=None):
        """
        Early fixé (ON/OFF), on balaie toutes les activations.
        """
        self.errors_dict_.clear()
        self.results_.clear()

        X = np.asarray(X)
        y_pm = np.where(y == 0, -1, 1).astype(int)

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val)
            y_val_pm = np.where(y_val == 0, -1, 1).astype(int)
        else:
            y_val_pm = None

        for act in self.activations_.keys():
            self._fit_single_activation(X, y_pm, act, use_early_stopping=use_early_stopping,
                                        X_val=X_val, y_val_pm=y_val_pm)

    # ---------- Visualisations ----------
    def plot_all_convergences(self, title="Courbes de convergence"):
        plt.figure(figsize=(10, 6))
        for label, errors in self.errors_dict_.items():
            plt.plot(range(1, len(errors)+1), errors, marker='o', label=label)
        plt.xlabel("Époques")
        plt.ylabel("Nombre d'erreurs")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, normalize=True, title="Matrices de confusion par run"):
        """
        Trace une matrice de confusion pour chaque 'label' (ordre trié).
        normalize=True -> normalisation par ligne (rappels par classe).
        """
        labels = sorted(self.results_.keys())  # tri pour un affichage stable
        n = len(labels)
        if n == 0:
            print("Aucun résultat disponible. Lance compare_... avant.")
            return
        ncols = min(4, n)
        nrows = int(ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.8*nrows))
        axes = np.array(axes).reshape(nrows, ncols)

        for i, lab in enumerate(labels):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            cm = self.results_[lab]['cm'].astype(float)

            if normalize:
                row_sums = cm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                cm_disp = cm / row_sums
            else:
                cm_disp = cm

            im = ax.imshow(cm_disp, cmap='Blues', vmin=0.0, vmax=1.0 if normalize else None)
            for (rr, cc), val in np.ndenumerate(cm_disp):
                ax.text(cc, rr, f"{val:.2f}" if normalize else int(cm[rr, cc]),
                        ha='center', va='center', fontsize=10)
            ax.set_title(lab)
            ax.set_xlabel("Prédit")
            ax.set_ylabel("Réel")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["0(-1)", "1(+1)"])
            ax.set_yticklabels(["0(-1)", "1(+1)"])

        # Enlève axes vides
        for j in range(n, nrows*ncols):
            r, c = divmod(j, ncols)
            fig.delaxes(axes[r, c])

        fig.suptitle(title, y=1.02, fontsize=14)
        fig.tight_layout()
        plt.show()

    def print_metrics_table(self):
        """
        Affiche un tableau texte des métriques par run.
        """
        if not self.results_:
            print("Aucun résultat. Lance compare_... avant.")
            return
        hdr = f"{'Run':<18} | {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} | {'Epochs':>6}"
        print(hdr)
        print("-"*len(hdr))
        for lab in sorted(self.results_.keys()):
            m = self.results_[lab]['metrics']
            n_ep = self.results_[lab]['n_epochs']
            print(f"{lab:<18} | {m['accuracy']:6.3f} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} | {n_ep:6d}")
