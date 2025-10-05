import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """
    Implémentation avancée du Perceptron :
    - Early stopping
    - Shuffle des données
    - Test de plusieurs fonctions d'activation
    """

    def __init__(self, learning_rate=0.01, n_iter=1000, patience=10, shuffle=True, verbose=False):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.patience = patience
        self.shuffle = shuffle
        self.verbose = verbose

        self.weights_ = None
        self.bias_ = 0
        self.best_error_ = np.inf
        self.best_weights_ = None
        self.best_bias_ = None

        self.errors_dict_ = {}  # erreurs par activation
        self.activations_ = {
            'sign': lambda z: np.where(z >= 0, 1, -1),
            'sigmoid': lambda z: 1 / (1 + np.exp(-z)),
            'tanh': lambda z: np.tanh(z),
            'relu': lambda z: np.maximum(0, z),
        }

    def _predict_single(self, xi, activation_fn):
        z = np.dot(xi, self.weights_) + self.bias_
        if activation_fn == 'sign':
            return 1 if z >= 0 else -1
        else:
            a = self.activations_[activation_fn](z)
            return 1 if a >= 0.5 else -1

    def _fit_single_activation(self, X, y, act_name, use_early_stopping=True):
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0

        best_error = np.inf
        best_weights = None
        best_bias = None
        errors = []
        no_improve_count = 0

        for epoch in range(self.n_iter):
            epoch_errors = 0

            if self.shuffle:
                idx = np.random.permutation(n_samples)
                X, y = X[idx], y[idx]

            for xi, target in zip(X, y):
                prediction = self._predict_single(xi, act_name)
                update = self.lr * (target - prediction)
                if update != 0.0:
                    self.weights_ += update * xi
                    self.bias_ += update
                    epoch_errors += 1

            errors.append(epoch_errors)

            if epoch_errors < best_error:
                best_error = epoch_errors
                best_weights = self.weights_.copy()
                best_bias = self.bias_
                no_improve_count = 0
            else:
                no_improve_count += 1

            if use_early_stopping and no_improve_count >= self.patience:
                if self.verbose:
                    print(f"[{act_name}] Early stopping à époque {epoch+1}")
                break

        label = f'{act_name}_early' if use_early_stopping else f'{act_name}_noearly'
        self.errors_dict_[label] = errors

    def compare_early_vs_noearly(self, X, y):
        """
        Compare chaque activation avec et sans early stopping
        """
        self.errors_dict_.clear()
        X = np.asarray(X)
        y = np.where(y == 0, -1, 1)
        for act in self.activations_.keys():
            self._fit_single_activation(X, y, act, use_early_stopping=True)
            self._fit_single_activation(X, y, act, use_early_stopping=False)

    def compare_activations(self, X, y, use_early_stopping=True):
        """
        Compare les différentes fonctions d'activation (early stopping ON ou OFF)
        """
        self.errors_dict_.clear()
        X = np.asarray(X)
        y = np.where(y == 0, -1, 1)
        for act in self.activations_.keys():
            self._fit_single_activation(X, y, act, use_early_stopping=use_early_stopping)

    def plot_all_convergences(self, title="Courbes de convergence"):
        """
        Affiche toutes les courbes enregistrées dans self.errors_dict_
        """
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
