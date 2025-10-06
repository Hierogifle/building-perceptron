import numpy as np

class Perceptron :
    def __init__(self, n_features, learning_rate=0.01, n_iter=1000):
        #Initialisation des hyperparamètres
        self.learning_rate = learning_rate
        self.n_iter = n_iter

        #Initialisation des poids et biais
        self.weight = np.zeros(n_features)
        self.biais = 0
    
    def _activation(self, net_input) :
        """
        Fonction d'activation du perceptron
        Retroune 1 si la valeur calculer est >= 0
        Sinon 0
        """
        return np.where(net_input >= 0, 1, 0)

    def predict(self, X) :
        """
        Calcul de la valeur renvoyer par le modèle
        """
        return self._activation(np.dot(X, self.weight) + self.biais)
    
        
    def fitOnline(self, X, Y) :
        """
        Fonction d'entrainement du modèle
        """
        for epoch in range(self.n_iter) :
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            for Xi, target in zip(X_shuffled, Y_shuffled) :
                y_predict = self.predict(Xi)
                error = target - y_predict
                if error != 0 :
                    self.weight += self.learning_rate*error*Xi
                    self.biais += self.learning_rate*error
    
    def fitBatch(self, X, Y) :
        """
        Fonction d'entraînement du modèle par batch
        """
        for epoch in range(self.n_iter):
            yPredict = self.predict(X)
            errors = Y - yPredict
            if np.any(errors != 0):
                self.weight += self.learning_rate * np.dot(errors, X)
                self.biais += self.learning_rate * np.sum(errors)