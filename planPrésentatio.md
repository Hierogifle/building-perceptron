# 🧠 Projet : Implémentation avancée du Perceptron

## 📌 Contexte
Ce projet vise à implémenter et comparer plusieurs versions d’un Perceptron personnalisé dans un cadre de classification binaire, avec gestion du déséquilibre des classes, sélection de variables (Boruta), réduction de dimension (PCA), et comparaison avec d’autres modèles classiques de machine learning.

---

## 🎓 Cours : Le Perceptron
- Rappels théoriques
- Fonctionnement de la mise à jour des poids
- Limites du Perceptron classique (linéarité, absence de convergence sur certains jeux)

---

## 🔍 Inspection initiale des données
### 📂 Informations sur le dataset brut
- Dimensions, types, aperçu général (`df.info()`, `df.head()`)

### 🎯 Analyse de la cible (`target`)
- Distribution des classes (déséquilibre à noter)
- Encodage éventuel si nécessaire

---

## 📊 Analyse exploratoire des données (EDA)
### 📈 Visualisations :
- Distribution des features (histogrammes)
- Outliers (boxplots, violin plots)
- Matrice de corrélation
- Heatmap ciblée sur la `target`
- Courbes cumulatives si utile

### 🧹 Nettoyage :
- Vérification des valeurs manquantes
- Recherche de doublons
- Traitement des valeurs aberrantes

---

## 🛠️ Feature Engineering
- Normalisation / Standardisation (choix en fonction de la distribution)
- Encodage éventuel
- Création de nouvelles features si utile

---

## ✂️ Split des données
- `X_train`, `X_test`, `y_train`, `y_test` avec stratification
- Visualisation éventuelle des proportions

---

## 🌲 Sélection des variables avec Boruta
- Modèles testés : `RandomForest`, `ExtraTrees`
- Choix de l'estimateur final
- Visualisation des features retenues
- Analyse : faut-il faire l’union des résultats ?
- Création du nouveau `X_train_boruta`

---

## 🧪 Réduction de dimension avec PCA
- Objectif : réduire la colinéarité, améliorer convergence
- Explication détaillée du fonctionnement
- Choix du nombre de composantes (variance expliquée)
- Graphe des composantes
- Projection de `X_train_boruta` ➝ `X_train_pca`

---

## 🧱 Classe `Perceptron` (Custom)
- Implémentation from scratch en Python
- Méthodes : `fit`, `predict`, `net_input`, `plot_errors`

---

## 🛑 Perceptron avec Early Stopping
- Ajout de la patience
- Suivi du meilleur score
- Affichage de la convergence
- Éviter le surapprentissage

---

## ⚡ Perceptron avec plusieurs fonctions d’activations
- `step`, `sigmoid`, `tanh`, `relu`
- Résultats obtenus sur chaque activation
- Tableau comparatif F1 / Accuracy / Recall / Balanced Accuracy
- Graphe comparatif

---

## 🧮 Matrices de confusion
- Affichage pour chaque version de Perceptron
- Interprétation : erreurs de faux négatifs / faux positifs

---

## 📏 Évaluation par métriques adaptées
- Pourquoi l’accuracy n’est pas suffisante ?
- Utilisation de :
  - F1-score
  - Recall
  - Precision
  - Balanced Accuracy
  - ROC AUC (optionnel)

---

## ⚖️ Comparaison avec d'autres modèles
- Perceptron `sklearn`
- Régression Logistique
- MLP (Multi-Layer Perceptron)
- SVM (SVC)
- Tous entraînés sur le même `X_train_pca`
- Tableau comparatif + graphe en barres

---

## 📊 Graphique de contribution des variables
- Interprétation des composantes principales
- Quels features influencent le plus chaque PC

---

## ✅ Conclusion
- Quelle version de Perceptron est la plus performante ?
- Impact du PCA et de Boruta sur la performance
- Apports du early stopping
- Intérêt des différentes fonctions d’activation

---

## 🚧 Limites et Perspectives
- PCA non supervisé ➝ potentiel biais
- Boruta dépend fortement de l’estimateur
- Possibilité d’intégrer grid search, cross-validation
- Ouverture vers du Deep Learning (TensorFlow, PyTorch)

---

## 📎 Annexes
- Courbes de convergence
- Répartition des classes
- Tests statistiques complémentaires
