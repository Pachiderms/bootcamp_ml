# Résumé Complet du Projet : Bootcamp ML (Pachiderms / 42 Machine Learning)

Ce projet est une formation pratique et rigoureuse dédiée à l'apprentissage des fondamentaux du Machine Learning (ML), inspirée du cursus de l'École 42.

L'objectif principal de ce bootcamp est de comprendre, concevoir et implémenter **from scratch** les algorithmes essentiels du Machine Learning en Python sous forme vectorisée (via `NumPy` et `Pandas`), sans recourir à des bibliothèques clés en main telles que `Scikit-Learn`.

---

## Table des Matières
1. [Vue d'Ensemble & Philosophie](#-vue-densemble--philosophie)
2. [Détail des Modules & Progression](#-détail-des-modules--progression)
   - [Jour 00 : Fondations Mathématiques & Manipulation de Données](#jour-00--fondations-mathématiques--manipulation-de-données)
   - [Jour 01 : Régression Linéaire Simple & Descente de Gradient](#jour-01--régression-linéaire-simple--descente-de-gradient)
   - [Jour 02 : Régression Polynomiale, Normalisation & Régularisation](#jour-02--régression-polynomiale-normalisation--régularisation)
   - [Jour 03 : Régression Logistique & Métriques de Classification](#jour-03--régression-logistique--métriques-de-classification)
   - [Jour 04 : Classification Multi-classes & Concepts Avancés](#jour-04--classification-multi-classes--concepts-avancés)
3. [Récapitulatif des Algorithmes et Formules Mathématiques](#-récapitulatif-des-algorithmes-et-formules-mathématiques)
4. [Technologies & Bonnes Pratiques Code](#-technologies--bonnes-pratiques-code)
5. [Guide d'Installation et d'Exécution](#-guide-dinstallation-et-dexécution)
6. [Ressources & Documentations Officielles](#-ressources--documentations-officielles)

---

## Vue d'Ensemble & Philosophie

Le projet s'articule autour d'une approche progressive par la pratique (*learning by doing*). Chaque module (ou "Jour") introduit de nouveaux concepts mathématiques et algorithmiques que l'étudiant doit traduire en classes et méthodes Python réutilisables.

### Principes Clés :
- **Préparation de données** : Découpage des dataset en $K$ sous-ensembles (folds) pour évaluer la capacité de généralisation sans biais.
- **Implémentation *From Scratch*** : Reconstruction complète de la structure des modèles (`fit`, `predict`, `loss`, `gradient`).
- **Vectorisation Systématique** : Utilisation exclusive des opérations matricielles NumPy pour maximiser les performances computationnelles (interdiction des boucles `for` sur les échantillons).
- **Rigueur Mathématique** : Calcul explicite des gradients, des dérivées partielles et des fonctions de coût.
- **Analyse & Visualisation** : Diagnostic des modèles à l'aide de graphiques (courbes d'apprentissage, frontières de décision, résidus).


## Détail des Modules & Progression

### Jour 00 : Fondations Mathématiques & Manipulation de Données
- **`TinyStatistician`** : Création d'une classe d'outils statistiques élémentaires acceptant des `list` ou des `np.ndarray`.
  - Calcul de la moyenne ($\mu$), médiane, quartiles ($Q_1, Q_3$), variance ($\sigma^2$), et écart-type ($\sigma$).
- **Analyse Exploratoire de Données (EDA)** :
  - Manipulation de datasets CSV via Pandas.
  - Traitement des valeurs manquantes (`NaN`), filtrage et aggrégations.
  - Représentation graphique avec Matplotlib : Histogrammes, Boxplots et Scatter plots.

---

### Jour 01 : Régression Linéaire Simple & Descente de Gradient
Implémentation du premier modèle prédictif supervisé.
- **Hypothèse & Prédiction** :
  $$\hat{y} = X \cdot 	\theta = 	\theta_0 + 	\theta_1 x$$
- **Fonction de Coût (Mean Squared Error - MSE)** :
  $$J(\theta) = {1}/{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$
- **Optimisation par Descente de Gradient** :
  Calcul du vecteur de gradient et mise à jour itérative des paramètres :
  $$	\theta = \theta -  \alpha {X}^{'T}({X}^{'} \cdot \theta - Y)$$
- **Évaluation du Modèle** :
  - Coefficient de détermination ($R^2$ Score) pour mesurer la qualité de l'ajustement.
  - Erreur Quadratique Moyenne (RMSE).

---

### Jour 02 : Régression Polynomiale, Normalisation & Régularisation
Passage aux données multidimensionnelles et gestion des problèmes de surapprentissage (*overfitting*).
- **Régression Linéaire Multivariée** : Extension aux matrices d'entrée de dimension $(m, n)$.
- **Features Polynomiales** : Génération automatique de caractéristiques d'ordre supérieur ($x, x^2, x^3, \dots$) pour capturer des relations non-linéaires.
- **Mise à l'Échelle des Données (Feature Scaling)** :
  - **Standardisation (Z-score)** : $x_{scaled} = \frac{x - \mu}{\sigma}$
  - **Min-Max Normalization** : $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$

---

### Jour 03 : Régression Logistique & Métriques de Classification
Passage des problèmes de régression aux problèmes de classification binaire.
- **Fonction Sigmoïde ($\sigma$)** :
  $$g(z) = \frac{1}{1 + e^{-z}}$$
- **Hypothèse Probabiliste** : $\hat{y} = g(X	\theta) = P(y=1|X;	\theta)$
- **Fonction de Perte Binary Cross-Entropy (Log Loss)** :
  $$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})\right ]$$
- **Évaluation Complète d'un Classificateur** :
  - Matrice de Confusion : Vrais Positifs (TP), Faux Positifs (FP), Vrais Négatifs (TN), Faux Négatifs (FN).
  - **Accuracy** : $\frac{TP + TN}{TP + TN + FP + FN}$
  - **Precision** : $\frac{TP}{TP + FP}$
  - **Recall (Sensibilité)** : $\frac{TP}{TP + FN}$
  - **F1-Score** : $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$

---

### Jour 04 : Classification Multi-classes & Concepts Avancés
- **Régularisation ($L_1$ et $L_2$)** :
  - Ajout de pénalités sur la magnitude des poids $	\theta$ dans la fonction de perte pour éviter le surapprentissage.
  - **Ridge ($L_2$)** : $J(\theta)_{Ridge} = J(\theta) + \frac{\lambda}{2m} \sum_{j=1}^{n} 	\theta_j^2$
- **Classification Multi-classes (One-vs-All / OvA)** :
  - Entraînement de $K$ classificateurs régression logistique binaire distincts (un par classe).
  - Prédiction de la classe finale via $ arg\max_k P(y=k|X)$.
- **Validation Croisée (K-Fold Cross-Validation)** :
  - Implémentation du découpage du dataset en $K$ sous-ensembles (folds) pour évaluer la capacité de généralisation sans biais.
---

## Récapitulatif des Algorithmes et Formules Mathématiques

| Concept / Modèle | Domaine | Fonction Clé / Formule | Objectif |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Régression | $\hat{Y} = X'	\theta$ avec $X' = [1, X]$ | Prédire une valeur continue |
| **MSE Loss** | Coût Régression | $J(	\theta) = \frac{1}{2m} \|X'	\theta - Y\|^2$ | Mesurer l'erreur quadratique |
| **Gradient Descent** | Optimisation | $	\theta = 	\theta - \frac{ \alpha}{m} X'^T (X'	\theta - Y)$ | Minimiser la fonction de coût |
| **Logistic Regression**| Classification | $\hat{Y} = \sigma(X'	\theta)$ | Prédire une probabilité $[0, 1]$ |
| **Log Loss** | Coût Classification| $J(	\theta) = -\frac{1}{m} [Y^T \log(\hat{Y}) + (1-Y)^T \log(1-\hat{Y})]$ | Mesurer l'erreur de classification |
| **Ridge Regularization**| Régularisation | $J(\theta)_{Ridge} = J(\theta) + \frac{\lambda}{2m} \sum_{j=1}^{n} 	\theta_j^2$ | Réduire le surapprentissage ($L_2$) |

---

## 🛠 Technologies & Bonnes Pratiques Code

- **Langage** : Python 3.8+
- **Bibliothèques Autorisées** :
  - `numpy` : Manipulations matricielles et calculs vectoriels hautement optimisés.
  - `pandas` : Chargement, exploration et prétraitement des fichiers `.csv`.
  - `matplotlib` : Graphiques, courbes de perte et visualisations de données.
- **Règles Strictes de Code** :
  - **Aucune dépendance externe ML** (interdiction d'utiliser `scikit-learn`, `tensorflow`, `torch`).
  - Code modularisé sous forme de classes (ex: `MyLinearRegression`, `MyLogisticRegression`).
  - Respect des normes PEP 8.
  - Gestion rigoureuse des dimensions des tableaux NumPy avec la méthode `.reshape(-1, 1)` pour éviter les erreurs de broadcasting.

---

## Guide d'Installation et d'Exécution

### 1. Cloner le Dépôt
```bash
git clone https://github.com/Pachiderms/bootcamp_ml.git
cd bootcamp_ml
```

### 2. Créer et Activer un Environnement Virtuel et Installer les Dépendences
```bash
uv sync
source venv/bin/activate  # Sur Linux/macOS
# ou venv\Scripts\activate sur Windows
```

### 3. Exécuter un Module / Test
Chaque sous-dossier contient ses propres jupyter notebook de démonstration.

---

## Synthèse des Acquis

Ce bootcamp permet d'acquérir une **compréhension profonde et intuitive** des algorithmes fondamentaux du Machine Learning. En construisant chaque algorithme ligne par ligne, on développe une maîtrise complète de :
1. La **vectorisation** et l'algèbre linéaire appliquée à la science des données.
2. La dynamique de l'**optimisation mathématique** (descente de gradient, taux d'apprentissage $ \alpha$).
3. Le **diagnostic de modèle** (détection de l'underfitting/overfitting, interprétation des métriques).

## Ressources & Documentations Officielles

Voici la liste des bibliothèques utilisées dans ce projet avec les liens vers leurs documentations officielles respectives :

- 🐍 **Python** (Langage principal) :
  - [Guide du style PEP 8](https://peps.python.org/pep-0008/)

- 📐 **NumPy** (Calcul matriciel et vectorisation) :
  - [Documentation officielle NumPy](https://numpy.org/doc/stable/)
  - [Guide de démarrage NumPy pour les tableaux N-dimensionnels](https://numpy.org/doc/stable/user/quickstart.html)

- 🐼 **Pandas** (Analyse et manipulation de données) :
  - [Documentation officielle Pandas](https://pandas.pydata.org/docs/)
  - [Guide "10 Minutes to pandas"](https://pandas.pydata.org/docs/user_guide/10min.html)

- 📊 **Matplotlib** (Visualisation graphique 2D/3D) :
  - [Documentation officielle Matplotlib](https://matplotlib.org/stable/contents.html)
  - [Galerie d'exemples Matplotlib](https://matplotlib.org/stable/gallery/index.html)

- 🧠 **Ressources Théoriques Complémentaires** (Pour référence ou validation des implémentations) :
  - [What is ML? Quick introduction to ML algorithms use cases](https://www.ibm.com/fr-fr/think/topics/machine-learning)
  - [Scikit-Learn Documentation](https://scikit-learn.org/stable/) *(Utilisé à des fins de comparaison uniquement)*
  - [Stanford CS229: Machine Learning Course by Andrew Ng](https://www.youtube.com/watch?v=vStJoetOxJg&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI)