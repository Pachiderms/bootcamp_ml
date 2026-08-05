# ML Bootcamp by Pachiderms

This project is a hands-on, rigorous training program focused on mastering the fundamentals of Machine Learning (ML), inspired by the curriculum of École 42.

The primary goal of this bootcamp is to understand, design, and implement the core Machine Learning algorithms **from scratch** in Python using vectorized operations (with `NumPy` and `Pandas`), without relying on machine learning libraries such as `Scikit-Learn`.

---

## Table of Contents

1. [Overview](#-overview-)
2. [Module Breakdown & Learning Path](#-module-breakdown--learning-path)
   - [Day 00: Mathematical Foundations & Data Manipulation](#day-00-mathematical-foundations--data-manipulation)
   - [Day 01: Simple Linear Regression & Gradient Descent](#day-01-simple-linear-regression--gradient-descent)
   - [Day 02: Polynomial Regression, Normalization & Regularization](#day-02-polynomial-regression-normalization--regularization)
   - [Day 03: Logistic Regression & Classification Metrics](#day-03-logistic-regression--classification-metrics)
   - [Day 04: Multiclass Classification & Advanced Concepts](#day-04-multiclass-classification--advanced-concepts)
3. [Algorithm & Mathematical Formula Summary](#-algorithm--mathematical-formula-summary)
4. [Technologies & Coding Best Practices](#-technologies--coding-best-practices)
5. [Installation & Usage Guide](#-installation--usage-guide)
6. [Official Resources & Documentation](#-official-resources--documentation)

---

## Overview

This project follows a progressive, hands-on approach (*learning by doing*). Each module (or "Day") introduces new mathematical and algorithmic concepts that students translate into reusable Python classes and methods.

### Key Principles

- **Data Preparation**: Split datasets into $K$ subsets (folds) to evaluate a model's ability to generalize without introducing evaluation bias.
- **From-Scratch Implementation**: Rebuild every component of each model, including the `fit`, `predict`, `loss`, and `gradient` methods.
- **Systematic Vectorization**: Rely exclusively on NumPy matrix operations to maximize computational performance (no `for` loops over training samples).
- **Mathematical Rigor**: Explicitly compute gradients, partial derivatives, and cost functions.
- **Analysis & Visualization**: Evaluate model performance using learning curves, decision boundaries, residual plots, and other visualizations.

---

## Module Breakdown & Learning Path

### Day 00: Mathematical Foundations & Data Manipulation

- **`TinyStatistician`**: Build a basic statistical utility class that accepts both `list` and `np.ndarray` inputs.
  - Compute the mean ($\mu$), median, quartiles ($Q_1$, $Q_3$), variance ($\sigma^2$), and standard deviation ($\sigma$).
- **Exploratory Data Analysis (EDA)**:
  - Load and manipulate CSV datasets with Pandas.
  - Handle missing values (`NaN`), filtering, and aggregations.
  - Create visualizations with Matplotlib, including histograms, box plots, and scatter plots.

---

### Day 01: Simple Linear Regression & Gradient Descent

Implement the first supervised predictive model.

- **Hypothesis & Prediction**:

  $$\hat{y} = X \cdot \theta = \theta_0 + \theta_1 x$$

- **Cost Function (Mean Squared Error - MSE)**:

  $$J(\theta) = {1}/{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

- **Gradient Descent Optimization**:

  Compute the gradient vector and iteratively update the model parameters:

  $$\theta = \theta - \alpha {X}^{'T}({X}^{'} \cdot \theta - Y)$$

- **Model Evaluation**:
  - Coefficient of Determination ($R^2$ Score) to measure goodness of fit.
  - Root Mean Squared Error (RMSE).

---

### Day 02: Polynomial Regression, Normalization & Regularization

Extend linear regression to multidimensional datasets while addressing overfitting.

- **Multivariate Linear Regression**: Extend the model to input matrices of dimension $(m, n)$.
- **Polynomial Features**: Automatically generate higher-order features ($x$, $x^2$, $x^3$, $\dots$) to capture nonlinear relationships.
- **Feature Scaling**:
  - **Standardization (Z-score)**:

    $$x_{scaled} = \frac{x - \mu}{\sigma}$$

  - **Min-Max Normalization**:

    $$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

---

### Day 03: Logistic Regression & Classification Metrics

Transition from regression problems to binary classification.

- **Sigmoid Function ($\sigma$)**:

  $$g(z) = \frac{1}{1 + e^{-z}}$$

- **Probabilistic Hypothesis**:

  $$\hat{y} = g(X\theta) = P(y=1|X, \theta)$$

- **Binary Cross-Entropy Loss (Log Loss)**:

  $$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})\right ]$$

- **Comprehensive Classifier Evaluation**:
  - Confusion Matrix: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).
  - **Accuracy**:

    $$\frac{TP + TN}{TP + TN + FP + FN}$$

  - **Precision**:

    $$\frac{TP}{TP + FP}$$

  - **Recall (Sensitivity)**:

    $$\frac{TP}{TP + FN}$$

  - **F1 Score**:

    $$2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

---

### Day 04: Multiclass Classification & Advanced Concepts

- **Regularization ($L_1$ and $L_2$)**:
  - Add penalties to the magnitude of the model weights $\theta$ in the loss function to reduce overfitting.
  - **Ridge ($L_2$)**:

    $$J(\theta)_{Ridge} = J(\theta) + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

- **Multiclass Classification (One-vs-All / OvA)**:
  - Train $K$ independent binary logistic regression classifiers (one for each class).
  - Predict the final class using:

    $$\arg\max_k P(y=k|X)$$

- **K-Fold Cross-Validation**:
  - Split the dataset into $K$ folds to evaluate a model's generalization performance without introducing bias.

---

## Algorithm & Mathematical Formula Summary

| Concept / Model | Domain | Key Function / Formula | Purpose |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Regression | $\hat{Y} = X'\theta$ where $X' = [1, X]$ | Predict a continuous value |
| **MSE Loss** | Regression Loss | $J(\theta) = \frac{1}{2m} \|X'\theta - Y\|^2$ | Measure squared prediction error |
| **Gradient Descent** | Optimization | $\theta = \theta - \frac{\alpha}{m} X'^T (X'\theta - Y)$ | Minimize the cost function |
| **Logistic Regression** | Classification | $\hat{Y} = \sigma(X'\theta)$ | Predict a probability in $[0,1]$ |
| **Log Loss** | Classification Loss | $J(\theta) = -\frac{1}{m}[Y^T\log(\hat{Y}) + (1-Y)^T\log(1-\hat{Y})]$ | Measure classification error |
| **Ridge Regularization** | Regularization | $J(\theta)_{Ridge}=J(\theta)+\frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2$ | Reduce overfitting ($L_2$) |


## Technologies & Coding Best Practices

- **Language**: Python 3.8+
- **Allowed Libraries**:
  - `numpy`: Highly optimized matrix operations and vectorized numerical computations.
  - `pandas`: Loading, exploring, and preprocessing `.csv` datasets.
  - `matplotlib`: Data visualization, loss curves, and plotting utilities.
- **Strict Coding Guidelines**:
  - **No external machine learning libraries** (the use of `scikit-learn`, `tensorflow`, `torch`, etc. is prohibited).
  - Organize code into reusable classes (e.g., `MyLinearRegression`, `MyLogisticRegression`).
  - Follow the PEP 8 style guide.
  - Carefully manage NumPy array dimensions using `.reshape(-1, 1)` to prevent broadcasting errors.

---

## Installation & Usage Guide

### 1. Clone the Repository

```bash
git clone https://github.com/Pachiderms/bootcamp_ml.git
cd bootcamp_ml
```

### 2. Create and Activate a Virtual Environment, Then Install Dependencies

```bash
uv sync
source venv/bin/activate  # On Linux/macOS
# or venv\Scripts\activate on Windows
```

### 3. Run a Module or Test# ML Bootcamp

Each subdirectory contains its own demonstration Jupyter notebooks and a test file that can be executed with `pytest`.

```bash
cd tests/
pytest {file_name} # Or simply run `pytest` to execute the entire test suite
```

---

## Learning Outcomes

By completing this bootcamp, you will gain a **deep and intuitive understanding** of the fundamental algorithms behind Machine Learning. Implementing every algorithm from scratch develops a solid mastery of:

1. **Vectorization** and linear algebra for data science.
2. **Mathematical optimization**, including gradient descent and the learning rate ($\alpha$).
3. **Model evaluation and diagnostics**, including identifying underfitting and overfitting, and interpreting evaluation metrics.


## Official Resources & Documentation

Below is a list of the libraries used throughout this project, along with links to their official documentation.

- 🐍 **Python** (Primary Programming Language):
  - [PEP 8 Style Guide](https://peps.python.org/pep-0008/)

- 📐 **NumPy** (Matrix Operations & Vectorization):
  - [Official NumPy Documentation](https://numpy.org/doc/stable/)
  - [NumPy Quickstart Guide](https://numpy.org/doc/stable/user/quickstart.html)

- 🐼 **Pandas** (Data Analysis & Manipulation):
  - [Official Pandas Documentation](https://pandas.pydata.org/docs/)
  - [10 Minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)

- 📊 **Matplotlib** (2D/3D Data Visualization):
  - [Official Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
  - [Matplotlib Example Gallery](https://matplotlib.org/stable/gallery/index.html)

- ✅ **Pytest** (Test Automation):
  - [Official Pytest Documentation](https://docs.pytest.org/en/stable/)

- 🧠 **Additional Learning Resources** (For reference and implementation validation):
  - [What is Machine Learning? A Quick Introduction to Machine Learning and Its Use Cases](https://www.ibm.com/fr-fr/think/topics/machine-learning)
  - [Scikit-Learn Documentation](https://scikit-learn.org/stable/) *(For comparison purposes only.)*
  - [DeepLearningAI: Machine Learning Course by Andrew Ng](https://www.youtube.com/watch?v=vStJoetOxJg&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI)
