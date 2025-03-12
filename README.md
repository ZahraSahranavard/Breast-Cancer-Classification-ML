# Breast Cancer Classification with Machine Learning
## Overview
This project focuses on the classification of breast cancer using various machine learning algorithms. The goal is to build and compare different models to accurately predict whether a tumor is malignant or benign, based on the features extracted from the breast cancer dataset.

## Features

- **Data Preprocessing:** Includes data scaling and train/test splitting for robust model training.
- **Multiple Classification Models:** Implements several popular machine learning algorithms for classification.
- **Performance Evaluation:** Provides comprehensive performance metrics, including accuracy, precision, and recall.
- **Visualization:** Offers visual comparison of model performance using bar charts.
- **Reproducible Research:** The code is well-documented and structured for easy reproducibility.
## Dataset

The project utilizes the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) from scikit-learn. This dataset contains 569 instances with 30 features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image. The target variable indicates whether the tumor is malignant (0) or benign (1).

## Algorithms

The following machine learning algorithms are implemented and compared in this project:

-   **Gaussian Naive Bayes:** A probabilistic classifier based on Bayes' theorem.
-   **K-Nearest Neighbors (KNN):** A non-parametric algorithm that classifies data based on the majority class among its k nearest neighbors.
-   **Decision Tree:** A tree-like model that makes decisions based on feature values.
-   **Random Forest:** An ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting.
-   **Support Vector Machine (SVM):** A powerful algorithm that finds the optimal hyperplane to separate data into different classes.
-   **Logistic Regression:** A linear model that predicts the probability of a binary outcome.
-   **Artificial Neural Network (ANN):** A complex model inspired by the structure of the human brain, capable of learning intricate patterns.

## Dependencies

-   Python (>=3.7)
-   scikit-learn
-   matplotlib


You can install the required dependencies using pip:

```bash
pip install scikit-learn matplotlib 