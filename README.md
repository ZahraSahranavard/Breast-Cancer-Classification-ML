# 🩺 Breast Cancer Classification with Machine Learning
![Heart Disease Prediction](https://github.com/ZahraSahranavard/Breast-Cancer-Classification-ML/blob/main/Image/Breast%20Cancer%20Classification.png)

## 🔹 Overview
This project demonstrates the classification of breast cancer using multiple **machine learning algorithms**. The main goal is to predict whether a tumor is **malignant** or **benign** based on features extracted from the Breast Cancer Wisconsin (Diagnostic) Dataset. The project also compares the performance of different models to identify the most effective approach.


## 📂 Project Structure

- **Data Preprocessing:** Includes train/test splitting, scaling with MinMaxScaler, and Check for missing values. <br><br>
- **Multiple Classification Models:** Implements popular algorithms for robust comparison:
  - Gaussian Naive Bayes
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Artificial Neural Network (ANN) <br><br>
- **Performance Evaluation:** Computes comprehensive metrics including:
  - Accuracy (Train/Test)
  - Precision
  - Recall
  - F1-Score
  - ROC Curve & AUC <br><br>
- **Visualization:** Clear bar charts and ROC curves for model comparison. <br><br>
- **Reproducibility:** Well-structured and documented code for easy understanding and reuse.
  
##  🗂 Dataset

The project uses the **[Breast Cancer Wisconsin (Diagnostic) Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)** from scikit-learn:

- **Instances:** 569
- **Features:** 30 numeric features describing characteristics of cell nuclei from FNA (Fine Needle Aspirate) images.
- **Target:** Binary classification  
  - Malignant (0)  
  - Benign (1)

## 🛠 Algorithms

| Algorithm | Description |
|-----------|-------------|
| Gaussian Naive Bayes (GNB) | Probabilistic classifier based on Bayes' theorem assuming Gaussian distribution of features. |
| K-Nearest Neighbors (KNN) | Classifies data based on the majority class among the k nearest neighbors. |
| Decision Tree (DT) | Splits data recursively to make classification decisions based on feature values. |
| Random Forest (RF) | Ensemble of decision trees to reduce overfitting and improve accuracy. |
| Support Vector Machine (SVM) | Finds the optimal hyperplane that maximizes margin between classes. |
| Logistic Regression (LR) | Linear model predicting the probability of a binary outcome. |
| Artificial Neural Network (ANN) | Multi-layer perceptron capable of learning complex non-linear pa

## 💻 Installation

Ensure you have **Python 3.7+** installed. Then install required dependencies:

```bash
pip install scikit-learn matplotlib seaborn pandas
```

## 🚀 Usage

### 1. Clone the repository:

```bash
git clone <your-repo-url>
```
```bash
cd <repo-folder>
```

### 2. Run the Jupyter Notebook:

```bash
jupyter notebook breast_cancer.ipynb
```

### 3. Follow the notebook to:

- Preprocess the dataset

- Train multiple models

- Evaluate metrics (accuracy, precision, recall, F1-score)

- Visualize model performance (bar charts and ROC curves)

## 📊 Results

- The notebook generates visual comparisons of all models on train and test sets.

- ROC curves and AUC values are plotted for each model.

- Provides a clear understanding of the best-performing algorithm for this dataset.

## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

