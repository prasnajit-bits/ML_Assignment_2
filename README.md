# Machine Learning Assignment 2

## Bank Marketing Classification

---

## 1. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for predicting whether a bank customer will subscribe to a term deposit.

The project implements five classification algorithms on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

The models are evaluated using the following performance metrics:

* **Accuracy**
* **AUC Score**
* **Precision**
* **Recall**
* **F1 Score**
* **Matthews Correlation Coefficient (MCC)**

An interactive **Streamlit web application** is also developed to allow users to upload test data, select a machine learning model, generate predictions, and view evaluation results.

---

## 2. Dataset Description

### Dataset Name

**Bank Marketing Dataset**

### Source

**Kaggle**

**Dataset Link:**
https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset

### Dataset Objective

The dataset contains information about customers of a bank and their interactions with a marketing campaign. The target variable indicates whether the customer subscribed to a term deposit.

### Dataset Size

* **Number of instances:** 11,162
* **Number of input features:** 16
* **Target variable:** `deposit`
* **Classification type:** Binary Classification

The dataset satisfies the assignment requirement of at least **12 features** and **500 instances**.

### Target Variable

The target variable is:

```text
deposit
```

Target values:

* `yes` → Customer subscribed to a term deposit
* `no` → Customer did not subscribe to a term deposit

### Target Distribution

| Target | Count |
| :----- | ----: |
| No     | 5,873 |
| Yes    | 5,289 |

The target distribution is reasonably balanced, with approximately **52.61%** of instances belonging to the `no` class and **47.39%** belonging to the `yes` class.

### Numerical Features

The dataset contains the following numerical features:

* `age`
* `balance`
* `day`
* `duration`
* `campaign`
* `pdays`
* `previous`

### Categorical Features

The dataset contains the following categorical features:

* `job`
* `marital`
* `education`
* `default`
* `housing`
* `loan`
* `contact`
* `month`
* `poutcome`

### Data Quality

* **Missing values:** 0
* **Duplicate rows:** 0

### Preprocessing

The following preprocessing steps were performed:

1. The target variable `deposit` was separated from the input features.
2. The target values were converted to binary values:

   * `no` → `0`
   * `yes` → `1`
3. The dataset was divided into training and testing sets using an **80:20 split**.
4. **Stratified splitting** was used to preserve the class distribution.
5. Numerical features were standardized using `StandardScaler`.
6. Categorical features were converted into numerical representations using `OneHotEncoder`.
7. The preprocessing steps were implemented using a `ColumnTransformer`.
8. The final processed dataset contained **51 features** after one-hot encoding.

---

## 3. GitHub Repository Link

**GitHub Repository:**

> **https://github.com/prasnajit-bits/ML_Assignment_2.git**


### Repository Structure

```text
ML_Assignment_2/
│
├── data/
│   └── bank.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── random_forest.pkl
│
├── test_data.csv
├── model_results.csv
├── main.py
├── requirements.txt
└── README.md
```

---

## 4. Models Used

The following five machine learning classification models were implemented:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN)**
4. **Gaussian Naive Bayes**
5. **Random Forest Classifier (Ensemble)**

### 4.1 Model Comparison

| ML Model Name                |   Accuracy |        AUC |  Precision |     Recall |   F1 Score |        MCC |
| :--------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression          |     0.8262 |     0.9071 |     0.8278 |     0.7996 |     0.8135 |     0.6513 |
| Decision Tree                |     0.7944 |     0.7933 |     0.7894 |     0.7722 |     0.7807 |     0.5874 |
| KNN                          |     0.8173 |     0.8796 |     0.8199 |     0.7873 |     0.8033 |     0.6333 |
| Naive Bayes                  |     0.7201 |     0.8042 |     0.7837 |     0.5652 |     0.6568 |     0.4472 |
| **Random Forest (Ensemble)** | **0.8621** | **0.9193** | **0.8301** | **0.8913** | **0.8596** | **0.7262** |

---

## 5. Observations on Model Performance

| ML Model Name                | Observation about Model Performance                                                                                                                                    |
| :--------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**      | Achieved **82.62% accuracy** and **90.71% AUC**, providing a strong baseline.                                                                                          |
| **Decision Tree**            | Achieved **79.44% accuracy** and **79.33% AUC**, with lower overall performance than Logistic Regression, KNN, and Random Forest.                                      |
| **KNN**                      | Achieved **81.73% accuracy** and **87.96% AUC**, showing good performance but lower performance than Random Forest.                                                    |
| **Naive Bayes**              | Achieved **72.01% accuracy** and **80.42% AUC**. Its **56.52% recall** was the lowest among the five models.                                                           |
| **Random Forest (Ensemble)** | Achieved the **best overall performance**, with **86.21% accuracy**, **91.93% AUC**, **83.01% precision**, **89.13% recall**, **85.96% F1 score**, and **72.62% MCC**. |

---

## 6. Overall Winner

### Random Forest (Ensemble)

**Random Forest is the overall winner for this dataset because it achieved the highest score across all six evaluation metrics.**

| Metric        | Random Forest Score |
| :------------ | ------------------: |
| **Accuracy**  |          **86.21%** |
| **AUC**       |          **91.93%** |
| **Precision** |          **83.01%** |
| **Recall**    |          **89.13%** |
| **F1 Score**  |          **85.96%** |
| **MCC**       |          **72.62%** |

The Random Forest model demonstrates the strongest overall classification performance. In particular, its high **recall of 89.13%** indicates that it successfully identifies a large proportion of customers who subscribed to a term deposit. Its **AUC of 91.93%** also indicates strong ability to distinguish between the two target classes.

Therefore, based on the reported evaluation metrics, **Random Forest is selected as the best-performing model for the Bank Marketing Classification task**.
