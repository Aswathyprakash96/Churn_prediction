# 📊 Churn Prediction Project

This project focuses on predicting customer churn using machine learning techniques. The dataset includes customer usage details such as call minutes, charges, customer service calls, and subscription plans.

---

## 📁 Dataset Features

| Column Name             | Description                               |
|-------------------------|-------------------------------------------|
| `State`                 | US state abbreviation                     |
| `Account length`        | Duration of account in days               |
| `Area code`             | Area code assigned                        |
| `International plan`    | Whether international plan is active      |
| `Voice mail plan`       | Whether voice mail plan is active         |
| `Total day minutes`     | Daytime call minutes                      |
| `Total day calls`       | Number of daytime calls                   |
| `Total day charge`      | Total charge during the day               |
| `Total eve minutes`     | Evening call minutes                      |
| `Total eve calls`       | Evening call count                        |
| `Total eve charge`      | Total evening charges                     |
| `Total night minutes`   | Night call minutes                        |
| `Total night calls`     | Number of night calls                     |
| `Total night charge`    | Total night charges                       |
| `Total intl minutes`    | International call minutes                |
| `Total intl calls`      | International call count                  |
| `Total intl charge`     | International call charges                |
| `Customer service calls`| Calls made to customer service            |
| `Churn`                 | Target variable (1 = churned, 0 = not)    |

---

## 🧹 Data Preprocessing

- Cleaned string columns (e.g., `International plan`, `Voice mail plan`) and mapped to binary
- One-hot encoded `State` column
- Outlier detection (IQR method)
- Scaled numerical features using `StandardScaler`

---

## ⚖️ Handling Class Imbalance

Used **SMOTE** (Synthetic Minority Oversampling Technique) to balance the dataset since churned customers are fewer.

---

## 🔍 Feature Selection

Used `SelectKBest` with `chi2` score function to select top k features that are most relevant for churn prediction.

```python
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(score_func=chi2, k=10)
X_selected = selector.fit_transform(X_resampled, y_resampled)

🧠 Model Training
Trained multiple classifiers:
RandomForestClassifier
LogisticRegression
Support Vector Classifier (SVC)


📈 Model Evaluation
Used the following metrics:
Accuracy
Confusion Matrix
Precision, Recall, F1-score
