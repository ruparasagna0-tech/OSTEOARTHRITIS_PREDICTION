# 🧠 Automated Classification of Osteoarthritis using Machine Learning

## 📌 Overview

This project focuses on the automated classification of Osteoarthritis (OA) and healthy individuals using gene expression data. The goal is to develop an efficient and objective diagnostic support system using machine learning techniques.

Osteoarthritis is a degenerative joint disease that is often diagnosed late due to subjective assessment methods. This project leverages machine learning to improve early detection and classification accuracy.

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Removed missing and duplicate values
* Normalized gene expression values
* Applied feature selection using Principal Component Analysis (PCA)

  * Retained 95% variance to reduce dimensionality

### 2. Model Training

The following machine learning models were implemented:

* Random Forest
* Support Vector Machine (SVM)
* Logistic Regression
* K-Nearest Neighbors (KNN)

### 3. Validation

* Used **10-fold cross-validation** for robust evaluation
* Evaluated models using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix
  * ROC Curve

---

## 📊 Results

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Random Forest       | 98%      | 0.98     |
| KNN                 | 98%      | 0.98     |
| Logistic Regression | 94%      | 0.95     |
| SVM                 | 90%      | 0.91     |

* Random Forest and KNN achieved the best performance
* ROC-AUC score reached **1.0**, indicating excellent classification ability

---

## 🧰 Tech Stack

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

---

## 📈 Key Features

* Handles high-dimensional gene expression data
* Uses PCA for dimensionality reduction
* Compares multiple ML models
* Provides detailed evaluation metrics
* Suitable for healthcare diagnostic support systems

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/osteoarthritis-classification-ml
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the project:

```bash
python main.py
```

---

## 🔮 Future Scope

* Apply deep learning models for improved performance
* Use larger and external datasets for validation
* Explore advanced feature selection techniques
* Deploy as a clinical decision support tool

---

## 👩‍💻 Author

* K. Rupa Rasagna

---

## 📬 Contact

For any queries, feel free to reach out via email.
