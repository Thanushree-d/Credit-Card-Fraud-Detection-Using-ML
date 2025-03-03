# Credit Card Fraud Detection using Machine Learning
This project aims to detect fraudulent credit card transactions using machine learning techniques. It utilizes datasets containing transaction records labeled as fraudulent or legitimate and applies 
various ML algorithms to classify transactions accurately.

#Dataset
Dataset: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
Contains transaction data from European credit cardholders.
Highly imbalanced (fraud cases are only ~0.17% of total transactions).
Features include numerical values (PCA-transformed) and transaction details.

#Features & Preprocessing
Handled class imbalance using techniques like **SMOTE**(Synthetic Minority Over-sampling Technique).
Scaled features using **StandardScaler** to improve ML model convergence.
Split dataset into training & testing sets.

#Machine Learning Models Used
Logistic Regression

#Installation & Setup
## Requirements
Python 3.x
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
XGBoost (if used)

## Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt




