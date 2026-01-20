# Customer Churn Prediction (Telecom Domain)

## Problem Statement

Customer churn is a major challenge for telecom companies, as acquiring new customers is more expensive than retaining existing ones.  
This project predicts whether a customer is likely to churn based on service usage, contract details, and billing information.

## Dataset

- IBM Telco Customer Churn Dataset
- ~7,000 customer records
- Target variable: Churn (Yes / No)

## Approach

- Data cleaning and handling hidden missing values
- Exploratory Data Analysis (EDA) with business insights
- One-hot encoding of categorical features
- Addressed class imbalance using recall-focused evaluation
- Model comparison:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Deep Learning (MLP)
- Hyperparameter tuning using GridSearchCV

## Evaluation Metric

- Primary metric: Recall (Churn = 1)
- Reason: Missing a churn customer is more costly than targeting a loyal one

## Results

- Achieved ~78–79% recall in identifying churn customers
- Found that simpler models performed comparably to complex models
- Selected Logistic Regression for deployment due to interpretability

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras

## Key Learning

Model complexity does not always guarantee better performance; business-aligned metrics and interpretability matter.
