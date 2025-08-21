# AI-System-for-Predicting-Loan-Default-Risk
A Lightweight AI System for Predicting Loan Default Risk in Underbanked Segments
import pandas as pd
import sqlite3  # Can be swapped with PostgreSQL
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt

# Connect to local SQLite DB
conn = sqlite3.connect("finance.db")
query = """
SELECT age, income, loan_amount, term_months, credit_score, employment_years, has_defaulted
FROM borrowers;
"""
df = pd.read_sql(query, conn)

# Clean/check dataset
if df.isnull().values.any():
    df = df.dropna()

# Define input and target
X = df.drop("has_defaulted", axis=1)
y = df["has_defaulted"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Build model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
predictions = clf.predict(X_test)
print("📊 Model Evaluation:\n", classification_report(y_test, predictions))

# Explain model predictions using SHAP
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

print("🔍 Feature importance plot loading...")
shap.summary_plot(shap_values[1], X_test)
