# End-to-End Machine Learning Toy Project

This repository contains an **end-to-end machine learning toy project** built as part of the **100 Days of Machine Learning** series by CampusX.  
The goal of this project is to understand the **complete ML pipeline** — from raw data to a trained and evaluated model — using a simple and interpretable approach.

---

## 📌 Project Overview

This project demonstrates how a real-world machine learning workflow is structured, including:

- Data loading and exploration
- Data preprocessing
- Feature selection
- Model training using Logistic Regression
- Model evaluation
- Saving the trained model for future use

The project is intentionally kept simple ("toy project") to focus on **concept clarity and workflow understanding** rather than performance optimization.

---

## 🧠 Problem Statement

To build a **binary classification model** that predicts an outcome based on input features using a clean and reproducible machine learning pipeline.

---

## 🗂️ Dataset

- The dataset used is a **small, structured dataset** suitable for binary classification.
- It contains:
  - Numerical features
  - A target column representing two classes (0/1)

*(Exact dataset name is not important here as the focus is on the ML pipeline rather than domain complexity.)*

---

## ⚙️ Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Pickle

---

## 🔄 Project Workflow

### 1. Data Loading
- Dataset is loaded using Pandas
- Basic inspection (`head`, `shape`, `info`) is performed

### 2. Data Preprocessing
- Handling missing values
- Feature selection
- Train-test split

### 3. Model Selection
- **Logistic Regression** is used because:
  - It is simple and interpretable
  - Works well for binary classification
  - Ideal for learning end-to-end ML flow

### 4. Model Training
- Model is trained on the training dataset
- Default hyperparameters are used for simplicity

### 5. Model Evaluation
- Accuracy score is calculated on test data
- Performance is analyzed to understand model behavior

### 6. Model Serialization
- Trained model is saved using `pickle`
- Enables reuse without retraining

---

## 📈 Results

- The model achieves reasonable accuracy for a toy dataset
- The focus is on **pipeline correctness**, not metric maximization
