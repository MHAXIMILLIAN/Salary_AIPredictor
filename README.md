# Salary Prediction System with Explainable Artificial Intelligence

## Abstract
This project presents a machine learning–based salary prediction system that estimates professional salaries using demographic, educational, occupational, and market-related features. The system integrates an end-to-end preprocessing and modeling pipeline with explainable artificial intelligence (XAI) techniques to ensure transparency and interpretability. A Streamlit-based web application is provided for real-time user interaction and inference.

---

## 1. Introduction
Salary estimation is a critical problem in labor economics, human resource analytics, and workforce planning. Traditional salary benchmarks often fail to capture individual-specific attributes and evolving market conditions. This project addresses this gap by developing a supervised learning model capable of predicting salaries from structured personal and job-related data, while also providing explainability through feature attribution methods.

---

## 2. System Overview
The system consists of three core components:
1. A trained machine learning pipeline for salary prediction
2. A preprocessing module that ensures feature consistency between training and inference
3. A Streamlit-based user interface for interactive prediction and visualization

The trained model is serialized and reused directly within the application to guarantee reproducibility.

---

## 3. Dataset and Features
The model is trained on a structured salary dataset containing the following features, all of which are reflected in the Streamlit input interface:

### Input Features
- **Age** (numeric)
- **Gender** (categorical)
- **Education Level** (High School, Bachelor's, Master's, PhD)
- **Job Title** (categorical / text)
- **Years of Experience** (numeric)
- **Industry** (categorical)
- **Location / City** (categorical)
- **Company Size** (Small, Medium, Large)
- **Market Index** (numeric, optional external adjustment)
- **Skill Indicators** (binary):
  - Python
  - SQL
  - Machine Learning
  - Data Visualization
  - Project Management

### Target Variable
- **Annual Salary**

---

## 4. Methodology
An end-to-end machine learning pipeline was implemented using Scikit-learn. The pipeline includes:
- Feature preprocessing (numerical scaling and categorical encoding)
- Model training using an ensemble-based regressor
- Model evaluation and selection
- Model persistence using `joblib`

The pipeline architecture ensures that preprocessing and prediction remain consistent across training and deployment environments.

---

## 5. Explainable AI (XAI)
To enhance transparency, the system incorporates SHAP (SHapley Additive exPlanations) for post-hoc model interpretability. SHAP values quantify the marginal contribution of each feature to an individual prediction, allowing users and evaluators to understand the driving factors behind salary estimates.

This approach aligns with current best practices in responsible and trustworthy AI.

---

## 6. Application Architecture
The Streamlit application performs the following steps:
1. Collects structured user inputs via an interactive UI
2. Constructs a feature-aligned DataFrame
3. Loads the trained machine learning pipeline
4. Generates salary predictions
5. Optionally displays feature importance information when supported by the model

---

## 7. Project Structure
```text
NEW_SalaryAIPredicator/
│
├── salary_app.py            # Streamlit application
├── best_salary_model.pkl    # Trained ML pipeline
├── Salary_Data.csv          # Dataset
├── requirements.txt         # Python dependencies
├── README.md
│
├── notebooks/               # Model development and training
└── payslips/                # Optional auxiliary documents
# Salary Prediction System with Explainable Artificial Intelligence

## Abstract
This project presents a machine learning–based salary prediction system that estimates professional salaries using demographic, educational, occupational, and market-related features. The system integrates an end-to-end preprocessing and modeling pipeline with explainable artificial intelligence (XAI) techniques to ensure transparency and interpretability. A Streamlit-based web application is provided for real-time user interaction and inference.

---

## 1. Introduction
Salary estimation is a critical problem in labor economics, human resource analytics, and workforce planning. Traditional salary benchmarks often fail to capture individual-specific attributes and evolving market conditions. This project addresses this gap by developing a supervised learning model capable of predicting salaries from structured personal and job-related data, while also providing explainability through feature attribution methods.

---

## 2. System Overview
The system consists of three core components:
1. A trained machine learning pipeline for salary prediction
2. A preprocessing module that ensures feature consistency between training and inference
3. A Streamlit-based user interface for interactive prediction and visualization

The trained model is serialized and reused directly within the application to guarantee reproducibility.

---

## 3. Dataset and Features
The model is trained on a structured salary dataset containing the following features, all of which are reflected in the Streamlit input interface:

### Input Features
- **Age** (numeric)
- **Gender** (categorical)
- **Education Level** (High School, Bachelor's, Master's, PhD)
- **Job Title** (categorical / text)
- **Years of Experience** (numeric)
- **Industry** (categorical)
- **Location / City** (categorical)
- **Company Size** (Small, Medium, Large)
- **Market Index** (numeric, optional external adjustment)
- **Skill Indicators** (binary):
  - Python
  - SQL
  - Machine Learning
  - Data Visualization
  - Project Management

### Target Variable
- **Annual Salary**

---

## 4. Methodology
An end-to-end machine learning pipeline was implemented using Scikit-learn. The pipeline includes:
- Feature preprocessing (numerical scaling and categorical encoding)
- Model training using an ensemble-based regressor
- Model evaluation and selection
- Model persistence using `joblib`

The pipeline architecture ensures that preprocessing and prediction remain consistent across training and deployment environments.

---

## 5. Explainable AI (XAI)
To enhance transparency, the system incorporates SHAP (SHapley Additive exPlanations) for post-hoc model interpretability. SHAP values quantify the marginal contribution of each feature to an individual prediction, allowing users and evaluators to understand the driving factors behind salary estimates.

This approach aligns with current best practices in responsible and trustworthy AI.

---

## 6. Application Architecture
The Streamlit application performs the following steps:
1. Collects structured user inputs via an interactive UI
2. Constructs a feature-aligned DataFrame
3. Loads the trained machine learning pipeline
4. Generates salary predictions
5. Optionally displays feature importance information when supported by the model

---

## 7. Project Structure
```text
NEW_SalaryAIPredicator/
│
├── salary_app.py            # Streamlit application
├── best_salary_model.pkl    # Trained ML pipeline
├── Salary_Data.csv          # Dataset
├── requirements.txt         # Python dependencies
├── README.md
│
├── notebooks/               # Model development and training
└── payslips/                # Optional auxiliary documents

---
8. Installation
Prerequisites

Python 3.11 or later

pip package manager

Steps

git clone https://github.com/your-username/NEW_SalaryAIPredicator.git
cd NEW_SalaryAIPredicator
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt


9. Running the Application
streamlit run salary_app.py


The application will launch in a web browser and allow users to input their professional details for salary prediction.

10. Limitations

Model accuracy depends on the representativeness of the training data

External market index integration is optional and subject to API availability

Predictions are estimates and should not be interpreted as contractual salary guarantees

11. Future Work

Integration of real-time labor market datasets

Fairness and bias auditing across demographic groups

Deployment on Streamlit Cloud or containerized environments

Extension to multi-country salary benchmarking

12. Author

Maximillian Onoyima

13. License

This project is intended for academic, educational, and research purposes.