
# 🩺 Diabetes AI Prediction

An interactive machine learning web application that predicts the probability of diabetes based on patient health and medical parameters.

The project uses **Logistic Regression** for binary classification and **Streamlit** to provide a professional, interactive web dashboard.

---

## 🚀 Live Demo

🔗 **Live Application:**  
_Add your Streamlit deployment link here_

---

## 📌 Project Overview

Diabetes is a common chronic disease that can be influenced by several factors such as age, BMI, blood glucose level, HbA1c level, hypertension, heart disease, and smoking history.

This project uses machine learning to estimate whether a patient is likely to belong to the diabetes class based on these health-related features.

The application provides:

- Patient data input
- Diabetes prediction
- Probability estimation
- Risk classification
- Prediction history
- CSV export
- Model insights
- Interactive visual dashboard

> ⚠️ **Disclaimer:** This application is intended for educational and research purposes only. The model prediction is not a medical diagnosis and should not be used for clinical decision-making.

---

# ✨ Features

### 🧠 Machine Learning

- Logistic Regression classification
- Automated preprocessing pipeline
- Numerical feature scaling
- Categorical feature encoding
- Missing-value handling
- Probability prediction using `predict_proba()`

### 🎨 Interactive UI

- Modern dark-themed dashboard
- Glassmorphism-inspired interface
- Responsive layout
- Interactive patient input form
- Diabetes probability gauge
- Risk-level visualization
- Patient summary

### 📊 Prediction Analytics

- Diabetes / No Diabetes prediction
- Probability percentage
- Low / Moderate / High risk classification
- HbA1c and blood glucose summary
- Prediction progress indicator

### 📜 Prediction History

- Stores predictions during the current session
- Displays previous predictions
- Calculates prediction statistics
- Download history as CSV
- Clear prediction history

### 📚 Model Insights

- Explanation of Logistic Regression
- Machine learning pipeline visualization
- Feature descriptions
- Technical implementation details

---

# 🧠 Machine Learning Model

The project uses:

## Logistic Regression

Logistic Regression is a supervised machine learning algorithm commonly used for binary classification.

In this project, the model estimates the probability of a patient belonging to the diabetes class.

### Target Variable

```text
diabetes

0 → No Diabetes
1 → Diabetes
```

---

# 📋 Features Used

The model uses the following features:

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Age of the patient |
| `hypertension` | Binary | Whether the patient has hypertension |
| `heart_disease` | Binary | Whether the patient has heart disease |
| `bmi` | Numeric | Body Mass Index |
| `HbA1c_level` | Numeric | HbA1c blood sugar measurement |
| `blood_glucose_level` | Numeric | Blood glucose measurement |
| `gender` | Categorical | Patient gender |
| `smoking_history` | Categorical | Patient smoking history |

---

# ⚙️ Machine Learning Pipeline

The project uses a Scikit-learn pipeline to ensure that the same preprocessing is applied during both training and prediction.

```text
                    Patient Data
                         │
                         ▼
                 Data Validation
                         │
                         ▼
               ┌──────────────────┐
               │ Preprocessing     │
               └────────┬─────────┘
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
       Numeric Features      Categorical Features
             │                     │
             ▼                     ▼
      Median Imputation     Most Frequent Imputation
             │                     │
             ▼                     ▼
       StandardScaler       OneHotEncoder
             │                     │
             └──────────┬──────────┘
                        │
                        ▼
              Logistic Regression
                        │
                        ▼
               Diabetes Probability
                        │
                        ▼
                  Final Prediction
```

---

# 🛠️ Tech Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Logistic Regression
- StandardScaler
- OneHotEncoder
- SimpleImputer
- Pipeline
- ColumnTransformer

### Data Processing

- Pandas
- NumPy

### Model Serialization

- Joblib

### Web Application

- Streamlit

### Development

- VS Code
- Git
- GitHub

---

# 📁 Project Structure

```text
diabetes-ai-prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── diabetes_prediction_dataset.csv
│
└── models/
    └── diabetes_pipeline.joblib
```

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/diabetes-ai-prediction.git
```

Move into the project directory:

```bash
cd diabetes-ai-prediction
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

Usually:

```text
http://localhost:8501
```

---

# 🏋️ Train the Model

If you want to retrain the model using the dataset:

```bash
python train_model.py
```

Or specify the dataset and model paths:

```bash
python train_model.py \
    --data data/diabetes_prediction_dataset.csv \
    --model models/diabetes_pipeline.joblib
```

The trained model will be saved as:

```text
models/diabetes_pipeline.joblib
```

---

# 📊 Model Evaluation

During training, the application evaluates the model using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Example output:

```text
Accuracy:  0.xxxx
Precision: 0.xxxx
Recall:    0.xxxx
F1-score:  0.xxxx

Confusion matrix [[TN, FP], [FN, TP]]:
[[.... ....]
 [.... ....]]
```

The exact metrics depend on the dataset and training configuration.

---

# 🖥️ Application Pages

## 🏠 Prediction

The main dashboard allows users to enter:

- Age
- Gender
- BMI
- Hypertension
- Heart Disease
- Smoking History
- HbA1c Level
- Blood Glucose Level

The model then generates:

```text
Prediction
     +
Probability
     +
Risk Level
```

---

## 📊 Model Insights

This section explains:

- Logistic Regression
- Feature preprocessing
- Machine learning pipeline
- Input features
- Technical implementation

---

## 📜 Prediction History

The application keeps track of predictions made during the current session.

Users can:

- View previous predictions
- Analyze prediction statistics
- Download prediction history as CSV
- Clear the session history

---

## ℹ️ About Project

Provides information about:

- Project objective
- Machine learning algorithm
- Technology stack
- Input features
- Medical disclaimer

---

# 🔐 Data & Privacy

The application is designed as an educational machine learning demonstration.

Prediction history is maintained only within the current Streamlit session and can be exported by the user.

No medical diagnosis or treatment decision should be based solely on this application.

---

# 📈 Future Improvements

Potential future improvements include:

- [ ] Random Forest comparison
- [ ] XGBoost model comparison
- [ ] ROC-AUC visualization
- [ ] Precision-Recall curve
- [ ] Feature importance visualization
- [ ] SHAP-based explainability
- [ ] Model performance dashboard
- [ ] User authentication
- [ ] Database-backed prediction history
- [ ] PDF prediction reports
- [ ] Cloud deployment
- [ ] Automated model retraining
- [ ] Model monitoring

---

# 🎯 Learning Outcomes

This project demonstrates practical implementation of:

- Exploratory Data Analysis
- Data preprocessing
- One-hot encoding
- Feature scaling
- Train-test splitting
- Logistic Regression
- Classification metrics
- Confusion matrix
- Machine learning pipelines
- Model serialization
- Streamlit application development
- Git and GitHub
- ML model deployment

---

# ⚠️ Disclaimer

This project is created for **educational, academic, and demonstration purposes**.

The predictions generated by this application are statistical estimates from a machine learning model.

They are **not a medical diagnosis** and should not be used as a replacement for professional medical advice, diagnosis, or treatment.

Always consult a qualified healthcare professional for medical decisions.

---

# 👨‍💻 Author

**Divyansh Virole**

Information Technology Student  
Interested in Machine Learning, AI, Data Analytics, and Software Development.

---

## ⭐ If you found this project useful

Consider giving the repository a ⭐ on GitHub!
