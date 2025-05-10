# Heart Failure Readmission Prediction

ML pipeline for Veersa Hackathon 2026 (Use Case 4) by [Saksham]. Predicts 30-day readmissions for heart failure patients using MIMIC-III data. Contributions will be merged into team repo: https://github.com/aaanishaaa/HeartByte.

## Project Overview
This project analyzes patterns in 30-day readmissions for heart failure patients using the MIMIC-III database.

### Dataset Statistics
- Total heart failure patients: 10,272
- Total admissions analyzed: 16,756
- Average admissions per patient: 1.63
- 30-day readmission rate: 9.75%
- Total readmissions within 30 days: 1,634

### Data Preprocessing Pipeline
The project includes a robust preprocessing pipeline (`src/data_processing.py`) that handles:
- Table merging (patients, admissions, diagnoses, procedures)
- Missing value imputation (mean for numerical, mode for categorical)
- Feature encoding (one-hot encoding for categorical variables)
- Feature normalization (StandardScaler for numerical features)
- Readmission target creation (30-day threshold)
- Train/test splitting with stratification

### Key Findings
1. Comorbidity Analysis
   - Identified top comorbidities with highest readmission rates:
     - ICD-9 code 42843: 12.84% readmission rate
     - ICD-9 code 42841: 10.96% readmission rate
     - ICD-9 code 42833: 10.82% readmission rate

2. Clinical Patterns
   - Length of stay correlates with readmission risk
   - Previous admission history influences readmission probability
   - Multiple diagnoses associated with higher readmission rates

## Repository Structure
- `/notebooks/`: Jupyter notebooks for data analysis
  - `01_data_exploration.ipynb`: Initial data exploration
  - `02_data_analysis.ipynb`: Detailed analysis of readmission patterns
  - `03_model_training.ipynb`: Model training and evaluation
- `/src/`: Source code for data loading and preprocessing
  - `data_loader.py`: Functions to load and filter MIMIC-III data
  - `data_processing.py`: Data preprocessing and feature engineering
  - `modeling.py`: ML model training, evaluation and visualization
- `/data/`: MIMIC-III heart failure patient data
- `/models/`: Trained machine learning models

## Machine Learning Models
The project implements three different models with increasing complexity:
1. **Logistic Regression**: Baseline model with interpretable coefficients
2. **Random Forest**: Ensemble approach less prone to overfitting
3. **XGBoost**: Gradient boosting framework offering top performance

Each model is evaluated using metrics appropriate for imbalanced classification:
- Precision, Recall, F1 Score
- ROC-AUC and Confusion Matrix
- Feature importance analysis

## Getting Started
1. Set up Python environment with required dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run notebooks in sequence:
   - Start with data exploration (01_data_exploration.ipynb)
   - Continue with feature analysis (02_data_analysis.ipynb)
   - Train and evaluate models (03_model_training.ipynb)
3. Review model performance and insights

## Team
-
- Team repository: https://github.com/aaanishaaa/HeartByte