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
- `/src/`: Source code for data loading and preprocessing
- `/data/`: MIMIC-III heart failure patient data

## Getting Started
1. Set up Python environment with required dependencies
2. Run notebooks in sequence
3. Review analysis findings in `02_data_analysis.ipynb`

## Team
- [Saksham]: Project Lead
- Team repository: https://github.com/aaanishaaa/HeartByte