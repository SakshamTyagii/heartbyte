"""
Data preprocessing utilities for heart failure readmission prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def preprocess_features(features_df, numerical_features=None, categorical_features=None):
    """
    Preprocess features for modeling by handling missing values, encoding categorical variables,
    and normalizing numerical features.
    
    Args:
        features_df (pd.DataFrame): Input features dataframe
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
    
    Returns:
        pd.DataFrame: Preprocessed features
    """
    # Deep copy to avoid modifying original
    df = features_df.copy()
    
    # Default feature groups if not provided
    if numerical_features is None:
        numerical_features = ['age', 'length_of_stay', 'admission_count', 
                            'num_diagnoses', 'num_procedures', 'days_since_last_admission']
    
    if categorical_features is None:
        categorical_features = ['gender'] if 'gender' in df.columns else []
    
    # Handle missing values
    numerical_imputer = SimpleImputer(strategy='mean')
    df[numerical_features] = numerical_imputer.fit_transform(df[numerical_features])
    
    if categorical_features:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])
    
    # Encode categorical variables
    if categorical_features:
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Normalize numerical features
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df

def prepare_model_data(features_df, target_col='is_readmission', test_size=0.2, random_state=42):
    """
    Prepare data for modeling by preprocessing features and splitting into train/test sets.
    
    Args:
        features_df (pd.DataFrame): Input features dataframe
        target_col (str): Name of target column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Preprocess features
    X = preprocess_features(features_df)
    y = features_df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def merge_tables(patients_df, admissions_df, diagnoses_df, procedures_df=None):
    """
    Merge multiple tables to create the base dataset for heart failure analysis.
    
    Args:
        patients_df (pd.DataFrame): Patient demographics
        admissions_df (pd.DataFrame): Hospital admissions
        diagnoses_df (pd.DataFrame): Patient diagnoses
        procedures_df (pd.DataFrame, optional): Patient procedures
    
    Returns:
        pd.DataFrame: Merged dataset
    """
    # Start with admissions and patients
    df = pd.merge(admissions_df, patients_df, on='subject_id', how='left')
    
    # Add diagnoses
    df = pd.merge(df, diagnoses_df, on=['subject_id', 'hadm_id'], how='left')
    
    # Add procedures if available
    if procedures_df is not None:
        df = pd.merge(df, procedures_df, on=['subject_id', 'hadm_id'], how='left')
    
    return df

def calculate_readmissions(admissions_df):
    """
    Calculate readmission flags for each admission based on 30-day threshold.
    
    Args:
        admissions_df (pd.DataFrame): Admissions dataframe with required columns
            - subject_id
            - admittime
            - dischtime
    
    Returns:
        pd.DataFrame: Admissions dataframe with added readmission columns
    """
    # Sort admissions by patient and date
    df = admissions_df.sort_values(['subject_id', 'admittime'])
    
    # Convert admission and discharge times to datetime
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dischtime'] = pd.to_datetime(df['dischtime'])
    
    # Calculate days until next admission for each patient
    df['next_admittime'] = df.groupby('subject_id')['admittime'].shift(-1)
    df['days_to_next_admission'] = (df['next_admittime'] - df['dischtime']).dt.total_seconds() / (24*60*60)
    
    # Mark readmissions (1 if readmitted within 30 days, 0 if not)
    df['is_readmission'] = (df['days_to_next_admission'] <= 30).astype(int)
    
    return df
