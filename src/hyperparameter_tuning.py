"""
Hyperparameter tuning for heart failure readmission models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, recall_score


def tune_logistic_regression(
    X_train, y_train, 
    cv=5, 
    scoring='f1', 
    n_iter=20,
    verbose=1
):
    """
    Tune hyperparameters for logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_iter: Number of parameter settings sampled
        verbose: Verbosity level
        
    Returns:
        Best model and search results
    """
    # Define parameter grid
    param_grid = {
        'C': np.logspace(-4, 4, 20),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced', None],
        'max_iter': [1000, 2000]
    }
    
    # Define scoring metrics
    scoring_metrics = {
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score),
        'recall': make_scorer(recall_score)
    }
    
    # Create base model
    lr = LogisticRegression(random_state=42)
    
    # Set up randomized search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=lr,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring_metrics,
        refit=scoring,
        random_state=42,
        verbose=verbose,
        n_jobs=-1
    )
    
    # Fit the search
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best {scoring} score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search


def tune_random_forest(
    X_train, y_train, 
    cv=5, 
    scoring='f1', 
    n_iter=20,
    verbose=1
):
    """
    Tune hyperparameters for random forest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_iter: Number of parameter settings sampled
        verbose: Verbosity level
        
    Returns:
        Best model and search results
    """
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 5, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt'],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Define scoring metrics
    scoring_metrics = {
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score),
        'recall': make_scorer(recall_score)
    }
    
    # Create base model
    rf = RandomForestClassifier(random_state=42)
    
    # Set up randomized search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring_metrics,
        refit=scoring,
        random_state=42,
        verbose=verbose,
        n_jobs=-1
    )
    
    # Fit the search
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best {scoring} score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search


def tune_xgboost(
    X_train, y_train, 
    cv=5, 
    scoring='f1', 
    n_iter=20,
    verbose=1
):
    """
    Tune hyperparameters for XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_iter: Number of parameter settings sampled
        verbose: Verbosity level
        
    Returns:
        Best model and search results
    """
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'scale_pos_weight': [1, 5, 10, 15, 20]  # For class imbalance
    }
    
    # Define scoring metrics
    scoring_metrics = {
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score),
        'recall': make_scorer(recall_score)
    }
    
    # Create base model
    xgb = XGBClassifier(random_state=42)
    
    # Set up randomized search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring_metrics,
        refit=scoring,
        random_state=42,
        verbose=verbose,
        n_jobs=-1
    )
    
    # Fit the search
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best {scoring} score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search


def run_grid_search(
    base_model, 
    param_grid,
    X_train, y_train, 
    cv=5, 
    scoring='f1',
    verbose=1
):
    """
    Run grid search for hyperparameter tuning.
    
    Args:
        base_model: Base model estimator
        param_grid: Parameter grid to search
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        scoring: Scoring metric
        verbose: Verbosity level
        
    Returns:
        Best model and search results
    """
    # Define scoring metrics
    scoring_metrics = {
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score),
        'recall': make_scorer(recall_score)
    }
    
    # Set up grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring_metrics,
        refit=scoring,
        verbose=verbose,
        n_jobs=-1
    )
    
    # Fit the search
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best {scoring} score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search
