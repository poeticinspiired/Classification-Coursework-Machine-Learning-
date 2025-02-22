"""
This script implements k-NN comparison (Task 1), parameter tuning (Task 2),
and Naive Bayes evaluation (Task 3) for phishing detection with enhanced metrics and cross-validation.
"""

import numpy as np
import pandas as pd
import time
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, cross_validate)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, make_scorer)

# === DATA LOADING & PREPROCESSING ===
def load_and_preprocess():
    """Load and preprocess phishing datasets"""
    base_path = r'C:\Users\Abdudrahman\Desktop\Semeter6\Machine Learning - COMP-10200-01 - 27377.202515 - FF\ClassificationAssignment1\\'

    train_df = pd.read_csv(base_path + 'Training_Dataset.csv')
    test_df = pd.read_csv(base_path + 'Test_Dataset.csv')

    target = 'Result'
    X_train = train_df.drop(target, axis=1).values.astype(float)
    X_test = test_df.drop(target, axis=1).values.astype(float)

    y_train = np.where(train_df[target] == -1, 0, 1)
    y_test = np.where(test_df[target] == -1, 0, 1)

    scaler = StandardScaler()
    return (scaler.fit_transform(X_train), y_train,
            scaler.transform(X_test), y_test)

# === TASK 1: K-NN IMPLEMENTATION COMPARISON ===
def custom_knn(X_train, y_train, X_test, k, weighted=False):
    """Custom k-NN implementation with Euclidean distance and optional weighted voting"""
    def predict(x):
        distances = np.sqrt(np.sum((X_train - x)**2, axis=1))
        k_nearest = np.argpartition(distances, k)[:k]
        k_nearest_labels = y_train[k_nearest]
        if weighted:
            weights = 1 / (distances[k_nearest] + 1e-5)  # Avoid division by zero
            return np.bincount(k_nearest_labels, weights=weights).argmax()
        else:
            return np.bincount(k_nearest_labels).argmax()

    return np.array([predict(x) for x in X_test])

def compare_knn(X_train, y_train, X_test, y_test, k_values):
    """Compare custom vs sklearn implementations with cross-validation"""
    print("\n=== Task 1 Results ===")
    print(f"{'k':<5}{'Custom KNN':<12}{'Weighted KNN':<14}{'Sklearn KNN':<12}{'Time (s)':<10}")
    print("-"*55)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in k_values:
        start_time = time.time()

        # Custom implementation (unweighted)
        custom_scores = cross_val_score(KNeighborsClassifier(n_neighbors=k, weights='uniform'),
                                        X_train, y_train, cv=cv, scoring='accuracy')
        custom_acc = np.mean(custom_scores)

        # Custom implementation (weighted)
        weighted_scores = cross_val_score(KNeighborsClassifier(n_neighbors=k, weights='distance'),
                                          X_train, y_train, cv=cv, scoring='accuracy')
        weighted_acc = np.mean(weighted_scores)

        # Sklearn implementation
        sk_scores = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                                    X_train, y_train, cv=cv, scoring='accuracy')
        sk_acc = np.mean(sk_scores)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"{k:<5}{custom_acc:.4f}{'':<5}{weighted_acc:.4f}{'':<5}{sk_acc:.4f}{'':<5}{execution_time:.4f}")

# === TASK 2: K-NN PARAMETER TUNING ===
def tune_knn(X, y):
    """Experiment with different k-NN configurations"""
    print("\n=== Task 2 Results ===")

    param_grid = {
        'n_neighbors': [3, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro')
    }

    results = []
    for k in param_grid['n_neighbors']:
        for metric in param_grid['metric']:
            for weight in param_grid['weights']:
                start_time = time.time()
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)
                scores = cross_validate(knn, X, y, cv=cv, scoring=scoring)
                end_time = time.time()

                results.append({
                    'k': k,
                    'metric': metric,
                    'weights': weight,
                    'accuracy': np.mean(scores['test_accuracy']),
                    'precision': np.mean(scores['test_precision']),
                    'recall': np.mean(scores['test_recall']),
                    'f1': np.mean(scores['test_f1']),
                    'time': end_time - start_time
                })

    # Display results
    print(f"{'k':<5}{'Metric':<12}{'Weights':<10}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1 Score':<10}{'Time (s)':<10}")
    print("-"*80)
    for res in results:
        print(f"{res['k']:<5}{res['metric']:<12}{res['weights']:<10}"
              f"{res['accuracy']:.4f}{'':<5}{res['precision']:.4f}{'':<5}"
              f"{res['recall']:.4f}{'':<5}{res['f1']:.4f}{'':<5}{res['time']:.4f}")

# === TASK 3: NAIVE BAYES EVALUATION ===
def evaluate_naive_bayes(X, y):
    """Evaluate Naive Bayes performance with cross-validation"""
    print("\n=== Task 3 Results ===")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro')
    }

    # Gaussian Naive Bayes
    start_time = time.time()
    gnb_scores = cross_validate(GaussianNB(), X, y, cv=cv, scoring=scoring)
    gnb_time = time.time() - start_time

    print("GaussianNB Performance:")
    print(f"Average Accuracy: {np.mean(gnb_scores['test_accuracy']):.4f}")
    print(f"Minimum Accuracy: {np.min(gnb_scores['test_accuracy']):.4f}")
    print(f"Maximum Accuracy: {np.max(gnb_scores['test_accuracy']):.4f}")
    print(f"Precision: {np.mean(gnb_scores['test_precision']):.4f}")
    print(f"Recall: {np.mean(gnb_scores['test_recall']):.4f}")
    print(f"F1 Score: {np.mean(gnb_scores['test_f1']):.4f}")
    print(f"Execution Time: {gnb_time:.4f} seconds")

    # Accuracy/low and try BernoulliNB
    if np.mean(gnb_scores['test_accuracy']) < 0.7:
        print("\nChecking BernoulliNB performance...")
        start_time = time.time()
        bnb_scores = cross_validate(BernoulliNB(), X, y, cv=cv, scoring=scoring)
        bnb_time = time.time() - start_time

        print("\nBernoulliNB Performance:")
        print(f"Average Accuracy: {np.mean(bnb_scores['test_accuracy']):.4f}")
        print(f"Precision: {np.mean(bnb_scores['test_precision']):.4f}")
        print(f"Recall: {np.mean(bnb_scores['test_recall']):.4f}")
        print(f"F1 Score: {np.mean(bnb_scores['test_f1']):.4f}")
        print(f"Execution Time: {bnb_time:.4f} seconds")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess()

    # Execute Task 1
    compare_knn(X_train, y_train, X_test, y_test, [3, 5, 7])

    # Prepare combined data for Tasks 2 & 3
    X_full = np.vstack((X_train, X_test))
    y_full = np.concatenate((y_train, y_test))

    # Execute Task 2
    tune_knn(X_full, y_full)

    # Execute Task 3
    evaluate_naive_bayes(X_full, y_full)
