# random_forest_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def run_random_forest_model(data, labels):
    # Check that the number of samples in data and labels match
    assert data.shape[0] == len(labels), "Data and labels do not have the same number of samples!"

    # Initialize RandomForestClassifier model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Set up 10-fold cross-validation
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Perform cross-validation and get accuracy scores
    cv_results = cross_val_score(rf_model, data, labels, cv=kf, scoring='accuracy')
    
    # Display mean accuracy and standard deviation
    print(f"Random Forest Model Accuracy (10-fold CV): {cv_results.mean():.4f} ± {cv_results.std():.4f}")

    # Fit the model on the full dataset
    rf_model.fit(data, labels)
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, rf_model.predict_proba(data)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest Model ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
