import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
from datetime import datetime
import joblib
import os
import time

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Random Forest training for Avocado Ripeness Classification")
    parser.add_argument("--n_estimators", type=int, default=100, 
                        help="Number of trees in the forest")
    parser.add_argument("--max_depth", type=int, default=None, 
                        help="Maximum depth of the trees")
    parser.add_argument("--min_samples_split", type=int, default=2, 
                        help="Minimum number of samples required to split a node")
    parser.add_argument("--min_samples_leaf", type=int, default=1, 
                        help="Minimum number of samples required at a leaf node")
    return parser.parse_args()

# Load dataset yang sudah dipreprocessing
def load_data():
    data = pd.read_csv('avocado_ripeness_dataset_preprocessed.csv')
    return data

def main():
    # Parse args
    args = parse_args()
    
    # Start time
    start_time = time.time()
    
    # Load data
    data = load_data()
    
    # Pisahkan fitur dan target
    X = data.drop('ripeness', axis=1)
    y = data['ripeness']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set experiment
    experiment_name = "Avocado_Ripeness_Classification_CI"
    mlflow.set_experiment(experiment_name)
    
    # Mulai tracking dengan MLflow
    with mlflow.start_run(run_name=f"RF_CI_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("dataset_size", len(data))
        mlflow.log_param("features_count", X.shape[1])
        
        # Buat model Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42
        )
        
        # Latih model
        training_start_time = time.time()
        rf_model.fit(X_train, y_train)
        training_time = time.time() - training_start_time
        
        # Evaluasi model
        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)
        
        # Hitung metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrik standard
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log metrik tambahan
        mlflow.log_metric("training_time_seconds", training_time)
        
        try:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            mlflow.log_metric("roc_auc_score", roc_auc)
        except:
            print("Warning: Could not calculate ROC AUC Score")
        
        # Feature importance metrics
        feature_importances = rf_model.feature_importances_
        mlflow.log_metric("max_feature_importance", max(feature_importances))
        mlflow.log_metric("min_feature_importance", min(feature_importances))
        mlflow.log_metric("feature_importance_std", np.std(feature_importances))
        
        # Class imbalance
        class_distribution = y_train.value_counts()
        class_imbalance = class_distribution.max() / class_distribution.min()
        mlflow.log_metric("class_imbalance_ratio", class_imbalance)
        
        # Buat confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=rf_model.classes_,
                    yticklabels=rf_model.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        
        # Log feature importance
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df[:10])
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        mlflow.log_artifact('feature_importance.png')
        
        # Log model
        mlflow.sklearn.log_model(rf_model, "random_forest_model")
        
        # Save model locally
        model_path = "models"
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(rf_model, f'{model_path}/rf_model.joblib')
        
        # Cetak hasil
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nTop 5 Important Features:")
        print(feature_importance_df[:5])
    
    total_time = time.time() - start_time
    print(f"Model training completed and logged to MLflow in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()