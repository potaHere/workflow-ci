import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for CI
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import argparse
import joblib
import os
from datetime import datetime

# Konfigurasi MLflow
mlflow.set_experiment("Avocado_Ripeness_Classification")

# Load dataset yang sudah dipreprocessing
def load_data():
    try:
        data = pd.read_csv('avocado_ripeness_dataset_preprocessed.csv')
        print(f"✅ Dataset loaded successfully: {data.shape}")
        return data
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

def main(n_estimators=100, max_depth=20, min_samples_split=5, min_samples_leaf=2):
    try:
        # Set experiment
        mlflow.set_experiment("Avocado_Ripeness_Classification")
        
        # Load data
        data = load_data()
        
        # Pisahkan fitur dan target
        X = data.drop('ripeness', axis=1)
        y = data['ripeness']
        
        print(f"Features shape: {X.shape}")
        print(f"Target classes: {y.unique()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Mulai tracking dengan MLflow
        with mlflow.start_run(run_name=f"RF_Params_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("min_samples_split", min_samples_split)
            mlflow.log_param("min_samples_leaf", min_samples_leaf)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            
            # Aktifkan autolog
            mlflow.sklearn.autolog()
            
            # Buat model Random Forest dengan parameter yang diberikan
            rf_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
            print("🚀 Training model...")
            # Latih model
            rf_model.fit(X_train, y_train)
            
            # Evaluasi model
            y_pred = rf_model.predict(X_test)
            
            # Hitung metrik
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log metrik
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Cetak hasil
            print(f"Model Parameters:")
            print(f"  - n_estimators: {n_estimators}")
            print(f"  - max_depth: {max_depth}")
            print(f"  - min_samples_split: {min_samples_split}")
            print(f"  - min_samples_leaf: {min_samples_leaf}")
            print(f"\nModel Performance:")
            print(f"  - Accuracy: {accuracy:.4f}")
            print(f"  - Precision: {precision:.4f}")
            print(f"  - Recall: {recall:.4f}")
            print(f"  - F1 Score: {f1:.4f}")
            
            # Buat confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=rf_model.classes_,
                        yticklabels=rf_model.classes_)
            plt.title(f'Confusion Matrix - RF (n_est={n_estimators}, max_depth={max_depth})')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log confusion matrix sebagai artifact
            mlflow.log_artifact('confusion_matrix.png')
            
            # Simpan model secara manual juga
            model_path = "random_forest_model.joblib"
            joblib.dump(rf_model, model_path)
            mlflow.log_artifact(model_path)
            
            # Log model menggunakan MLflow
            mlflow.sklearn.log_model(
                sk_model=rf_model,
                artifact_path="model",
                registered_model_name="AvocadoRipenessClassifier"
            )
            
            # Log dataset info
            mlflow.log_param("dataset_shape", f"{data.shape[0]}x{data.shape[1]}")
            mlflow.log_param("feature_count", len(X.columns))
            mlflow.log_param("target_classes", list(rf_model.classes_))
            
            print(f"\n✅ Model training completed and logged to MLflow")
            print(f"📊 Confusion matrix saved as 'confusion_matrix.png'")
            print(f"🎯 Model registered as 'AvocadoRipenessClassifier'")
            
            return mlflow.active_run().info.run_id
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Avocado Ripeness Classification Model')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of estimators for Random Forest')
    parser.add_argument('--max_depth', type=int, default=20, help='Maximum depth of trees')
    parser.add_argument('--min_samples_split', type=int, default=5, help='Minimum samples required to split an internal node')
    parser.add_argument('--min_samples_leaf', type=int, default=2, help='Minimum samples required to be at a leaf node')
    
    args = parser.parse_args()
    
    print(f"Starting model training with parameters:")
    print(f"  - n_estimators: {args.n_estimators}")
    print(f"  - max_depth: {args.max_depth}")
    print(f"  - min_samples_split: {args.min_samples_split}")
    print(f"  - min_samples_leaf: {args.min_samples_leaf}")
    
    try:
        # Run training with command line parameters
        run_id = main(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf
        )
        
        print(f"\n🎉 Training completed! MLflow Run ID: {run_id}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        exit(1)
