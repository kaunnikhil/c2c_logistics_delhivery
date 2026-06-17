"""
This module trains an XGBoost classifier to proactively predict SLA breaches
based strictly on dispatch time network constraints to prevent data leakage.
"""

import os
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure enterprise-grade logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SLABreachPredictor:
    def __init__(self, data_path: str, visuals_dir: str):
        self.data_path = data_path
        self.visuals_dir = visuals_dir
        self.model = None
        self.feature_names = None
        
        os.makedirs(self.visuals_dir, exist_ok=True)

    def load_and_preprocess(self):
        logger.info(f"Ingesting operational dataset from {self.data_path}")
        
        df = pd.read_csv(self.data_path)

        # TARGET LEAKAGE PREVENTION
        # We must drop variables that are not known at the time of dispatch.
        leaky_columns = [
            'Shipment_ID', 
            'Shipment_Date', 
            'Origin_Congestion_Delay_Hours', 
            'Destination_Congestion_Delay_Hours', 
            'Base_Transit_Delay_Hours', 
            'Weather_Transit_Delay_Hours', 
            'Actual_Transit_Time_Hours', 
            'SLA_Window_Hours'
        ]
        
        y = df['SLA_Breach']
        X = df.drop(columns=leaky_columns + ['SLA_Breach'], errors='ignore')

        # Encode categorical variables for the tree model
        categorical_cols = X.select_dtypes(include=['object']).columns
        logger.info(f"Encoding categorical dispatch features: {list(categorical_cols)}")
        X_encoded = pd.get_dummies(X, columns=categorical_cols)
        
        self.feature_names = X_encoded.columns.tolist()
        return X_encoded, y

    def train_evaluate(self, X: pd.DataFrame, y: pd.Series):
        """Trains the XGBoost model with dynamic imbalance handling."""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        logger.info(f"Training set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")

        # Dynamic class weighting for imbalanced logistics SLA breaches
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        logger.info(f"Calculated scale_pos_weight for class imbalance: {pos_weight:.2f}")

        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42,
            eval_metric='aucpr',
            early_stopping_rounds=30
        )

        logger.info("Training XGBoost Classifier...")
        eval_set = [(X_train, y_train), (X_test, y_test)]
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        self._generate_performance_report(y_test, y_pred, y_prob)
        self._plot_feature_importance()
        self._plot_confusion_matrix(y_test, y_pred)

    def _generate_performance_report(self, y_test, y_pred, y_prob):
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        print("\n" + "="*50)
        print("EXECUTIVE MODEL PERFORMANCE REPORT")
        print("="*50)
        print(f"ROC-AUC Score:          {roc_auc:.4f}")
        print(f"Precision-Recall AUC:   {pr_auc:.4f}")
        print("-" * 50)
        print(classification_report(y_test, y_pred, target_names=["On-Time", "SLA Breach"]))
        print("="*50 + "\n")

    def _plot_feature_importance(self):
        plt.figure(figsize=(10, 8))
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[-15:]
        
        plt.barh(range(len(indices)), importance[indices], color='#2C3E50', align='center')
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.xlabel('Relative Feature Importance (Gain)')
        plt.title('Predictive Drivers of SLA Breaches at Dispatch')
        
        plt.tight_layout()
        out_path = os.path.join(self.visuals_dir, 'xgboost_feature_importance.png')
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Feature importance plot saved to {out_path}")

    def _plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['On-Time', 'Breach'], yticklabels=['On-Time', 'Breach'])
        plt.ylabel('Actual Outcome')
        plt.xlabel('Predicted Outcome')
        plt.title('SLA Breach Prediction Matrix')
        
        out_path = os.path.join(self.visuals_dir, 'xgboost_confusion_matrix.png')
        plt.savefig(out_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    data_file = r"D:\cv_projects_2\c2c_logistics\data\network_operations_data.csv"
    visuals_folder = r"D:\cv_projects_2\c2c_logistics\visuals"
    
    predictor = SLABreachPredictor(data_path=data_file, visuals_dir=visuals_folder)
    X, y = predictor.load_and_preprocess()
    predictor.train_evaluate(X, y)