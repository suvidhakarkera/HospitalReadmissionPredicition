"""
Hospital Readmission Prediction - Complete ML Pipeline
Dataset: UCI Diabetes 130-US Hospitals Dataset
Goal: Predict 30-day hospital readmission risk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class ReadmissionPredictor:
    """
    Complete pipeline for hospital readmission prediction
    """
    
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def load_data(self, filepath):
        """Load and perform initial data exploration"""
        print("=" * 60)
        print("STEP 1: LOADING AND EXPLORING DATA")
        print("=" * 60)
        
        df = pd.read_csv(filepath)
        
        print(f"\nDataset Shape: {df.shape}")
        print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
        
        print("\n--- First Few Rows ---")
        print(df.head())
        
        print("\n--- Data Types ---")
        print(df.dtypes.value_counts())
        
        print("\n--- Missing Values ---")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing[missing > 0],
            'Percentage': missing_pct[missing > 0]
        }).sort_values('Percentage', ascending=False)
        print(missing_df)
        
        return df
    
    def preprocess_data(self, df):
        """Clean and preprocess the dataset"""
        print("\n" + "=" * 60)
        print("STEP 2: DATA PREPROCESSING")
        print("=" * 60)
        
        # Create a copy
        df_clean = df.copy()
        
        # Target variable: readmitted within 30 days
        # Typical encoding: '<30' = 1 (readmitted), '>30' or 'NO' = 0
        if 'readmitted' in df_clean.columns:
            df_clean['readmitted_30days'] = (df_clean['readmitted'] == '<30').astype(int)
            print(f"\nTarget Distribution:")
            print(df_clean['readmitted_30days'].value_counts())
            print(f"Readmission Rate: {df_clean['readmitted_30days'].mean():.2%}")
        
        # Handle missing values
        # Drop columns with >50% missing data
        high_missing = df_clean.columns[df_clean.isnull().mean() > 0.5]
        if len(high_missing) > 0:
            print(f"\nDropping columns with >50% missing: {list(high_missing)}")
            df_clean = df_clean.drop(columns=high_missing)
        
        # Remove irrelevant columns (IDs, etc.)
        id_cols = ['encounter_id', 'patient_nbr']
        df_clean = df_clean.drop(columns=[col for col in id_cols if col in df_clean.columns])
        
        # Handle '?' as missing values (common in this dataset)
        df_clean = df_clean.replace('?', np.nan)
        
        # Drop rows with missing target
        if 'readmitted_30days' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['readmitted_30days'])
        
        print(f"\nShape after preprocessing: {df_clean.shape}")
        
        return df_clean
    
    def engineer_features(self, df):
        """Create new features from existing data"""
        print("\n" + "=" * 60)
        print("STEP 3: FEATURE ENGINEERING")
        print("=" * 60)
        
        df_eng = df.copy()
        
        # Age grouping (if age is in ranges like '[0-10)')
        if 'age' in df_eng.columns:
            age_map = {
                '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
                '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
                '[80-90)': 85, '[90-100)': 95
            }
            df_eng['age_numeric'] = df_eng['age'].map(age_map)
        
        # Medication count (sum of medication-related columns)
        med_cols = [col for col in df_eng.columns if 'insulin' in col.lower() 
                    or 'diabetesMed' in col]
        if len(med_cols) > 0:
            df_eng['med_count'] = df_eng[med_cols].notna().sum(axis=1)
        
        # Total procedures
        proc_cols = ['num_lab_procedures', 'num_procedures', 'num_medications']
        available_proc = [col for col in proc_cols if col in df_eng.columns]
        if len(available_proc) > 0:
            df_eng['total_procedures'] = df_eng[available_proc].sum(axis=1)
        
        # Prior admission indicator
        if 'number_inpatient' in df_eng.columns:
            df_eng['prior_admissions'] = (df_eng['number_inpatient'] > 0).astype(int)
        
        # Diagnosis severity (simplified grouping)
        diag_cols = [col for col in df_eng.columns if 'diag' in col.lower()]
        
        print(f"\nNew features created: {len([col for col in df_eng.columns if col not in df.columns])}")
        print(f"Total features: {df_eng.shape[1]}")
        
        return df_eng
    
    def prepare_for_modeling(self, df, target_col='readmitted_30days'):
        """Encode categorical variables and split data"""
        print("\n" + "=" * 60)
        print("STEP 4: PREPARING FOR MODELING")
        print("=" * 60)
        
        df_model = df.copy()
        
        # Separate features and target
        if target_col in df_model.columns:
            y = df_model[target_col]
            X = df_model.drop(columns=[target_col, 'readmitted'], errors='ignore')
        else:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        print(f"\nEncoding {len(categorical_cols)} categorical columns...")
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = X[col].astype(str)  # Convert to string first
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print(f"\nTraining set size: {X_train.shape[0]:,}")
        print(f"Test set size: {X_test.shape[0]:,}")
        print(f"Number of features: {X_train.shape[1]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns
    
    def train_models(self, X_train, y_train):
        """Train multiple classification models"""
        print("\n" + "=" * 60)
        print("STEP 5: TRAINING MODELS")
        print("=" * 60)
        
        # Logistic Regression
        print("\n--- Training Logistic Regression ---")
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr
        
        # Random Forest
        print("--- Training Random Forest ---")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, 
                                     class_weight='balanced', max_depth=10)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        
        print("\n✓ Models trained successfully!")
        
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n" + "=" * 60)
        print("STEP 6: MODEL EVALUATION")
        print("=" * 60)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n--- {name} ---")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
        
        return results
    
    def visualize_results(self, results, y_test, feature_names):
        """Create visualizations for model performance"""
        print("\n" + "=" * 60)
        print("STEP 7: VISUALIZATIONS")
        print("=" * 60)
        
        # 1. Model Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['accuracy', 'precision', 'recall', 'roc_auc']
        colors = ['#3498db', '#e74c3c']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = [results[model][metric] for model in results.keys()]
            bars = ax.bar(results.keys(), values, color=colors)
            ax.set_title(f'{metric.upper().replace("_", " ")}', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.set_ylabel('Score')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: model_comparison.png")
        plt.show()
        
        # 2. ROC Curves
        plt.figure(figsize=(10, 8))
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})", linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(alpha=0.3)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: roc_curves.png")
        plt.show()
        
        # 3. Feature Importance (Random Forest)
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_imp, y='feature', x='importance', palette='viridis')
            plt.title('Top 15 Most Important Features (Random Forest)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: feature_importance.png")
            plt.show()
            
            self.feature_importance = feature_imp
    
    def get_insights(self):
        """Generate key insights from the analysis"""
        print("\n" + "=" * 60)
        print("KEY INSIGHTS & RECOMMENDATIONS")
        print("=" * 60)
        
        if self.feature_importance is not None:
            print("\n🔍 TOP RISK FACTORS FOR READMISSION:")
            for idx, row in self.feature_importance.head(5).iterrows():
                print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")
        
        print("\n💡 RECOMMENDATIONS:")
        print("   • Focus interventions on high-risk patients identified by the model")
        print("   • Monitor patients with high values in top risk factors closely")
        print("   • Consider post-discharge follow-up programs for at-risk patients")
        print("   • Use model predictions to allocate care management resources")
        
        print("\n📊 MODEL DEPLOYMENT CONSIDERATIONS:")
        print("   • Random Forest typically offers best performance for this task")
        print("   • Consider recall (sensitivity) as key metric to minimize missed readmissions")
        print("   • Regular model retraining needed as patient populations change")
        print("   • Integrate predictions into Electronic Health Record (EHR) systems")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    print("HOSPITAL READMISSION PREDICTION PROJECT")
    print("=" * 60)
    
    # Initialize predictor
    predictor = ReadmissionPredictor()
    
    # Note: Replace with actual dataset path
    # Download from: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
    
    print("\n📝 TO USE THIS CODE:")
    print("1. Download the UCI Diabetes 130-US Hospitals dataset")
    print("2. Save as 'diabetic_data.csv'")
    print("3. Update the filepath below")
    print("4. Run the complete pipeline")
    
    # UNCOMMENT AND RUN WHEN YOU HAVE THE DATASET:
    """
    # Load data
    df = predictor.load_data('diabetic_data.csv')
    
    # Preprocess
    df_clean = predictor.preprocess_data(df)
    
    # Engineer features
    df_featured = predictor.engineer_features(df_clean)
    
    # Prepare for modeling
    X_train, X_test, y_train, y_test, feature_names = predictor.prepare_for_modeling(df_featured)
    
    # Train models
    predictor.train_models(X_train, y_train)
    
    # Evaluate
    results = predictor.evaluate_models(X_test, y_test)
    
    # Visualize
    predictor.visualize_results(results, y_test, feature_names)
    
    # Get insights
    predictor.get_insights()
    """
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE - Ready to run when dataset is available!")
    print("=" * 60)