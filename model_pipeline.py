import pandas as pd
import numpy as np
import json
import time
from itertools import combinations

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV

#models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#metrics
from sklearn.metrics import f1_score, recall_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix


start = time.time()
# Load dataset
file_path = 'heart.csv'  # Update the path if needed
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
features_to_select = np.array(df.columns.drop(labels=["output", "chol", "fbs", "restecg"]))

print(features_to_select)
print(len(features_to_select))

# Prepare features and target
X = df.drop(columns=["output", "chol", "fbs", "restecg"])
y = df["output"]


###Feature selection with Random Forest
# Feature selection using RandomForestClassifier
# feature_selector = RandomForestClassifier(random_state=42, n_estimators=100)
# feature_selector.fit(X, y)
# selector = SelectFromModel(feature_selector, prefit=True)
# X_selected = selector.transform(X)

# selected_features_mask = selector.get_support()

# Get the names of the selected features
# selected_features = X.columns[selected_features_mask]

# print("Selected Features:")
# print(selected_features)


# Models and hyperparameter grids
models_params_gridsearch_updated = {
    "KNeighborsClassifier": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7]
        }
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 15]
        }
    },
    "SVC": {
        "model": SVC(probability=True),
        "params": {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "kernel": ["linear", "rbf"]
        }
    },
    "LogisticRegression": {
        "model": LogisticRegression(),
        "params": {
            "C": [0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs', 'newton-cg']
    }
    },
    "XGBoost": {
    "model": XGBClassifier(eval_metric='logloss', random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7]
    }
    },
    "LightGBM": {
    "model": XGBClassifier(random_state=42),
    "params": {
        "n_estimators": [50, 100, 200],
        "max_depth": [-1, 5, 10]
    }
}
}

# Define the pipeline function
def evaluate_models_pipeline(X, y, models_params_gridsearch, smote_strategy=0.91, test_size=0.2):
    
    smote = SMOTE(random_state=32, sampling_strategy=smote_strategy)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_size, random_state=32, stratify=y_resampled
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_model = None
    best_params = None
    best_score = 0
    best_accuracy = 0
    best_conf_mx = None
    best_class_report = None
    best_roc_auc = 0

    for model_name, mp in models_params_gridsearch.items():
        clf = GridSearchCV(
            mp['model'],
            mp['params'],
            cv=stratified_kfold,
            scoring='roc_auc',
            return_train_score=False,
            n_jobs=-1
        )
        
        clf.fit(X_train_scaled, y_train)
        model = clf.best_estimator_
        model.fit(X_train_scaled, y_train)

        # Step 6: Evaluation
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        conf_mx = confusion_matrix(y_test, y_pred, labels=[0, 1])
        class_report = classification_report(y_test, y_pred, target_names=['lower chance of heart disease', 'higher chance of heart disease'], digits=5)

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        avg_score = (roc_auc + recall) / 2

        if avg_score > best_score:
            best_score = avg_score
            best_model = model_name
            best_params = clf.best_params_
            best_accuracy = accuracy
            best_conf_mx = conf_mx.tolist()
            best_class_report = class_report
            best_roc_auc = roc_auc

    best_result = {
        "smote": smote_strategy,
        "test_size": test_size,
        "model_name": best_model,
        "best_params": best_params,
        "accuracy": best_accuracy,
        "avg_score": best_score,
        "confusion_matrix": best_conf_mx,
        "roc_auc": best_roc_auc,
        "classification_report": best_class_report
    }

    return best_result

log_file_path = "model_evaluation_progress.log"

X_set = []
feature_columns = X.columns.tolist()  # List of column names
for num_features in range(1, len(feature_columns) + 1):  # From 1 feature to all features
    for subset_columns in combinations(feature_columns, num_features):
        X_set.append(X[list(subset_columns)])  # Add the subset as a DataFrame


smote_range = np.arange(0.90, 0.911, 0.005)
test_size_range = [0.2]

best_result = None
best_accuracy = 0
all_results_updated = []
cnt_subset = 0
with open(log_file_path, "a") as log_file:
    for smote_strategy in smote_range:
        for test_size in test_size_range:
            for subset in X_set:

                cnt_subset += 1
                log_message = (
                    f"---------\n"
                    f"Subset: {cnt_subset}/{len(X_set)*len(smote_range)*len(test_size_range)}\n"
                    f"SMOTE Strategy: {smote_strategy}, Test Size: {test_size}\n"
                    f"Feature Subset: {subset.columns.tolist()}\n"
                )

                t2 = time.time()

                elapsed_time_message = (
                    f"Elapsed Time: {int((t2 - start) // 60)}:{np.round((t2 - start) % 60, 2):02f}\n"
                )

                print(log_message)
                print(elapsed_time_message)
                log_file.write(log_message)
                log_file.write(elapsed_time_message)

                current_result = evaluate_models_pipeline(
                    subset, y, models_params_gridsearch_updated, smote_strategy, np.round(test_size, 2)
                )
                if current_result["accuracy"] > best_accuracy:
                    best_accuracy = current_result["accuracy"]
                    best_result = current_result
                all_results_updated.append(current_result)
                
                current_result_message = (
                    f"{current_result['model_name']}\n"
                    f"{current_result['best_params']}\n"
                    f"{current_result['accuracy']}\n"
                    f"{"\n".join(map(str,current_result['confusion_matrix']))}\n"
                    f"------END--------\n"
                )
                print(current_result_message)
                log_file.write(current_result_message)
                
                log_file.flush()

# Save
output_file_updated = "updated_model_results_no_lgbm.json"
with open(output_file_updated, "w") as f:
    json.dump(all_results_updated, f, indent=4)

print("Best Model Evaluation:")
print("---------------------------------------------------------------------------------------------------------------------")
print(f"SMOTE Strategy: {best_result['smote']}")
print(f"Test Size: {best_result['test_size']}")
print(f"Model Name: {best_result['model_name']}")
print(f"Best Parameters: {best_result['best_params']}")
print(f"Accuracy: {best_result['accuracy']:.5f}")
print(f"Average Score (ROC AUC + Recall): {best_result['avg_score']:.5f}")
print(f"ROC AUC: {best_result['roc_auc']:.5f}")
print("Confusion Matrix:")
conf_matrix = best_result['confusion_matrix']
print(f"    True Positive: {conf_matrix[1][1]}")
print(f"    False Positive: {conf_matrix[0][1]}")
print(f"    True Negative: {conf_matrix[0][0]}")
print(f"    False Negative: {conf_matrix[1][0]}")
print("\nClassification Report:")
print(best_result['classification_report'])

end = time.time()
print(f"Elapsed time: {int((end-start) // 60):02d} : {int((end-start)) % 60:02d}")