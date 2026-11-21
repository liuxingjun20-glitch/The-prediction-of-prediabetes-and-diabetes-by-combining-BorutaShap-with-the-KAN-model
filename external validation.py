#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

# ============================================================
# Part 1. Define model directory
# ============================================================

save_dir = (
    r"C:\Users\65163\Desktop\python"
    r"\NHANES_Diabetes\BorutaShap_Saved_Models"
)

# ============================================================
# Part 2. List all saved files
# ============================================================

print("=" * 60)
print("SAVED MODEL FILES")
print("=" * 60)

for file_name in sorted(os.listdir(save_dir)):
    file_path = os.path.join(save_dir, file_name)

    if os.path.isfile(file_path):
        print(
            f"{file_name:<45}"
            f"{os.path.getsize(file_path) / 1024:.2f} KB"
        )

# ============================================================
# Part 3. Check directory
# ============================================================

print("\nDirectory exists:", os.path.exists(save_dir))


# In[2]:


import os
import joblib
import cloudpickle
import tensorflow as tf

# ============================================================
# Part 1. Define directory
# ============================================================

save_dir = (
    r"C:\Users\65163\Desktop\python"
    r"\NHANES_Diabetes\BorutaShap_Saved_Models"
)

# ============================================================
# Part 2. Load MLP, CNN and GRU models
# ============================================================

models_deep_boruta = {}

for model_name in ["MLP", "CNN", "GRU"]:

    model_path = os.path.join(
        save_dir,
        f"{model_name}_BorutaShap.keras"
    )

    models_deep_boruta[model_name] = (
        tf.keras.models.load_model(
            model_path,
            compile=False
        )
    )

    print(f"{model_name} loaded successfully.")


# ============================================================
# Part 3. Load KAN model
# ============================================================

kan_path = os.path.join(
    save_dir,
    "KAN_BorutaShap_cloudpickle.pkl"
)

with open(kan_path, "rb") as f:
    models_deep_boruta["KAN"] = cloudpickle.load(f)

models_deep_boruta["KAN"].eval()

print("KAN loaded successfully.")


# ============================================================
# Part 4. Load preprocessors
# ============================================================

preprocessors_deep_boruta = {}

for model_name in ["MLP", "CNN", "GRU", "KAN"]:

    preprocessor_path = os.path.join(
        save_dir,
        f"{model_name}_preprocessor.joblib"
    )

    preprocessors_deep_boruta[model_name] = (
        joblib.load(preprocessor_path)
    )

    print(f"{model_name} preprocessor loaded successfully.")


# ============================================================
# Part 5. Load class labels
# ============================================================

classes_deep_boruta = {}

for model_name in ["MLP", "CNN", "GRU", "KAN"]:

    classes_path = os.path.join(
        save_dir,
        f"{model_name}_classes.joblib"
    )

    classes_deep_boruta[model_name] = (
        joblib.load(classes_path)
    )

    print(
        f"{model_name} classes:",
        classes_deep_boruta[model_name]
    )


# ============================================================
# Part 6. Load BorutaShap feature information
# ============================================================

feature_info_deep_boruta = joblib.load(
    os.path.join(
        save_dir,
        "BorutaShap_feature_info.joblib"
    )
)

print("\nFeature information:")
print(feature_info_deep_boruta)


# ============================================================
# Part 7. Check model input dimensions
# ============================================================

print("\n" + "=" * 60)
print("MODEL LOADING VALIDATION")
print("=" * 60)

for model_name in ["MLP", "CNN", "GRU"]:

    print(
        f"{model_name} input shape:",
        models_deep_boruta[model_name].input_shape
    )

print(
    "KAN model width:",
    models_deep_boruta["KAN"].width
)

print(
    "KAN model grid:",
    models_deep_boruta["KAN"].grid
)

print("\nAll models and preprocessors loaded successfully.")


# In[3]:


import os
import numpy as np
import pandas as pd

# ============================================================
# 1. File paths
# ============================================================

base_dir = (
    r"C:\Users\65163\Desktop\python"
    r"\NHANES_Diabetes\2021-2023"
)

input_file = os.path.join(
    base_dir,
    "NHANES_2021_2023_BorutaShap_External_Validation.xlsx"
)

output_file = os.path.join(
    base_dir,
    "NHANES_2021_2023_BorutaShap_External_Validation_Results.xlsx"
)

# ============================================================
# 2. Load external validation data
# ============================================================

df_external = pd.read_excel(input_file)

# Use the exact original training feature order
feature_columns = feature_info_deep_boruta["original_features"]

target_column = "Diabetes"

X_external = df_external[feature_columns].copy()
y_external = df_external[target_column].astype(int).to_numpy()

# ============================================================
# 3. Basic checks
# ============================================================

assert X_external.shape[1] == 8
assert len(X_external) == len(y_external)
assert not X_external.isnull().any().any()
assert set(np.unique(y_external)).issubset({0, 1})

print("=" * 65)
print("NHANES 2021-2023 EXTERNAL VALIDATION")
print("=" * 65)

print("Sample size:", len(X_external))
print("Number of features:", X_external.shape[1])
print("Feature order:", feature_columns)

print("\nTarget distribution:")
print(pd.Series(y_external).value_counts().sort_index())

print("\nTarget percentage:")
print(
    pd.Series(y_external)
    .value_counts(normalize=True)
    .sort_index() * 100
)

print("\nData check completed.")


# In[4]:


import torch
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. Define KAN prediction function
# ============================================================

def kan_predict_proba(model_deep, X_deep):

    model_deep.eval()

    with torch.no_grad():

        X_tensor_deep = torch.tensor(
            X_deep,
            dtype=torch.float32
        )

        logits_deep = model_deep(X_tensor_deep)

        probabilities_deep = torch.sigmoid(logits_deep)

    return probabilities_deep.cpu().numpy().reshape(-1)


# ============================================================
# 2. Initialize results
# ============================================================

external_results = []

external_predictions = {}


# ============================================================
# 3. External validation for each model
# ============================================================

for model_name in ["MLP", "CNN", "GRU", "KAN"]:

    print("\n" + "=" * 65)
    print(f"Evaluating {model_name}")
    print("=" * 65)


    # --------------------------------------------------------
    # A. Load corresponding preprocessor
    # --------------------------------------------------------

    preprocessor = preprocessors_deep_boruta[model_name]

    # Transform only, do not fit again

    X_transformed = preprocessor.transform(X_external)

    X_transformed = np.asarray(
        X_transformed,
        dtype=np.float32
    )

    print("Transformed feature shape:", X_transformed.shape)


    # --------------------------------------------------------
    # B. Prepare model-specific input
    # --------------------------------------------------------

    if model_name == "MLP":

        X_model = X_transformed

        model = models_deep_boruta["MLP"]

        y_prob = model.predict(
            X_model,
            verbose=0
        ).reshape(-1)


    elif model_name == "CNN":

        X_model = X_transformed.reshape(
            X_transformed.shape[0],
            X_transformed.shape[1],
            1
        )

        model = models_deep_boruta["CNN"]

        y_prob = model.predict(
            X_model,
            verbose=0
        ).reshape(-1)


    elif model_name == "GRU":

        X_model = X_transformed.reshape(
            X_transformed.shape[0],
            X_transformed.shape[1],
            1
        )

        model = models_deep_boruta["GRU"]

        y_prob = model.predict(
            X_model,
            verbose=0
        ).reshape(-1)


    elif model_name == "KAN":

        X_model = X_transformed

        model = models_deep_boruta["KAN"]

        y_prob = kan_predict_proba(
            model,
            X_model
        )


    # --------------------------------------------------------
    # C. Check predicted probabilities
    # --------------------------------------------------------

    if len(y_prob) != len(y_external):

        raise ValueError(
            f"{model_name}: prediction length does not match "
            "the external validation sample size."
        )


    if not np.all(np.isfinite(y_prob)):

        raise ValueError(
            f"{model_name}: predicted probabilities contain "
            "NaN or infinite values."
        )


    if np.any((y_prob < 0) | (y_prob > 1)):

        raise ValueError(
            f"{model_name}: predictions are outside [0, 1]."
        )


    # --------------------------------------------------------
    # D. Bootstrap evaluation (1000 resamples)
    # --------------------------------------------------------

    y_true_external = np.asarray(y_external).ravel()

    y_pred_external = (y_prob >= 0.5).astype(int)

    n_samples_external = len(y_true_external)

    rng_external = np.random.default_rng(50)

    metrics_external = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "AUC": []
    }


    # Original estimates

    original_metrics_external = {

        "Accuracy": accuracy_score(
            y_true_external,
            y_pred_external
        ),

        "Precision": precision_score(
            y_true_external,
            y_pred_external,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true_external,
            y_pred_external,
            zero_division=0
        ),

        "F1": f1_score(
            y_true_external,
            y_pred_external,
            zero_division=0
        ),

        "AUC": roc_auc_score(
            y_true_external,
            y_prob
        )
    }


    # 1000 Bootstrap resamples

    for i in range(1000):

        indices_external = rng_external.choice(
            n_samples_external,
            size=n_samples_external,
            replace=True
        )

        y_true_boot_external = y_true_external[
            indices_external
        ]

        y_prob_boot_external = y_prob[
            indices_external
        ]

        y_pred_boot_external = (
            y_prob_boot_external >= 0.5
        ).astype(int)


        # Skip AUC calculation if only one class is present

        if len(np.unique(y_true_boot_external)) < 2:

            continue


        metrics_external["Accuracy"].append(
            accuracy_score(
                y_true_boot_external,
                y_pred_boot_external
            )
        )

        metrics_external["Precision"].append(
            precision_score(
                y_true_boot_external,
                y_pred_boot_external,
                zero_division=0
            )
        )

        metrics_external["Recall"].append(
            recall_score(
                y_true_boot_external,
                y_pred_boot_external,
                zero_division=0
            )
        )

        metrics_external["F1"].append(
            f1_score(
                y_true_boot_external,
                y_pred_boot_external,
                zero_division=0
            )
        )

        metrics_external["AUC"].append(
            roc_auc_score(
                y_true_boot_external,
                y_prob_boot_external
            )
        )


    # Calculate 95% confidence intervals

    bootstrap_results_external = []

    for metric_external in metrics_external:

        bootstrap_values_external = np.asarray(
            metrics_external[metric_external]
        )

        lower_95CI_external = np.percentile(
            bootstrap_values_external,
            2.5
        )

        upper_95CI_external = np.percentile(
            bootstrap_values_external,
            97.5
        )

        bootstrap_results_external.append({

            "Metric": metric_external,

            "Estimate": original_metrics_external[
                metric_external
            ],

            "Lower_95CI": lower_95CI_external,

            "Upper_95CI": upper_95CI_external,

            "Valid_Bootstrap": len(
                bootstrap_values_external
            )

        })


    bootstrap_results_external = pd.DataFrame(
        bootstrap_results_external
    )


    # --------------------------------------------------------
    # E. Store Bootstrap results
    # --------------------------------------------------------

    result_external = {

        "Model": model_name

    }


    for _, row_external in bootstrap_results_external.iterrows():

        result_external[row_external["Metric"]] = (

            f"{row_external['Estimate']:.3f} "
            f"({row_external['Lower_95CI']:.3f}–"
            f"{row_external['Upper_95CI']:.3f})"

        )


    result_external["Valid_Bootstrap"] = (

        bootstrap_results_external[
            "Valid_Bootstrap"
        ].iloc[0]

    )


    external_results.append(
        result_external
    )


    # --------------------------------------------------------
    # F. Store predictions
    # --------------------------------------------------------

    y_pred = (y_prob >= 0.5).astype(int)

    external_predictions[model_name] = {

        "Predicted_Label": y_pred,

        "Predicted_Probability": y_prob

    }


    print("\nBootstrap Evaluation Results:")

    print(bootstrap_results_external.round(4).to_string(index=False))


# ============================================================
# 4. Final external validation results
# ============================================================

external_results_df = pd.DataFrame(
    external_results
)


external_results_df = external_results_df[
    [
        "Model",
        "Accuracy",
        "Recall",
        "Precision",
        "F1",
        "AUC",
        "Valid_Bootstrap"
    ]
]


external_results_df = external_results_df.rename(
    columns={
        "Valid_Bootstrap": "Bootstrap"
    }
)


print(
    "\nEXTERNAL VALIDATION RESULTS (1000 Bootstrap, 95% CI)"
)

display(external_results_df)


# In[ ]:




