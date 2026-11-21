# BorutaShap-KAN-Framework-for-Prediabetes-and-Diabetes-Risk-Prediction

## Project Overview

This project implements a **BorutaShap feature selection and KAN-based prediction framework** for prediabetes and diabetes risk prediction using NHANES data.

The framework combines feature selection and deep learning models to investigate predictive performance on clinical tabular data.

## Dataset

This project uses data from the National Health and Nutrition Examination Survey (NHANES):

- **NHANES 2017–March 2020:** Used for model development and internal evaluation.
- **NHANES 2021–2023:** Used as an independent external validation dataset.

The datasets are used for data preprocessing, feature selection, model training, internal testing, and external validation.

## Code Information

The main components of the code include:

- Descriptive statistical analysis of the dataset
- Data preprocessing, including data standardization and categorical variable encoding
- Feature selection using the BorutaShap algorithm
- Training and evaluation of deep learning models, including CNN, GRU, MLP, and KAN
- Comparative experiments using conventional machine learning models, including Logistic Regression and Random Forest
- Model evaluation using Accuracy, Precision, Recall, F1-score, and AUC
- Comparative experiments using all original features as model inputs
- Ablation experiments comparing BorutaShap and LASSO feature selection methods
- Internal testing and independent external validation
- Bootstrap-based estimation of 95% confidence intervals

## Usage Instructions

To reproduce the experiments reported in this study, please follow these steps:

1. Clone or download this repository to your local machine.
2. Download the required NHANES datasets and place the dataset files in the appropriate project directory.
3. Ensure that all required Python libraries are installed (see the Requirements section).
4. Open the provided Jupyter Notebook files in the project directory.
5. Run the notebook cells sequentially to execute the experimental pipeline.

## Requirements

The experiments require the following Python libraries:

| Library | Version |
|---|---|
| numpy | 2.3.5 |
| pandas | 3.0.0 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.0 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| tensorflow | 2.20.0 |
| torch | 2.10.0+cpu |
| xgboost | 3.1.3 |
| BorutaShap | borutashap-modern 1.1.0 |
| KAN | pykan 0.2.8 |

## Software Environment

- **Operating System:** Windows 11, version 10.0.26200, 64-bit (AMD64)
- **Processor Architecture:** x86_64 / AMD64
- **Python Version:** 3.13
- **Computing Device:** CPU (no GPU used)
- **Development Environment:** Jupyter Notebook
