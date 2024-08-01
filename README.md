# Breast Cancer Prediction Using Machine Learning

## Project Overview

This project focuses on predicting breast cancer malignancy using various machine learning models. By leveraging data from the UCI Machine Learning Repository, the project aims to provide insights into effective diagnostic tools that can aid early detection of breast cancer.

## Models Implemented

The project utilizes several machine learning algorithms, including:

- Logistic Regression
- Decision Tree Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Gradient Boosting Classifier

## Key Features

- **Data Preprocessing**: Cleaning and preparing the dataset for analysis.
- **Exploratory Data Analysis (EDA)**: Visualizing data distribution and feature relationships.
- **Model Training and Evaluation**: Implementing and evaluating various models to determine the most accurate approach.
- **Hyperparameter Tuning**: Optimizing model performance using GridSearchCV.

## Results

The models were evaluated based on accuracy and ROC AUC scores, with the Support Vector Machine (SVM) demonstrating the highest performance. Visualization plots for ROC curves and performance evaluation are included in the analysis.

## Files Included

- `breast_cancer.csv`: Dataset used for training and testing the models.
- `main.py`: Python script containing the data preprocessing, model implementation, and analysis code.
- Images generated from the analysis:
  - `missing_data.png`: Visualizing missing data in the dataset.
  - `density_graphs.png`: Density plots for each feature.
  - `correlation_heatmap.png`: Correlation heatmap of the features.
  - `roc_breast_cancer.jpeg`: ROC curves for each model.
  - `PE_breast_cancer.jpeg`: Performance evaluation plot for the models.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Breast-Cancer-ML.git
