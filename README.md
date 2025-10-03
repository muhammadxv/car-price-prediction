#  Used Car Price Prediction
 Project Overview

This project predicts the price of used cars based on features such as engine size, horsepower, weight, and brand.
It demonstrates a full machine learning pipeline: data cleaning, feature engineering, model training, evaluation, and visualization.

I built this notebook as part of my data science portfolio to showcase skills in EDA, regression modeling, and model deployment readiness.


#  Tools & Libraries

Python (pandas, numpy, matplotlib, seaborn)

scikit-learn (Linear Regression, Random Forest, preprocessing, metrics)

Joblib (model persistence)

Jupyter/Kaggle Notebook


#  Methodology

Data Loading & Cleaning

- Imported dataset from Kaggle (CarPrice_Assignment.csv)

- Extracted car brand from name & fixed typos

- Handled categorical variables with one-hot encoding

Exploratory Data Analysis (EDA)

- Visualized correlations between features and car prices

- Checked for missing values & outliers

Modeling

- Linear Regression (baseline)

- Random Forest Regressor (improved accuracy)

Evaluation

- Metrics: R² score, RMSE

- Random Forest significantly outperformed Linear Regression


Results & Insights

- Horsepower, engine size, and curb weight were the most influential features

- Actual vs. Predicted plot showed good model performance


#  Results

Linear Regression:

R² = -1.0419497587222164e+24

RMSE = 9069493012738966.0

Random Forest:

R² = 0.9584

RMSE = 1811.84

| Actual vs Predicted (RF)                         | Feature Importances                                     |
| ------------------------------------------------ | ------------------------------------------------------- |
| ![Pred vs Actual](results/pred_vs_actual_rf.png) | ![Feature Importances](results/feature_importances.png) |

# 📂 Repository Structure
car-price-prediction/
│
├── used_car_price_prediction.ipynb   # Main notebook
├── requirements.txt                  # Libraries used
├── README.md                         # Project description
├── results/
│   ├── pred_vs_actual_rf.png
│   └── feature_importances.png
├── rf_model.joblib                   # Saved Random Forest model
├── lr_model.joblib                   # Saved Linear Regression model
└── scaler.joblib                     # Scaler for preprocessing

# Next Steps

Extend with XGBoost/LightGBM for better accuracy

Build a Streamlit web app for interactive predictions

Experiment with hyperparameter tuning (GridSearchCV / RandomizedSearchCV)

# Dataset
Source: [Kaggle – Car Price Prediction](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)


 *With this project, I demonstrated the ability to:*

Handle raw datasets, clean & preprocess data

Apply multiple ML models & compare performance

Save trained models for later use in apps

Communicate insights visually & clearly
