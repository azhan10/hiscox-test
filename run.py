
import pandas as pd
import numpy as np

import sys, pathlib, xgboost

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(pathlib.Path().resolve()) + "/scripts/")

import pre_process as preprocess_module
import analytics as analytics_module
import model as model_module


if __name__ == '__main__':
    
    
    # DATA PREPARATION

    pre_process_class = preprocess_module.PreProcessData(query="SELECT * FROM CLAIMS.DS_DATASET")
    
    print("\n\n")
    
    data_original_df = pre_process_class.get_data()
    
    print("Top 10 rows of the data")
    print(data_original_df.head(10))
    
    print("\n\n")
    
    print("The size of rows per unique value for claim_status column")
    print(pre_process_class.count_rows("claim_status"))
    
    print("\n\n")
    
    print("Saved the original data in the data directory")
    pre_process_class.save_data_to_spreadsheet(filename="dataset_from_database")
    
    print("\n\n")
    
    print("Missing data statistics")
    
    # CHECK THE TYPES, IT'S NOT MATCHING
    missing_data_statistics_df = pre_process_class.missing_statistics_data()
    print(missing_data_statistics_df)
    
    print("\n\n")
    
    drop_none_exclude_columns = ["family_history_3", "employment_type"]
    data_final_df = pre_process_class.drop_none_columns(columns=drop_none_exclude_columns)
    
    print("The data that excludes none rows within columns")
    for column in data_final_df.columns:
        print(f"Column: {column}, dtype: {data_final_df[column].dtype}")
    
    print("\n\n")
    
    print("The finalised data")
    print(data_final_df.head(10))
    
    
    # ANALYTICS
    
    analytics_class = analytics_module.Analytics(data=data_final_df)
    
    print("\n\n")
    
    pairwise_column = "claim_status"
    print(f"Produced the pairwise relationship with {pairwise_column}")
    analytics_class.display_pairwise_relationships(column=pairwise_column)
    
    
    print("\n\n")
    
    print("Running the correlation")
    
    correlation_exclude_columns = [
        "product_var_3", 
        "marital_status", 
        "occupation", 
        "location"
    ]
    correlation_df = analytics_class.get_correlation(exclude_columns=correlation_exclude_columns)
    print(correlation_df)
    
    
    print("\n\n")
    
    print("Running the heatmap")
    
    analytics_class.display_heatmap(correlation=correlation_df)
    
    boxplots_and_histogram_columns = [
        "age",
        "height_cm",
        "weight_kg",
        "income",
        "financial_hist_1",
        "financial_hist_2",
        "financial_hist_3",
        "financial_hist_4",
        "credit_score_1",
        "credit_score_2",
        "credit_score_3",
        "insurance_hist_1",
        "insurance_hist_2",
        "insurance_hist_3",
        "insurance_hist_4",
    ]
    analytics_class.display_boxplots_and_histogram(columns=boxplots_and_histogram_columns)
    
    
    # MODEL TRAINING
    
    print("\n\n")
    
    print("Running the model evaluation of initial model")
    
    model_class = model_module.Model(data=data_final_df, target="claim_status")
    
    evaluation_metrics = ["auc", "rmse", "logloss"]
    
    initial_model = xgboost.XGBClassifier(objective="binary:logistic", eval_metric=evaluation_metrics, enable_categorical=True)
    
    model_class.set_model(initial_model)
    
    model_class.train_model(initial_model)
    
    model_class.model_evaluation(name="initial model")
    
    print("\n\n")
    
    print("Running the model evaluation of cross validation model")
    
    cv_model = xgboost.XGBClassifier(
        objective="binary:logistic",
        eval_metric=evaluation_metrics,
        early_stopping_rounds=15,
        enable_categorical=True,
    )
    
    cv_best_parameters = model_class.cross_validation(cv_model)
    
    model_class.set_model(cv_model)
    
    model_class.train_model(cv_model)
    
    model_class.model_evaluation(name="cv model")
    
    
    print("\n\n")
    
    print("Running the model evaluation of best parameter model")
    
    best_parameter_model = xgboost.XGBClassifier(
        objective="binary:logistic",
        eval_metric=evaluation_metrics,
        early_stopping_rounds=15,
        enable_categorical=True,
        **cv_best_parameters.best_params_
    )
    
    model_class.set_model(best_parameter_model)
    
    model_class.train_model(best_parameter_model)
    
    model_class.model_evaluation(name="best parameter model")
    
    model_class.save_model(name="xgboost_model_optimised_with_cross_validation")
    
    
    print("\n\n")
    
    print("Running the summary")
    
    model_class.display_summary(name="best parameter summary")
    
    
    
    
    
    
    
    
    
    
    

