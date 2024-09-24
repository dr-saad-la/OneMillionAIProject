"""
    This module contains a set of function to perform salary predictions project.
"""

import os
import sys
from pathlib import Path
import time
import shutil
import requests

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score


def banner(ban_char, nban_char, title=None):
    if not isinstance(ban_char, str) or len(ban_char) != 1:
        raise ValueError("ban_char must be a single character string.")

    if not isinstance(nban_char, int) or nban_char <= 0:
        raise ValueError("nban_char must be a positive integer.")

    if title is not None:
        print(ban_char * nban_char)
        print(title.center(nban_char))
        print(ban_char * nban_char)

    if title is None:
        print(ban_char * nban_char)

def download_data(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Data downloaded successfully and saved at: {save_path}")
            return True
        else:
            print(f"Failed to download data. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error during download: {str(e)}")
        raise e


def read_data_from_url(url):
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        raise ValueError(f"Error reading data from URL: {str(e)}")


def describe_dataset(data, preview=True):
    if isinstance(data, str):
        try:
            df = pd.read_csv(data)
        except FileNotFoundError as fnf_error:
            raise FileNotFoundError(f"File not found: {data}") from fnf_error
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Unsupported data type. Expected either a file path (str) or pandas DataFrame.")
    n_rows, n_features = df.shape

    description = {
        'n_rows': n_rows,
        'n_features': n_features,
        'column_names': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict()
    }

    if preview and n_rows > 0:
        description['preview_rows'] = df.head()

    return description

def print_descriptive_statistics(df):
    descriptive_stats = df.describe(include='all')
    print(descriptive_stats)


def plot_fit_reg_line(data, target_name, ind_name, show_correlation=False):
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Unsupported data type. Expected either a file path (str) or pandas DataFrame.")

    if target_name not in df.columns or ind_name not in df.columns:
        raise ValueError(f"Columns {target_name} and/or {ind_name} not found in the dataset.")

    plt.figure(figsize=(10, 6))
    sns.regplot(x=df[ind_name], y=df[target_name], ci=None, scatter_kws={'alpha':0.5})

    if show_correlation:
        correlation_coef, _ = pearsonr(df[ind_name], df[target_name])
        plt.title(f'{target_name} vs {ind_name}\nCorrelation Coefficient: {correlation_coef:.2f}')
    else:
        plt.title(f'{target_name} vs {ind_name}')

    plt.xlabel(ind_name)
    plt.ylabel(target_name)
    plt.grid(True)
    plt.show()
    

def split_data(data, target_name, test_size=0.2, random_state=42):
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Unsupported data type. Expected either a file path (str) or pandas DataFrame.")

    if target_name not in df.columns:
        raise ValueError(f"Column {target_name} not found in the dataset.")
    
    X = df.drop(columns=[target_name])
    y = df[target_name]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "predictions": y_pred,
        'model': model,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def concatenate_results(X_test, y_test, predicted_label, predictions):
    if not isinstance(y_test, pd.DataFrame):
        y_test = y_test.to_frame()

    result = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    result[predicted_label] = predictions

    return result

def plot_fit_reg_line_with_predictions(df, target_name, ind_name, predictions, show_correlation=False):
    if target_name not in df.columns or ind_name not in df.columns:
        raise ValueError(f"Columns {target_name} and/or {ind_name} not found in the dataset.")

    plt.figure(figsize=(10, 6))
    sns.regplot(x=df[ind_name], y=df[target_name], ci=None, scatter_kws={'alpha':0.5}, label='Actual')

    # Overlay the predicted values
    plt.scatter(df[ind_name], df[predictions], color='red', alpha=0.5, label='Predicted')

    if show_correlation:
        correlation_coef, _ = pearsonr(df[ind_name], df[target_name])
        plt.title(f'{predictions} vs {ind_name}\nCorrelation Coefficient: {correlation_coef:.2f}')
    else:
        plt.title(f'{predictions} vs {ind_name}')

    plt.xlabel(ind_name)
    plt.ylabel(target_name)
    plt.legend()
    plt.grid(True)
    plt.show()