# Python file: utils/preprocess.py
#
# Content:
#
# Constants:
#    - RANDOM_STATE: Used to ensure reproducibility.
#    - NAN_THRESHOLD: Threshold to determine the maximum number of NaN values allowed in rows.
#    - CBC_ITERATIONS: Number of iterations for CatBoostClassifier.
#    - CBC_VERBOSE: Verbosity level for CatBoostClassifier.
#    - CBC_CLASS_WEIGHTS: Class weights for handling imbalanced data in CatBoost.
#
# Functions:
#    - process_na(df: pd.DataFrame): Cleans a DataFrame by removing rows with excessive NaN values and deceased clients.
#    - process_df(df: pd.DataFrame): Processes a DataFrame by adding new calculated features, cleaning data, and handling missing values.
#    - frequency_encoding(X: pd.DataFrame): Encodes categorical features in a DataFrame based on their frequency.
#
# Classes:
#    - DataFrameProcessor: A custom transformer class (inherits from BaseEstimator, TransformerMixin) that processes a DataFrame using the process_df function.
#
# ColumnTransformer:
#    - column_transformer: Applies numerical scaling, one-hot encoding, and frequency encoding to different columns.
#
# Pipeline:
#    - model: A machine learning pipeline that preprocesses the data (using DataFrameProcessor and column_transformer) and fits a multi-output CatBoostClassifier model.
#
# This file defines a data processing pipeline for preparing datasets for multi-output classification models using CatBoost. 
# It includes custom transformers for feature engineering, scaling, and encoding.

# Import standard libraries
import pandas as pd
import numpy as np

# Import scikit-learn's base classes and transformers
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer  # For applying transformations to different columns
from sklearn.multioutput import MultiOutputClassifier  # For multi-output classification tasks
from sklearn.pipeline import Pipeline  # To create machine learning pipelines

# Import typing for type hinting, which improves code readability and helps with IDE auto-completion
from typing import Optional, Dict, Any

# Import CatBoostClassifier, a gradient boosting library for classification
from catboost import CatBoostClassifier

# Import configuration variables and settings
from utils.config import (
    numerical_columns,  # List of numerical columns for feature engineering
    freq_encode_columns,  # Columns that will be frequency-encoded
    one_hot_columns,  # Columns that will be one-hot encoded
    RANDOM_STATE,  # Random seed for reproducibility
    NAN_THRESHOLD,  # Threshold for handling missing values
    CBC_ITERATIONS,  # Number of iterations for the CatBoost classifier
    CBC_VERBOSE,  # Verbosity level for CatBoost training output
    CBC_CLASS_WEIGHTS  # Class weights for handling imbalanced datasets in CatBoost
)


def process_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the dataframe by removing rows with too many NaN values and those marked as deceased.

    Args:
        df (pd.DataFrame): The input dataframe to process.

    Returns:
        pd.DataFrame: The cleaned dataframe with fewer NaN values and no deceased individuals.
    """
    # Keep rows where fewer than NAN_THRESHOLD NaN values exist and 'ind_deceased' is 'N'
    df = df[df.isna().sum(axis=1) < NAN_THRESHOLD]
    df = df[df['ind_deceased'] == 'N']
    
    return df


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DataFrame by adding calculated features and cleaning up data.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: Processed DataFrame with new columns and cleaned data.
    """
    # Add new features related to the number of products and fetch date
    df['number_of_products'] = df[[col for col in df.columns if col.startswith('ind_1m_')]].sum(axis=1)
    df['fetch_year'] = df['fetch_date'].dt.year
    df['fetch_month'] = df['fetch_date'].dt.month

    # Calculate tenure in months
    df['tenure_months'] = (
        (df['fetch_date'].dt.year - df['registration_date'].dt.year) * 12
        + (df['fetch_date'].dt.month - df['registration_date'].dt.month)
    )

    # Log-transform income to reduce skewness
    df['income'] = np.log(df['income'])

    # Clean up and encode client type and client segment
    df['client_type_1m'] = pd.to_numeric(df['client_type_1m'], errors='coerce').replace('P', 5)
    df['client_segment'] = df['client_segment'].map({
        None: 0,
        '02 - PARTICULARES': 2,
        '03 - UNIVERSITARIO': 3,
        '01 - TOP': 1
    })

    # Fill missing values in relevant columns
    df.fillna({
        'province_code': 0,
        'gender': 'V',
        'client_activity_1m': 'N',
        'entry_channel': 'UNK',
        'income': df['income'].median(),
        'client_type_1m': 0
    }, inplace=True)

    # Convert columns to integers
    columns_to_int = [
        'age', 'tenure_months', 'ind_new_client', 'client_relationship_status',
        'ind_client_activity', 'province_code', 'client_type_1m', 'number_of_products'
    ]
    df[columns_to_int] = df[columns_to_int].astype(int)

    # Drop irrelevant columns for modeling
    df = df.drop(
        columns=[
            'fetch_date', 'id', 'ind_deceased', 'ind_spouse_employee', 'last_date_as_primary',
            'address_type', 'ind_employee', 'country_of_residence', 'ind_resident', 'province_name', 'registration_date'
        ]
    )

    return df.fillna(0)


def frequency_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features in a DataFrame with their frequency.

    Args:
        X (pd.DataFrame): The DataFrame to encode.

    Returns:
        pd.DataFrame: A new DataFrame where each categorical feature is replaced by its frequency.
    """
    X_copy = X.copy()
    for col in X_copy.columns:
        freq_map = X_copy[col].value_counts(normalize=True).to_dict()
        X_copy[col] = X_copy[col].map(freq_map)
    return X_copy


class DataFrameProcessor(BaseEstimator, TransformerMixin):
    """
    A custom transformer that processes a DataFrame.
    
    This class applies the processing logic defined in process_df().
    """

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> 'DataFrameProcessor':
        """
        Fits the transformer (no-op for this transformer).

        Args:
            X (pd.DataFrame): Input data.
            y (Optional[pd.DataFrame]): Optional target data.

        Returns:
            DataFrameProcessor: The fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Transforms the input DataFrame by applying the processing logic.

        Args:
            X (pd.DataFrame): Input data.
            y (Optional[pd.DataFrame]): Optional target data.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        processed_X = process_df(X.copy())
        if y is not None:
            processed_y = y.loc[processed_X.index]
            return processed_X, processed_y
        return processed_X


# Define a column transformer to handle scaling, one-hot encoding, and frequency encoding
column_transformer = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_columns),  # Scale numerical columns
        ('onehot', OneHotEncoder(drop='first'), one_hot_columns),  # One-hot encode categorical columns
        ('freq', FunctionTransformer(frequency_encoding), freq_encode_columns)  # Frequency encode columns
    ], 
    remainder='passthrough',
    force_int_remainder_cols=False,  # Handle the rest of the columns
)


# Define a pipeline that includes preprocessing and model training
model = Pipeline(steps=[
    ('preprocessor', DataFrameProcessor()),  # Custom DataFrame processor
    ('encoder', column_transformer),  # Apply the column transformer
    ('model', MultiOutputClassifier(
        CatBoostClassifier(
            iterations=CBC_ITERATIONS,  # Number of iterations
            verbose=CBC_VERBOSE,  # Verbosity level
            class_weights=CBC_CLASS_WEIGHTS,  # Class weights to handle imbalance
            random_state=RANDOM_STATE  # Random state for reproducibility
        )
    ))
])
