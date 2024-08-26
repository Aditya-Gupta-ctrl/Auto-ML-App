import streamlit as st
import streamlit_antd_components as sac
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split

def load_data(file):
    """
    Load data from a CSV or Excel file.

    Args:
        file (UploadedFile): The uploaded file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    try:
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            data = pd.read_excel(file)
        else:
            st.error("Please upload a CSV or Excel file.")
            return None
        
        # Check if the file is empty
        if data.empty:
            st.error("The file is empty. Please upload a file with data.")
            return None
        
        return data
    except pd.errors.EmptyDataError:
        st.error("The file is empty. Please upload a file with data.")
        return None
    except pd.errors.ParserError:
        st.error("Error parsing the file. Please check the file format.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}")
        return None

# Function to preprocess data
def preprocess_data(data, target_column):
    """
    Preprocess the data by handling missing values and encoding categorical variables.

    Args:
        data (pd.DataFrame): The data to preprocess.
        target_column (str): The target column.

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    numeric_features = [col for col in numeric_features if col != target_column]
    categorical_features = [col for col in categorical_features if col != target_column]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('le', LabelEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed, y

# Function to train models
def train_models(X, y):
    """
    Train multiple models on the data.

    Args:
        X (pd.DataFrame): The preprocessed data.
        y (pd.Series): The target variable.

    Returns:
        dict: A dictionary containing the trained models and their performance metrics.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "AdaBoost": AdaBoostRegressor(),
    }
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for model_name, model in models.items():
        pipeline = Pipeline(steps=[('model', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        results[model_name] = {'model': pipeline, 'r2': r2, 'mse': mse, 'mae': mae, 'mape': mape}
    return results

# Set page config
st.set_page_config(
    page_title="ML model",
    page_icon=":rocket:",
    layout="wide",
)

# Create a session state variable to store the uploaded file
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Menu Bar
with st.sidebar:
    selected = sac.menu([
        sac.MenuItem('home', icon='house-fill'),
        sac.MenuItem(type='divider'),
        sac.MenuItem('products', icon='box-fill', children=[
            sac.MenuItem('Data Ingestion'),           
            sac.MenuItem('Data Transformation', icon='', description=''),
            sac.MenuItem('Auto Train ML Model', icon=''),
            sac.MenuItem('Freeze the Learning', icon=''),
        ]),
        sac.MenuItem('disabled', disabled=True),
        sac.MenuItem(type='divider'),
        sac.MenuItem('link', type='group', children=[
            sac.MenuItem('@1', icon='', href=''),
            sac.MenuItem('@2', icon='', href=''),
        ]),
    ], size='lg', variant='left-bar', color='grape', open_all=True, return_index=True)

# Home bar
if selected == 0:
    st.title("Home")
    st.write("Welcome to the ML model application!")

# Data Ingestion
elif selected == 3:
    st.title("Data Ingestion")
    file_uploader = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])
    if file_uploader is not None:
        st.session_state.uploaded_file = file_uploader
        data = load_data(st.session_state.uploaded_file)
        if data is not None:
            st.write("File uploaded successfully!")
            st.write("Data preview:")
            st.dataframe(data.head(10))  # Show the first 10 rows of the data
        else:
            st.write("Error loading data. Please check the file format.")
    else:
        st.write("Please upload a CSV or Excel file.")

# Data Transformation
elif selected == 4:
    st.title("Data Transformation")
    if st.session_state.uploaded_file is not None:
        data = load_data(st.session_state.uploaded_file)
        if data is not None:
            st.write("Data loaded successfully!")
            target_column = st.selectbox("Select the target column", data.columns)
            X, y = preprocess_data(data, target_column)
            st.write("Data preprocessed successfully!")
            st.write("Features:")
            st.write(X.head())
            st.write("Target:")
            st.write(y.head())
        else:
            st.write("Error loading data. Please check the file format.")
    else:
        st.write("Please upload a CSV or Excel file first.")

# Auto Train ML Model
elif selected == 5:
    st.title("Auto Train ML Model")
    if st.session_state.uploaded_file is not None:
        data = load_data(st.session_state.uploaded_file)
        if data is not None:
            target_column = st.selectbox("Select the target column", data.columns)
            X, y = preprocess_data(data, target_column)
            models = train_models(X, y)
            st.write("Models trained successfully!")
            model_names = list(models.keys())
            selected_model = st.selectbox("Select a model", model_names)
            st.write(f"Model: {selected_model}")
            st.write(f"R2 score: {models[selected_model]['r2']:.3f}")
            st.write(f"MSE: {models[selected_model]['mse']:.3f}")
            st.write(f"MAE: {models[selected_model]['mae']:.3f}")
            st.write(f"MAPE: {models[selected_model]['mape']:.3f}")
        else:
            st.write("Error loading data. Please check the file format.")
    else:
        st.write("Please upload a CSV or Excel file first.")

# Freeze the Learning
elif selected == 6:
    st.title("Freeze the Learning")
    if st.session_state.uploaded_file is not None:
        data = load_data(st.session_state.uploaded_file)
        if data is not None:
            target_column = st.selectbox("Select the target column", data.columns)
            X, y = preprocess_data(data, target_column)
            models = train_models(X, y)
            st.write("Models trained successfully!")
            model_names = list(models.keys())
            selected_models = st.multiselect("Select models to compare", model_names)
            if selected_models:
                st.write("Model comparison:")
                for model_name in selected_models:
                    st.write(f"Model: {model_name}")
                    st.write(f"R2 score: {models[model_name]['r2']:.3f}")
                    st.write(f"MSE: {models[model_name]['mse']:.3f}")
                    st.write(f"MAE: {models[model_name]['mae']:.3f}")
                    st.write(f"MAPE: {models[model_name]['mape']:.3f}")
                    st.write("")
            else:
                st.write("Please select at least one model to compare.")
        else:
            st.write("Error loading data. Please check the file format.")
    else:
        st.write("Please upload a CSV or Excel file first.")
