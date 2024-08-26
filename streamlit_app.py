import streamlit as st
import streamlit_antd_components as sac
import os
import io
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


# Set page config
st.set_page_config(
    page_title="ML model",
    page_icon=":rocket:",
    layout="wide",
)


# Create a session state variable to store the uploaded file
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None



#Menu Bar
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


#Home bar
if selected == 0:
    st.header("Welcome to ML Model")

    st.write("This is a machine learning model that allows you to upload your dataset, select the target column, and train a simple linear regression model. The model will then make predictions on the uploaded data.")

    st.subheader("Features")

    st.write("The following features are available in this model:")

    features = [
        "Data Ingestion: Upload your dataset in CSV or Excel format",
        "Target Column Selection: Select the column you want to predict",
        "Model Training: Train a simple linear regression model on your data",
        "Predictions: Get predictions on your uploaded data"
    ]

    for feature in features:
        st.write(f"* {feature}")

    st.subheader("How it Works")

    st.write("Here's a step-by-step guide on how to use this model:")

    steps = [
        "Upload your dataset using the file uploader",
        "Select the target column from the dropdown menu",
        "Click the 'Train Model' button to train the model",
        "Get predictions on your uploaded data"
    ]

    for step in steps:
        st.write(f"* {step}")

    st.subheader("Benefits")

    st.write("Using this model, you can:")

    benefits = [
        "Quickly upload and analyze your dataset",
        "Select the target column with ease",
        "Train a simple linear regression model with minimal effort",
        "Get accurate predictions on your uploaded data"
    ]

    for benefit in benefits:
        st.write(f"* {benefit}")
    
uploaded_file = None

# Data Ingestion tab
if selected == 3:
    st.header("Data Ingestion")
    
    # Create a file uploader
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], accept_multiple_files=False)
    
    if uploaded_file:
        # Store the uploaded file in the session state
        st.session_state.uploaded_file = uploaded_file
        
        # Create the uploads directory if it doesn't exist
        uploads_dir = "uploads"
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
    
        # Handle the uploaded file
        file_path = os.path.join(uploads_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("File uploaded successfully!")
    
        # Get the file name and path
        file_name = uploaded_file.name
        file_path = file_path
    
        # Display the file name and path
        st.write(f"File name: {file_name}")
        st.write(f"File path: {file_path}")
    
        # Load the uploaded data
        if file_name.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_name.endswith('.xlsx'):
            data = pd.read_excel(file_path)
    
        # Display the data dimensions
        st.write(f"Data shape: {data.shape}")

        sac.divider(label='Table', icon='Table', align='center', color='gray')
        
        # Display the data table
        st.write("Data Table:")
        st.write(data.head(10))  # display the first 10 rows of the data

    
        # Define the model file path
        model_file_path = "linear_reg_model(1).pkl"
    
        # Get the column names
        columns = data.columns.tolist()
    
        # Create a dropdown to select the target column
        target_column = st.selectbox("Select the target column", columns)
    
        # Select the correct features
        features = [col for col in columns if col != target_column]
    
        # Define X as the feature columns
        X = data[features]
    
        # Define y as the target column
        y = data[target_column]
    
        try:
            y = pd.to_numeric(y, errors='coerce')
        except ValueError as e:
            st.error(f"An error occurred while converting the target column to numeric: {e}")
    
        # Train and save the model if it doesn't exist
        if not os.path.exists(model_file_path):
            # Handle missing values
            numeric_features = [col for col in X.columns if X[col].dtype.kind in 'bifc']
    
            if not numeric_features:
                st.error("No numeric features found in the data. Please check your data and try again.")
            else:
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                ])
    
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                    ]
                )
    
                # Train a simple linear regression model
                model = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', LinearRegression())])
                model.fit(X, y)
    
                # Save the model to a file
                with open(model_file_path, 'wb') as handle:
                    pickle.dump(model, handle)
        
    
        # Load the pre-trained model
        with open(model_file_path, 'rb') as handle:
            model = pickle.load(handle)
    
        # Get the column names expected by the ColumnTransformer
        expected_columns = model.named_steps['preprocessor'].transformers_[0][2]
        
        # Get the actual columns in the data
        actual_columns = X.columns
        
        # Get the common columns between the expected columns and the actual columns
        common_columns = list(set(expected_columns) & set(actual_columns))
        
        # Check if all expected columns are present in X
        if not common_columns:
            st.error("No common columns found between the expected columns and the actual columns. Please check your data and try again.")
        else:
            # Use the common columns to make predictions
            X_common = X[common_columns]
        
            # Create a new ColumnTransformer with the common columns
            new_transformer = ColumnTransformer(
                transformers=[
                    ('num', model.named_steps['preprocessor'].transformers_[0][1], common_columns),
                ]
            )
        
            # Fit the new ColumnTransformer to the data
            new_transformer.fit(X_common)
        
            # Create a new Pipeline with the new ColumnTransformer
            new_model = Pipeline(steps=[('preprocessor', new_transformer),
                                      ('regressor', model.named_steps['regressor'])])
        
            # Fit the new Pipeline to the data
            new_model.fit(X_common, y)
        
            # Make predictions with the new model
            y_pred = new_model.predict(X_common)
        
            # Calculate the accuracy score (R-squared)
            r2 = r2_score(y, y_pred)
        
            # Calculate the Mean Squared Error (MSE)
            mse = mean_squared_error(y, y_pred)
        
            
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Display the accuracy score (R-squared)
            st.subheader("R2 Score")
            st.write(f"R-squared: {r2:.2f}")
    
            # Display the Mean Squared Error (MSE)
            st.subheader("MSE Score")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")

            # Display the predictions
            st.subheader("Prediction Result")
            st.write("Predictions:")
            st.write(y_pred)

# Data Transformation tab
if selected == 4:
    st.header("Data Transformation")
    
    # Check if a file has been uploaded
    if 'uploaded_file' in st.session_state:
        # Get the uploaded file
        uploaded_file = st.session_state.uploaded_file
        
        # Check if the uploaded file is not empty
        if uploaded_file.size > 0:
            # Load the uploaded data
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(io.StringIO(uploaded_file.getbuffer().decode('utf-8')))
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file.getbuffer())
            
            # Display the data dimensions
            st.write(f"Original Data Shape: {data.shape}")
            
            # Display the data table
            st.write("Original Data Table:")
            st.write(data.head(10))  # display the first 10 rows of the data
            
            # Add a feature to remove columns
            columns_to_remove = st.multiselect("Select columns to remove:", data.columns)
            if columns_to_remove:
                data = data.drop(columns=columns_to_remove)
                st.write("Columns removed successfully!")
            
            # Display the updated data dimensions
            st.write(f"Updated Data Shape: {data.shape}")
            
            # Display the updated data table
            st.write("Updated Data Table:")
            st.write(data.head(10))  # display the first 10 rows of the updated data
            
            # Store the updated data in the session state
            st.session_state.data = data
        else:
            st.write("The uploaded file is empty.")
    else:
        st.write("Please upload a file first.")

# Freeze the Learning tab
if selected == 6:
    st.header("Freeze the Learning")

    if st.session_state.uploaded_file is None:
        st.error("Please upload a file in the Data Ingestion section")
    else:
        # Load the uploaded file
        file_path = "uploads/" + st.session_state.uploaded_file.name
        if st.session_state.uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif st.session_state.uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(file_path)

        # Display the uploaded data
        st.write("Uploaded Data:")
        st.write(data)

        # Get the target column
        target_column = st.selectbox("Select the target column", data.columns)

        # Create a pipeline for the model
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

        # Define the models
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            #"XGBoost": XGBRegressor()
        }

        # Train the models
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Check if the target column is numeric
        if pd.api.types.is_numeric_dtype(y):
            for model_name, model in models.items():
                pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('model', model)])
                pipeline.fit(X, y)

                # Make predictions
                predictions = pipeline.predict(X)

                # Display the predictions
                st.write(f"{model_name} Predictions:")
                st.write(predictions)

                # Evaluate the model
                r2 = r2_score(y, predictions)
                mse = mean_squared_error(y, predictions)
                st.write(f"{model_name} R2 Score:", r2)
                st.write(f"{model_name} Mean Squared Error:", mse)
        else:
            st.error("The target column must be numeric.")
