import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prediction_model import PredictionModel
import time

# Page configuration
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🔮",
    layout="wide"
)

# Initialize session state
if "prediction_model" not in st.session_state:
    st.session_state.prediction_model = PredictionModel()

model = st.session_state.prediction_model

# Main title
st.title("🔮 Machine Learning Prediction App")
st.markdown("---")

# Sidebar for model selection
st.sidebar.header("Model Configuration")
model_type = st.sidebar.selectbox(
    "Select Prediction Type",
    ["House Price Prediction (Regression)", "Customer Churn Prediction (Classification)"]
)

# Main content
if model_type == "House Price Prediction (Regression)":
    st.header("🏠 House Price Prediction")
    
    # Create sample data
    if st.button("Generate Sample Data"):
        with st.spinner("Generating house price data..."):
            data = model.create_sample_data("regression")
            st.session_state.regression_data = data
            st.success("Sample data generated!")
    
    if "regression_data" in st.session_state:
        data = st.session_state.regression_data
        
        # Display data
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Sample Data")
            st.dataframe(data.head())
            
            # Data visualization
            fig = px.scatter(data, x='square_feet', y='price', 
                           title='House Price vs Square Feet')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Data Statistics")
            st.write(data.describe())
        
        # Train model
        if st.button("Train Model"):
            with st.spinner("Training regression model..."):
                results = model.train_regression_model(data)
                st.session_state.regression_results = results
                st.success("Model trained successfully!")
        
        if "regression_results" in st.session_state:
            results = st.session_state.regression_results
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MSE", f"{results['mse']:,.2f}")
            with col2:
                st.metric("RMSE", f"{results['rmse']:,.2f}")
            with col3:
                st.metric("Model Type", "Random Forest")
            
            # Feature importance
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame(
                list(results['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        orientation='h', title='Feature Importance')
            st.plotly_chart(fig, use_container_width=True)
            
            # Make predictions
            st.subheader("Make Predictions")
            
            col1, col2 = st.columns(2)
            with col1:
                square_feet = st.number_input("Square Feet", min_value=500, max_value=5000, value=2000)
                bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=3)
            
            with col2:
                bathrooms = st.number_input("Bathrooms", min_value=1, max_value=4, value=2)
                age = st.number_input("Age (years)", min_value=0, max_value=50, value=10)
            
            if st.button("Predict Price"):
                input_data = {
                    'square_feet': square_feet,
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'age': age
                }
                
                prediction = model.make_prediction(input_data)
                
                if isinstance(prediction, dict):
                    st.success(f"Predicted House Price: {prediction['prediction_text']}")
                    
                    # Show prediction confidence
                    st.info("💡 This prediction is based on the trained model's analysis of similar properties.")
                else:
                    st.error(prediction)

elif model_type == "Customer Churn Prediction (Classification)":
    st.header("👥 Customer Churn Prediction")
    
    # Create sample data
    if st.button("Generate Sample Data"):
        with st.spinner("Generating customer churn data..."):
            data = model.create_sample_data("classification")
            st.session_state.classification_data = data
            st.success("Sample data generated!")
    
    if "classification_data" in st.session_state:
        data = st.session_state.classification_data
        
        # Display data
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Sample Data")
            st.dataframe(data.head())
            
            # Data visualization
            fig = px.histogram(data, x='tenure', color='churn', 
                             title='Customer Tenure vs Churn')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Churn Distribution")
            churn_counts = data['churn'].value_counts()
            fig = px.pie(values=churn_counts.values, names=['No Churn', 'Churn'],
                        title='Churn Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Train model
        if st.button("Train Model"):
            with st.spinner("Training classification model..."):
                results = model.train_classification_model(data)
                st.session_state.classification_results = results
                st.success("Model trained successfully!")
        
        if "classification_results" in st.session_state:
            results = st.session_state.classification_results
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{results['accuracy']:.2%}")
            with col2:
                st.metric("Model Type", "Random Forest")
            
            # Classification report
            st.subheader("Classification Report")
            st.text(results['classification_report'])
            
            # Feature importance
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame(
                list(results['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        orientation='h', title='Feature Importance')
            st.plotly_chart(fig, use_container_width=True)
            
            # Make predictions
            st.subheader("Make Predictions")
            
            col1, col2 = st.columns(2)
            with col1:
                tenure = st.number_input("Tenure (months)", min_value=1, max_value=72, value=12)
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=20, max_value=150, value=65)
            
            with col2:
                total_charges = st.number_input("Total Charges ($)", min_value=100, max_value=10000, value=1000)
                contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            
            if st.button("Predict Churn"):
                input_data = {
                    'tenure': tenure,
                    'monthly_charges': monthly_charges,
                    'total_charges': total_charges,
                    'contract_type': contract_type
                }
                
                prediction = model.make_prediction(input_data)
                
                if isinstance(prediction, dict):
                    if prediction['prediction'] == 1:
                        st.error(f"Prediction: {prediction['prediction_text']} (Probability: {prediction['probability']:.2%})")
                    else:
                        st.success(f"Prediction: {prediction['prediction_text']} (Probability: {prediction['probability']:.2%})")
                    
                    # Show prediction confidence
                    st.info("💡 This prediction is based on the trained model's analysis of similar customer patterns.")
                else:
                    st.error(prediction)

# Model management
st.sidebar.markdown("---")
st.sidebar.header("Model Management")

if st.sidebar.button("Save Model"):
    if model.is_trained:
        result = model.save_model("trained_model.pkl")
        st.sidebar.success(result)
    else:
        st.sidebar.error("No trained model to save")

if st.sidebar.button("Load Model"):
    result = model.load_model("trained_model.pkl")
    st.sidebar.success(result)

# Information section
with st.expander("ℹ️ How Predictions Work"):
    st.write("""
    **Machine Learning Prediction Process:**
    
    1. **Data Collection**: Gather relevant features and target variables
    2. **Data Preprocessing**: Clean, scale, and prepare data for training
    3. **Model Training**: Train algorithms on historical data patterns
    4. **Feature Engineering**: Identify important variables that influence predictions
    5. **Model Evaluation**: Assess performance using metrics like accuracy, MSE, etc.
    6. **Prediction Generation**: Use trained model to make predictions on new data
    
    **Types of Predictions:**
    - **Regression**: Predict continuous values (e.g., house prices, sales forecasts)
    - **Classification**: Predict categories (e.g., churn/no churn, spam/not spam)
    - **Time Series**: Predict future values based on temporal patterns
    
    **Key Components:**
    - **Features**: Input variables used to make predictions
    - **Target**: The variable we want to predict
    - **Model**: The algorithm that learns patterns from data
    - **Accuracy**: How well the model performs on unseen data
    """)

# Footer
st.markdown("---")
st.markdown("🔮 Built with Streamlit and Scikit-learn | Made for educational purposes") 