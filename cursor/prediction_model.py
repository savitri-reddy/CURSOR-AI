import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import os

class PredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = None
        self.feature_names = None
        self.is_trained = False
        
    def create_sample_data(self, data_type="regression"):
        """Create sample data for demonstration"""
        np.random.seed(42)
        
        if data_type == "regression":
            # House price prediction data
            n_samples = 1000
            square_feet = np.random.normal(2000, 500, n_samples)
            bedrooms = np.random.randint(1, 6, n_samples)
            bathrooms = np.random.randint(1, 4, n_samples)
            age = np.random.randint(0, 50, n_samples)
            
            # Generate target variable (house price)
            price = (square_feet * 100 + bedrooms * 25000 + bathrooms * 15000 - age * 1000 + 
                    np.random.normal(0, 10000, n_samples))
            
            data = pd.DataFrame({
                'square_feet': square_feet,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'age': age,
                'price': price
            })
            
        elif data_type == "classification":
            # Customer churn prediction data
            n_samples = 1000
            tenure = np.random.randint(1, 72, n_samples)
            monthly_charges = np.random.normal(65, 30, n_samples)
            total_charges = tenure * monthly_charges + np.random.normal(0, 1000, n_samples)
            contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples)
            
            # Generate target variable (churn)
            churn_prob = 1 / (1 + np.exp(-(-0.1 * tenure + 0.02 * monthly_charges + 
                                           np.random.normal(0, 0.5, n_samples))))
            churn = (churn_prob > 0.5).astype(int)
            
            data = pd.DataFrame({
                'tenure': tenure,
                'monthly_charges': monthly_charges,
                'total_charges': total_charges,
                'contract_type': contract_type,
                'churn': churn
            })
            
        else:  # time_series
            # Sales forecasting data
            dates = pd.date_range('2020-01-01', periods=365, freq='D')
            trend = np.linspace(100, 200, 365)
            seasonality = 20 * np.sin(2 * np.pi * np.arange(365) / 365)
            noise = np.random.normal(0, 10, 365)
            sales = trend + seasonality + noise
            
            data = pd.DataFrame({
                'date': dates,
                'sales': sales
            })
            
        return data
    
    def train_regression_model(self, data):
        """Train a regression model"""
        X = data.drop('price', axis=1)
        y = data['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        
        self.model_type = "regression"
        self.feature_names = X.columns.tolist()
        self.is_trained = True
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
    
    def train_classification_model(self, data):
        """Train a classification model"""
        # Convert categorical variables
        data_encoded = pd.get_dummies(data, columns=['contract_type'])
        
        X = data_encoded.drop('churn', axis=1)
        y = data_encoded['churn']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.model_type = "classification"
        self.feature_names = X.columns.tolist()
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
    
    def make_prediction(self, input_data):
        """Make a prediction using the trained model"""
        if not self.is_trained:
            return "Model not trained yet"
        
        try:
            # Prepare input data
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = pd.DataFrame(input_data)
            
            # Scale features
            input_scaled = self.scaler.transform(input_df)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)
            
            if self.model_type == "classification":
                probability = self.model.predict_proba(input_scaled)
                return {
                    'prediction': int(prediction[0]),
                    'probability': float(probability[0][1]),
                    'prediction_text': 'Churn' if prediction[0] == 1 else 'No Churn'
                }
            else:
                return {
                    'prediction': float(prediction[0]),
                    'prediction_text': f"${prediction[0]:,.2f}"
                }
                
        except Exception as e:
            return f"Error making prediction: {str(e)}"
    
    def save_model(self, filename):
        """Save the trained model"""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, filename)
            return f"Model saved to {filename}"
        else:
            return "No trained model to save"
    
    def load_model(self, filename):
        """Load a trained model"""
        if os.path.exists(filename):
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            return f"Model loaded from {filename}"
        else:
            return "Model file not found" 