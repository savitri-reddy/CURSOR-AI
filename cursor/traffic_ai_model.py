import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import os
from datetime import datetime, timedelta
import random

class TrafficAIModel:
    def __init__(self):
        self.traffic_model = None
        self.congestion_model = None
        self.route_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def create_traffic_data(self, n_samples=10000):
        """Create synthetic traffic flow data"""
        np.random.seed(42)
        
        # Generate time-based features
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='H')
        
        # Generate location data
        locations = ['Downtown', 'Highway A', 'Highway B', 'City Center', 'Suburbs', 
                    'Airport Road', 'University Area', 'Shopping District', 'Industrial Zone']
        
        # Generate traffic data
        data = []
        for i in range(n_samples):
            date = dates[i]
            location = random.choice(locations)
            
            # Time-based patterns
            hour = date.hour
            day_of_week = date.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            is_rush_hour = 1 if (hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19) else 0
            
            # Base traffic flow
            base_flow = 1000
            
            # Add time-based variations
            if is_rush_hour:
                base_flow += 500
            if is_weekend:
                base_flow -= 200
            if hour >= 22 or hour <= 6:
                base_flow -= 400
                
            # Location-based variations
            location_multipliers = {
                'Downtown': 1.5, 'Highway A': 1.8, 'Highway B': 1.6,
                'City Center': 1.4, 'Suburbs': 0.8, 'Airport Road': 1.2,
                'University Area': 1.1, 'Shopping District': 1.3, 'Industrial Zone': 0.9
            }
            
            base_flow *= location_multipliers[location]
            
            # Add weather effects (simulated)
            weather_conditions = ['Clear', 'Rain', 'Snow', 'Fog']
            weather = random.choice(weather_conditions)
            weather_multipliers = {'Clear': 1.0, 'Rain': 0.8, 'Snow': 0.6, 'Fog': 0.7}
            base_flow *= weather_multipliers[weather]
            
            # Add random noise
            base_flow += np.random.normal(0, 100)
            base_flow = max(0, base_flow)
            
            # Calculate congestion level
            if base_flow > 1500:
                congestion_level = 'High'
            elif base_flow > 1000:
                congestion_level = 'Medium'
            else:
                congestion_level = 'Low'
            
            # Calculate average speed
            if congestion_level == 'High':
                avg_speed = np.random.normal(20, 5)
            elif congestion_level == 'Medium':
                avg_speed = np.random.normal(40, 8)
            else:
                avg_speed = np.random.normal(60, 10)
            
            avg_speed = max(5, min(80, avg_speed))
            
            data.append({
                'timestamp': date,
                'location': location,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'is_rush_hour': is_rush_hour,
                'weather': weather,
                'traffic_flow': base_flow,
                'avg_speed': avg_speed,
                'congestion_level': congestion_level
            })
        
        return pd.DataFrame(data)
    
    def train_traffic_prediction_model(self, data):
        """Train model to predict traffic flow"""
        # Prepare features
        feature_cols = ['hour', 'day_of_week', 'is_weekend', 'is_rush_hour']
        
        # Encode categorical variables
        data_encoded = data.copy()
        data_encoded['location_encoded'] = self.label_encoder.fit_transform(data['location'])
        data_encoded['weather_encoded'] = LabelEncoder().fit_transform(data['weather'])
        
        feature_cols.extend(['location_encoded', 'weather_encoded'])
        
        X = data_encoded[feature_cols]
        y = data_encoded['traffic_flow']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.traffic_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.traffic_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.traffic_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'feature_importance': dict(zip(feature_cols, self.traffic_model.feature_importances_))
        }
    
    def train_congestion_model(self, data):
        """Train model to predict congestion levels"""
        # Prepare features
        feature_cols = ['hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'traffic_flow']
        
        # Encode categorical variables
        data_encoded = data.copy()
        data_encoded['location_encoded'] = self.label_encoder.fit_transform(data['location'])
        data_encoded['weather_encoded'] = LabelEncoder().fit_transform(data['weather'])
        
        feature_cols.extend(['location_encoded', 'weather_encoded'])
        
        X = data_encoded[feature_cols]
        y = data_encoded['congestion_level']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.congestion_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.congestion_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.congestion_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': dict(zip(feature_cols, self.congestion_model.feature_importances_))
        }
    
    def predict_traffic_flow(self, input_data):
        """Predict traffic flow for given conditions"""
        if self.traffic_model is None:
            return "Traffic model not trained yet"
        
        try:
            # Prepare input
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = pd.DataFrame(input_data)
            
            # Encode categorical variables
            if 'location' in input_df.columns:
                input_df['location_encoded'] = self.label_encoder.transform(input_df['location'])
            if 'weather' in input_df.columns:
                input_df['weather_encoded'] = LabelEncoder().fit_transform(input_df['weather'])
            
            # Select features
            feature_cols = ['hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 
                          'location_encoded', 'weather_encoded']
            X = input_df[feature_cols]
            
            # Scale and predict
            X_scaled = self.scaler.transform(X)
            prediction = self.traffic_model.predict(X_scaled)[0]
            
            return {
                'predicted_flow': float(prediction),
                'flow_category': self._categorize_flow(prediction)
            }
            
        except Exception as e:
            return f"Error making prediction: {str(e)}"
    
    def predict_congestion(self, input_data):
        """Predict congestion level for given conditions"""
        if self.congestion_model is None:
            return "Congestion model not trained yet"
        
        try:
            # Prepare input
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = pd.DataFrame(input_data)
            
            # Encode categorical variables
            if 'location' in input_df.columns:
                input_df['location_encoded'] = self.label_encoder.transform(input_df['location'])
            if 'weather' in input_df.columns:
                input_df['weather_encoded'] = LabelEncoder().fit_transform(input_df['weather'])
            
            # Select features
            feature_cols = ['hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'traffic_flow',
                          'location_encoded', 'weather_encoded']
            X = input_df[feature_cols]
            
            # Scale and predict
            X_scaled = self.scaler.transform(X)
            prediction = self.congestion_model.predict(X_scaled)[0]
            probability = self.congestion_model.predict_proba(X_scaled)[0]
            
            return {
                'predicted_congestion': prediction,
                'confidence': float(max(probability)),
                'all_probabilities': dict(zip(self.congestion_model.classes_, probability))
            }
            
        except Exception as e:
            return f"Error making prediction: {str(e)}"
    
    def optimize_route(self, start_location, end_location, current_time, weather):
        """Optimize route based on traffic conditions"""
        # Simulate route options
        routes = [
            {
                'name': 'Fastest Route',
                'distance': 15.2,
                'estimated_time': 25,
                'traffic_level': 'Medium',
                'congestion': 'Low'
            },
            {
                'name': 'Scenic Route',
                'distance': 18.5,
                'estimated_time': 35,
                'traffic_level': 'Low',
                'congestion': 'Low'
            },
            {
                'name': 'Highway Route',
                'distance': 12.8,
                'estimated_time': 20,
                'traffic_level': 'High',
                'congestion': 'Medium'
            }
        ]
        
        # Adjust based on time and weather
        # Handle both string and datetime objects
        if isinstance(current_time, str):
            try:
                from datetime import datetime
                # Parse time string like "08:00" or "8:00"
                if ':' in current_time:
                    hour = int(current_time.split(':')[0])
                else:
                    hour = int(current_time)
            except:
                hour = 12  # Default to noon if parsing fails
        else:
            hour = current_time.hour
            
        is_rush_hour = (hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19)
        
        for route in routes:
            if is_rush_hour:
                route['estimated_time'] += 10
                if route['traffic_level'] == 'High':
                    route['estimated_time'] += 15
                    route['congestion'] = 'High'
            
            if weather == 'Rain':
                route['estimated_time'] += 5
            elif weather == 'Snow':
                route['estimated_time'] += 15
        
        # Sort by estimated time
        routes.sort(key=lambda x: x['estimated_time'])
        
        return routes
    
    def _categorize_flow(self, flow):
        """Categorize traffic flow"""
        if flow > 1500:
            return 'Heavy'
        elif flow > 1000:
            return 'Moderate'
        else:
            return 'Light'
    
    def get_traffic_insights(self, data):
        """Generate traffic insights from data"""
        insights = {
            'peak_hours': data.groupby('hour')['traffic_flow'].mean().nlargest(3).index.tolist(),
            'busiest_locations': data.groupby('location')['traffic_flow'].mean().nlargest(3).index.tolist(),
            'congestion_by_day': data.groupby('day_of_week')['congestion_level'].apply(
                lambda x: (x == 'High').sum() / len(x)).to_dict(),
            'weather_impact': data.groupby('weather')['traffic_flow'].mean().to_dict()
        }
        return insights
    
    def save_models(self, filename_prefix="traffic_models"):
        """Save trained models"""
        if self.traffic_model is not None:
            joblib.dump({
                'traffic_model': self.traffic_model,
                'congestion_model': self.congestion_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder
            }, f"{filename_prefix}.pkl")
            return f"Models saved to {filename_prefix}.pkl"
        else:
            return "No trained models to save"
    
    def load_models(self, filename):
        """Load trained models"""
        if os.path.exists(filename):
            model_data = joblib.load(filename)
            self.traffic_model = model_data['traffic_model']
            self.congestion_model = model_data['congestion_model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.is_trained = True
            return f"Models loaded from {filename}"
        else:
            return "Model file not found" 