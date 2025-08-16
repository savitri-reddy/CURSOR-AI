import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class PlantDiseaseModel:
    def __init__(self):
        self.model = None
        self.class_names = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Cherry___healthy',
            'Cherry___Powdery_mildew',
            'Corn___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn___Common_rust',
            'Corn___healthy',
            'Corn___Northern_Leaf_Blight',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___healthy',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper_bell___Bacterial_spot',
            'Pepper_bell___healthy',
            'Potato___Early_blight',
            'Potato___healthy',
            'Potato___Late_blight',
            'Strawberry___healthy',
            'Strawberry___Leaf_scorch',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___healthy',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
        ]
        self.is_trained = False
        
    def create_sample_data(self, n_samples=1000):
        """Create synthetic plant disease data for demonstration"""
        print("Creating synthetic plant disease data...")
        
        # Generate synthetic image data
        data = []
        for i in range(n_samples):
            # Random plant type and disease
            plant_type = random.choice(['Apple', 'Tomato', 'Potato', 'Corn', 'Grape'])
            disease_status = random.choice(['healthy', 'disease'])
            
            if disease_status == 'healthy':
                class_name = f"{plant_type}___healthy"
            else:
                diseases = {
                    'Apple': ['Apple_scab', 'Black_rot', 'Cedar_apple_rust'],
                    'Tomato': ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold'],
                    'Potato': ['Early_blight', 'Late_blight'],
                    'Corn': ['Cercospora_leaf_spot', 'Common_rust', 'Northern_Leaf_Blight'],
                    'Grape': ['Black_rot', 'Esca_(Black_Measles)', 'Leaf_blight_(Isariopsis_Leaf_Spot)']
                }
                disease = random.choice(diseases.get(plant_type, ['unknown']))
                class_name = f"{plant_type}___{disease}"
            
            # Generate synthetic image features (simulating CNN features)
            features = np.random.normal(0, 1, 1280)  # MobileNetV2 feature size
            
            # Add some correlation between features and disease
            if 'healthy' in class_name:
                features[:100] += np.random.normal(0.5, 0.1, 100)
            else:
                features[100:200] += np.random.normal(0.5, 0.1, 100)
            
            data.append({
                'class_name': class_name,
                'plant_type': plant_type,
                'disease_status': disease_status,
                'features': features
            })
        
        return pd.DataFrame(data)
    
    def build_model(self):
        """Build a plant disease classification model"""
        # Use MobileNetV2 as base model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create model
        self.model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(len(self.class_names), activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train_model(self, data, epochs=10):
        """Train the plant disease model"""
        if self.model is None:
            self.build_model()
        
        print("Training plant disease model...")
        
        # Prepare data for training
        X = np.array([row['features'] for _, row in data.iterrows()])
        y = pd.get_dummies(data['class_name']).values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        self.is_trained = True
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'history': history.history
        }
    
    def predict_disease(self, image_path):
        """Predict plant disease from uploaded image"""
        if self.model is None:
            return "Model not trained yet"
        
        try:
            # Load and preprocess image
            img = Image.open(image_path)
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Make prediction
            predictions = self.model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            predicted_class = self.class_names[predicted_class_idx]
            
            # Parse class name
            if '___' in predicted_class:
                plant_type, disease = predicted_class.split('___', 1)
            else:
                plant_type = "Unknown"
                disease = predicted_class
            
            # Determine if healthy or diseased
            is_healthy = 'healthy' in disease.lower()
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = []
            for idx in top_3_idx:
                class_name = self.class_names[idx]
                if '___' in class_name:
                    plant, dis = class_name.split('___', 1)
                else:
                    plant, dis = "Unknown", class_name
                top_3_predictions.append({
                    'plant_type': plant,
                    'disease': dis,
                    'confidence': float(predictions[0][idx])
                })
            
            return {
                'plant_type': plant_type,
                'disease': disease,
                'is_healthy': is_healthy,
                'confidence': confidence,
                'top_3_predictions': top_3_predictions,
                'all_predictions': dict(zip(self.class_names, predictions[0].tolist()))
            }
            
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
    def get_disease_info(self, disease_name):
        """Get information about a specific disease"""
        disease_info = {
            'Apple_scab': {
                'description': 'A fungal disease caused by Venturia inaequalis',
                'symptoms': 'Dark, scabby lesions on leaves and fruit',
                'treatment': 'Apply fungicides, remove infected plant debris',
                'prevention': 'Plant resistant varieties, maintain good air circulation'
            },
            'Black_rot': {
                'description': 'A fungal disease affecting apples and grapes',
                'symptoms': 'Black, circular lesions on leaves and fruit',
                'treatment': 'Remove infected parts, apply fungicides',
                'prevention': 'Prune properly, avoid overhead irrigation'
            },
            'Early_blight': {
                'description': 'A fungal disease caused by Alternaria solani',
                'symptoms': 'Dark brown spots with concentric rings',
                'treatment': 'Remove infected leaves, apply fungicides',
                'prevention': 'Mulch around plants, avoid overhead watering'
            },
            'Late_blight': {
                'description': 'A devastating disease caused by Phytophthora infestans',
                'symptoms': 'Water-soaked lesions that turn brown',
                'treatment': 'Remove infected plants, apply copper-based fungicides',
                'prevention': 'Plant resistant varieties, avoid overhead irrigation'
            },
            'Bacterial_spot': {
                'description': 'A bacterial disease affecting tomatoes and peppers',
                'symptoms': 'Small, dark spots with yellow halos',
                'treatment': 'Remove infected parts, apply copper-based bactericides',
                'prevention': 'Use disease-free seeds, avoid overhead watering'
            },
            'healthy': {
                'description': 'Plant appears healthy with no visible disease symptoms',
                'symptoms': 'Normal green leaves, healthy growth',
                'treatment': 'Continue current care routine',
                'prevention': 'Maintain good growing conditions, regular monitoring'
            }
        }
        
        # Find matching disease info
        for key in disease_info:
            if key.lower() in disease_name.lower():
                return disease_info[key]
        
        return {
            'description': 'Disease information not available',
            'symptoms': 'Consult a plant expert for accurate diagnosis',
            'treatment': 'General plant care and monitoring recommended',
            'prevention': 'Maintain good growing conditions'
        }
    
    def save_model(self, filename="plant_disease_model.h5"):
        """Save the trained model"""
        if self.model is not None and self.is_trained:
            self.model.save(filename)
            return f"Model saved to {filename}"
        else:
            return "No trained model to save"
    
    def load_model(self, filename):
        """Load a trained model"""
        if os.path.exists(filename):
            self.model = tf.keras.models.load_model(filename)
            self.is_trained = True
            return f"Model loaded from {filename}"
        else:
            return "Model file not found" 