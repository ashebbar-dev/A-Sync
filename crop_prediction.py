#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crop Prediction System

This module provides functionality to predict suitable crops based on location,
weather conditions, and soil type using machine learning techniques.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

class CropPredictionSystem:
    """A system for predicting suitable crops based on environmental factors."""
    
    def __init__(self, data_path=None):
        """Initialize the crop prediction system.
        
        Args:
            data_path (str, optional): Path to the crop dataset. Defaults to None.
        """
        self.data_path = data_path
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.target = 'label'
        
    def load_data(self, data_path=None):
        """Load the crop dataset from a CSV file.
        
        Args:
            data_path (str, optional): Path to the dataset. Defaults to None.
            
        Returns:
            pandas.DataFrame: The loaded dataset.
        """
        if data_path:
            self.data_path = data_path
            
        if self.data_path is None:
            raise ValueError("Data path not provided. Please provide a valid path to the dataset.")
            
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self):
        """Explore and visualize the dataset.
        
        Returns:
            dict: Summary statistics of the dataset.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        
        # Basic information
        print("\nDataset Information:")
        print(f"Shape: {self.data.shape}")
        print("\nColumn Data Types:")
        print(self.data.dtypes)
        
        # Summary statistics
        print("\nSummary Statistics:")
        summary = self.data.describe()
        print(summary)
        
        # Check for missing values
        print("\nMissing Values:")
        missing = self.data.isnull().sum()
        print(missing)
        
        # Distribution of crops
        if self.target in self.data.columns:
            print("\nCrop Distribution:")
            crop_counts = self.data[self.target].value_counts()
            print(crop_counts)
            
            # Visualize crop distribution
            plt.figure(figsize=(12, 6))
            sns.countplot(y=self.target, data=self.data, order=crop_counts.index)
            plt.title('Distribution of Crops')
            plt.tight_layout()
            plt.savefig('crop_distribution.png')
            plt.close()
            
            # Correlation matrix
            plt.figure(figsize=(10, 8))
            correlation = self.data[self.features].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig('feature_correlation.png')
            plt.close()
        
        return summary.to_dict()
    
    def preprocess_data(self):
        """Preprocess the dataset for model training.
        
        Returns:
            tuple: Preprocessed features and target variables.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None, None
        
        # Check if target column exists
        if self.target not in self.data.columns:
            print(f"Target column '{self.target}' not found in the dataset.")
            return None, None
        
        # Handle missing values if any
        if self.data.isnull().sum().sum() > 0:
            self.data = self.data.dropna()
            print(f"Dropped rows with missing values. New shape: {self.data.shape}")
        
        # Extract features and target
        X = self.data[self.features].copy()
        y = self.data[self.target].copy()
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print("Data preprocessing completed.")
        return X_scaled, y_encoded
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train a Random Forest model for crop prediction.
        
        Args:
            test_size (float, optional): Proportion of the dataset to include in the test split. 
                                         Defaults to 0.2.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            
        Returns:
            dict: Training results including accuracy and model details.
        """
        # Preprocess data
        X, y = self.preprocess_data()
        if X is None or y is None:
            return None
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Testing set size: {X_test.shape[0]} samples")
        
        # Initialize and train the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Training Completed")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"Cross-Validation Scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
        
        # Classification report
        print("\nClassification Report:")
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names)
        print(report)
        
        # Confusion matrix
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Feature importance
        plt.figure(figsize=(10, 6))
        feature_importance = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        # Return results
        results = {
            'accuracy': float(accuracy),
            'cv_mean_accuracy': float(cv_scores.mean()),
            'cv_scores': cv_scores.tolist(),
            'feature_importance': dict(zip(self.features, self.model.feature_importances_.tolist()))
        }
        
        return results
    
    def save_model(self, model_dir='models'):
        """Save the trained model and preprocessing components.
        
        Args:
            model_dir (str, optional): Directory to save the model. Defaults to 'models'.
            
        Returns:
            str: Path to the saved model.
        """
        if self.model is None:
            print("No trained model to save. Please train a model first.")
            return None
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate timestamp for the model filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"crop_prediction_model_{timestamp}.joblib")
        
        # Save model and preprocessing components
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'features': self.features,
            'target': self.target,
            'classes': self.label_encoder.classes_.tolist()
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """Load a previously trained model.
        
        Args:
            model_path (str): Path to the saved model.
            
        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.features = model_data['features']
            self.target = model_data['target']
            
            print(f"Model loaded successfully from {model_path}")
            print(f"Available crop classes: {model_data['classes']}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_crop(self, input_data):
        """Predict suitable crops based on input environmental factors.
        
        Args:
            input_data (dict): Dictionary containing environmental factors.
                Expected keys: 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
                
        Returns:
            dict: Prediction results including the predicted crop and probabilities.
        """
        if self.model is None:
            print("No trained model available. Please train or load a model first.")
            return None
        
        # Validate input data
        missing_features = [f for f in self.features if f not in input_data]
        if missing_features:
            print(f"Missing required features: {missing_features}")
            return None
        
        # Prepare input data
        X = np.array([[input_data[f] for f in self.features]])
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        y_pred = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Get predicted crop name
        predicted_crop = self.label_encoder.inverse_transform([y_pred])[0]
        
        # Get top 3 crop recommendations with probabilities
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_crops = self.label_encoder.inverse_transform(top_indices)
        top_probs = probabilities[top_indices]
        
        recommendations = [{
            'crop': crop,
            'probability': float(prob)
        } for crop, prob in zip(top_crops, top_probs)]
        
        # Prepare result
        result = {
            'predicted_crop': predicted_crop,
            'confidence': float(probabilities[y_pred]),
            'recommendations': recommendations,
            'input_data': input_data
        }
        
        return result
    
    def predict_from_location_weather(self, location, weather_data, soil_data):
        """Predict suitable crops based on location, weather, and soil data.
        
        Args:
            location (dict): Location information (latitude, longitude).
            weather_data (dict): Weather information (temperature, humidity, rainfall).
            soil_data (dict): Soil information (N, P, K, ph).
            
        Returns:
            dict: Prediction results.
        """
        # Combine data into input format
        input_data = {
            'N': soil_data.get('N'),
            'P': soil_data.get('P'),
            'K': soil_data.get('K'),
            'temperature': weather_data.get('temperature'),
            'humidity': weather_data.get('humidity'),
            'ph': soil_data.get('ph'),
            'rainfall': weather_data.get('rainfall')
        }
        
        # Validate input data
        missing_values = [k for k, v in input_data.items() if v is None]
        if missing_values:
            print(f"Missing required values: {missing_values}")
            return None
        
        # Make prediction
        prediction = self.predict_crop(input_data)
        
        if prediction:
            # Add location information to the result
            prediction['location'] = location
            
            # Generate timestamp
            prediction['timestamp'] = datetime.now().isoformat()
            
            return prediction
        
        return None
    
    def generate_report(self, prediction_result, output_file=None):
        """Generate a detailed report from prediction results.
        
        Args:
            prediction_result (dict): The prediction result from predict_crop method.
            output_file (str, optional): Path to save the report. Defaults to None.
            
        Returns:
            str: Report content.
        """
        if not prediction_result:
            print("No prediction result provided.")
            return None
        
        # Create report content
        report = [
            "Crop Prediction Report",
            "======================",
            "",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Add location information if available
        if 'location' in prediction_result:
            location = prediction_result['location']
            report.extend([
                "Location Information:",
                f"  Latitude: {location.get('latitude')}",
                f"  Longitude: {location.get('longitude')}",
                f"  Region: {location.get('region', 'N/A')}",
                ""
            ])
        
        # Add input data
        report.extend([
            "Environmental Factors:",
            f"  Nitrogen (N): {prediction_result['input_data']['N']} mg/kg",
            f"  Phosphorus (P): {prediction_result['input_data']['P']} mg/kg",
            f"  Potassium (K): {prediction_result['input_data']['K']} mg/kg",
            f"  Temperature: {prediction_result['input_data']['temperature']} °C",
            f"  Humidity: {prediction_result['input_data']['humidity']} %",
            f"  pH: {prediction_result['input_data']['ph']}",
            f"  Rainfall: {prediction_result['input_data']['rainfall']} mm",
            ""
        ])
        
        # Add prediction results
        report.extend([
            "Prediction Results:",
            f"  Recommended Crop: {prediction_result['predicted_crop']}",
            f"  Confidence: {prediction_result['confidence']:.2%}",
            "",
            "Alternative Recommendations:"
        ])
        
        # Add alternative recommendations
        for i, rec in enumerate(prediction_result['recommendations'], 1):
            report.append(f"  {i}. {rec['crop']} (Probability: {rec['probability']:.2%})")
        
        report.extend([
            "",
            "Note: These recommendations are based on the provided environmental factors and may vary",
            "with changes in actual field conditions. Consider local agricultural expertise and",
            "seasonal variations when making final planting decisions."
        ])
        
        # Join report lines
        report_content = "\n".join(report)
        
        # Save report if output file is provided
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_content)
                print(f"Report saved to {output_file}")
            except Exception as e:
                print(f"Error saving report: {e}")
        
        return report_content


# Example usage
def main():
    """Example usage of the CropPredictionSystem."""
    # Initialize the system
    crop_system = CropPredictionSystem()
    
    # Example: Load data
    # crop_system.load_data('path/to/crop_data.csv')
    
    # Example: Train model
    # crop_system.train_model()
    
    # Example: Save model
    # crop_system.save_model()
    
    # Example: Load model
    # crop_system.load_model('models/crop_prediction_model.joblib')
    
    # Example: Make prediction
    # sample_input = {
    #     'N': 90, 'P': 42, 'K': 43, 'temperature': 20.87,
    #     'humidity': 82.00, 'ph': 6.5, 'rainfall': 202.93
    # }
    # prediction = crop_system.predict_crop(sample_input)
    # print(json.dumps(prediction, indent=2))
    
    # Example: Generate report
    # report = crop_system.generate_report(prediction, 'crop_prediction_report.txt')
    
    print("Crop Prediction System initialized. Use the provided methods to load data, train models, and make predictions.")


if __name__ == "__main__":
    main()