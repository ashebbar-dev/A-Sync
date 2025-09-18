#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crop Prediction System Demo

This script demonstrates how to use the CropPredictionSystem class to predict
suitable crops based on location, weather, and soil data.
"""

import os
import json
from crop_prediction import CropPredictionSystem

def main():
    """Demonstrate the crop prediction system."""
    # Initialize the system
    print("Initializing Crop Prediction System...")
    crop_system = CropPredictionSystem()
    
    # Load the sample dataset
    print("\nLoading sample dataset...")
    crop_system.load_data('crop_data.csv')
    
    # Explore the dataset
    print("\nExploring dataset...")
    crop_system.explore_data()
    
    # Train the model
    print("\nTraining the prediction model...")
    results = crop_system.train_model()
    
    # Print training results
    print("\nTraining Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Cross-Validation Mean Accuracy: {results['cv_mean_accuracy']:.4f}")
    
    # Save the model
    print("\nSaving the trained model...")
    model_path = crop_system.save_model()
    
    # Make predictions with sample data
    print("\nMaking predictions with sample data...")
    
    # Example 1: Rice-like conditions
    sample_rice = {
        'N': 85, 'P': 50, 'K': 42, 'temperature': 22.0,
        'humidity': 81.0, 'ph': 6.7, 'rainfall': 220.0
    }
    
    # Example 2: Apple-like conditions
    sample_apple = {
        'N': 45, 'P': 30, 'K': 30, 'temperature': 24.5,
        'humidity': 42.0, 'ph': 7.7, 'rainfall': 185.0
    }
    
    # Example 3: Cotton-like conditions
    sample_cotton = {
        'N': 75, 'P': 25, 'K': 30, 'temperature': 23.5,
        'humidity': 60.5, 'ph': 7.0, 'rainfall': 130.0
    }
    
    # Make predictions
    print("\nPrediction 1 (Rice-like conditions):")
    prediction_rice = crop_system.predict_crop(sample_rice)
    print(json.dumps(prediction_rice, indent=2))
    
    print("\nPrediction 2 (Apple-like conditions):")
    prediction_apple = crop_system.predict_crop(sample_apple)
    print(json.dumps(prediction_apple, indent=2))
    
    print("\nPrediction 3 (Cotton-like conditions):")
    prediction_cotton = crop_system.predict_crop(sample_cotton)
    print(json.dumps(prediction_cotton, indent=2))
    
    # Generate reports
    print("\nGenerating prediction reports...")
    
    # Add location information for demonstration
    location_info = {
        'latitude': 28.6139,
        'longitude': 77.2090,
        'region': 'Northern India'
    }
    
    # Add location to prediction results
    prediction_rice['location'] = location_info
    
    # Generate and print report
    print("\nSample Report:")
    report = crop_system.generate_report(prediction_rice, 'rice_prediction_report.txt')
    print(report)
    
    # Demonstrate prediction from location and weather data
    print("\nPredicting crop from location, weather, and soil data:")
    
    # Sample weather data
    weather_data = {
        'temperature': 22.0,
        'humidity': 81.0,
        'rainfall': 220.0
    }
    
    # Sample soil data
    soil_data = {
        'N': 85,
        'P': 50,
        'K': 42,
        'ph': 6.7
    }
    
    # Make prediction
    location_prediction = crop_system.predict_from_location_weather(
        location_info, weather_data, soil_data
    )
    
    print(json.dumps(location_prediction, indent=2))
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()