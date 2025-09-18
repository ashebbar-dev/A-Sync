#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crop Prediction Integration Script

This script integrates the crop prediction system with weather data utilities
to provide a complete solution for predicting suitable crops based on location.
"""

import os
import sys
import json
import argparse
from datetime import datetime

from crop_prediction import CropPredictionSystem
from weather_util import get_complete_data_for_prediction


def setup_argparse():
    """Set up command-line argument parsing.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Predict suitable crops based on location and environmental factors"
    )
    
    # Main command options
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new crop prediction model")
    train_parser.add_argument(
        "--data", 
        default="crop_data.csv",
        help="Path to the training dataset (CSV format)"
    )
    train_parser.add_argument(
        "--output", 
        default="models",
        help="Directory to save the trained model"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict crops for a location")
    predict_parser.add_argument(
        "--location", 
        required=True,
        help="Location name (e.g., 'New Delhi', 'Mumbai')"
    )
    predict_parser.add_argument(
        "--model", 
        help="Path to a trained model file (.joblib)"
    )
    predict_parser.add_argument(
        "--report", 
        action="store_true",
        help="Generate a detailed report"
    )
    predict_parser.add_argument(
        "--output", 
        help="Path to save the prediction report"
    )
    
    # Manual input command
    manual_parser = subparsers.add_parser("manual", help="Predict using manually entered data")
    manual_parser.add_argument(
        "--model", 
        help="Path to a trained model file (.joblib)"
    )
    manual_parser.add_argument(
        "--report", 
        action="store_true",
        help="Generate a detailed report"
    )
    manual_parser.add_argument(
        "--output", 
        help="Path to save the prediction report"
    )
    
    return parser


def train_model(args):
    """Train a new crop prediction model.
    
    Args:
        args: Command-line arguments
        
    Returns:
        bool: True if training was successful, False otherwise
    """
    print(f"Training new model using dataset: {args.data}")
    
    # Initialize the system
    crop_system = CropPredictionSystem()
    
    # Load the dataset
    try:
        crop_system.load_data(args.data)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Explore the dataset
    print("\nExploring dataset...")
    crop_system.explore_data()
    
    # Train the model
    print("\nTraining model...")
    results = crop_system.train_model()
    
    if results is None:
        print("Error training model.")
        return False
    
    # Print training results
    print(f"\nTraining completed with accuracy: {results['accuracy']:.4f}")
    print(f"Cross-validation mean accuracy: {results['cv_mean_accuracy']:.4f}")
    
    # Save the model
    model_path = crop_system.save_model(args.output)
    
    if model_path:
        print(f"\nModel saved to: {model_path}")
        print("\nTo use this model for prediction, run:")
        print(f"python predict_crop.py predict --location 'New Delhi' --model {model_path}")
        return True
    else:
        print("Error saving model.")
        return False


def predict_from_location(args):
    """Predict suitable crops based on location.
    
    Args:
        args: Command-line arguments
        
    Returns:
        bool: True if prediction was successful, False otherwise
    """
    print(f"Predicting suitable crops for location: {args.location}")
    
    # Initialize the system
    crop_system = CropPredictionSystem()
    
    # Load the model if specified
    if args.model:
        print(f"\nLoading model from: {args.model}")
        if not crop_system.load_model(args.model):
            print("Error loading model. Please provide a valid model file.")
            return False
    else:
        # Try to find the most recent model in the models directory
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) 
                          if f.endswith(".joblib")]
            if model_files:
                # Sort by modification time (newest first)
                model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                latest_model = model_files[0]
                print(f"\nLoading most recent model: {latest_model}")
                if not crop_system.load_model(latest_model):
                    print("Error loading model. Please provide a valid model file.")
                    return False
            else:
                print("No model files found. Please train a model first or specify a model file.")
                print("Run: python predict_crop.py train --data crop_data.csv")
                return False
        else:
            print("No models directory found. Please train a model first or specify a model file.")
            print("Run: python predict_crop.py train --data crop_data.csv")
            return False
    
    # Get data for the location
    print(f"\nFetching data for location: {args.location}")
    location, weather, soil = get_complete_data_for_prediction(args.location)
    
    print("\nLocation Information:")
    print(json.dumps(location, indent=2))
    
    print("\nWeather Data:")
    print(json.dumps(weather, indent=2))
    
    print("\nSoil Data:")
    print(json.dumps(soil, indent=2))
    
    # Make prediction
    print("\nPredicting suitable crops...")
    prediction = crop_system.predict_from_location_weather(location, weather, soil)
    
    if prediction is None:
        print("Error making prediction.")
        return False
    
    # Print prediction results
    print("\nPrediction Results:")
    print(f"Recommended Crop: {prediction['predicted_crop']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    
    print("\nAlternative Recommendations:")
    for i, rec in enumerate(prediction['recommendations'], 1):
        print(f"{i}. {rec['crop']} (Probability: {rec['probability']:.2%})")
    
    # Generate report if requested
    if args.report:
        output_file = args.output or f"crop_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        print(f"\nGenerating detailed report: {output_file}")
        report = crop_system.generate_report(prediction, output_file)
        print("\nReport Preview:")
        print("-------------------")
        print(report[:500] + "..." if len(report) > 500 else report)
        print("-------------------")
    
    return True


def predict_from_manual_input(args):
    """Predict suitable crops based on manually entered data.
    
    Args:
        args: Command-line arguments
        
    Returns:
        bool: True if prediction was successful, False otherwise
    """
    print("Predicting suitable crops from manual input")
    
    # Initialize the system
    crop_system = CropPredictionSystem()
    
    # Load the model if specified
    if args.model:
        print(f"\nLoading model from: {args.model}")
        if not crop_system.load_model(args.model):
            print("Error loading model. Please provide a valid model file.")
            return False
    else:
        # Try to find the most recent model in the models directory
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) 
                          if f.endswith(".joblib")]
            if model_files:
                # Sort by modification time (newest first)
                model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                latest_model = model_files[0]
                print(f"\nLoading most recent model: {latest_model}")
                if not crop_system.load_model(latest_model):
                    print("Error loading model. Please provide a valid model file.")
                    return False
            else:
                print("No model files found. Please train a model first or specify a model file.")
                print("Run: python predict_crop.py train --data crop_data.csv")
                return False
        else:
            print("No models directory found. Please train a model first or specify a model file.")
            print("Run: python predict_crop.py train --data crop_data.csv")
            return False
    
    # Get manual input
    print("\nPlease enter the following environmental factors:")
    
    try:
        # Get soil data
        n = float(input("Nitrogen (N) content in soil (mg/kg): "))
        p = float(input("Phosphorus (P) content in soil (mg/kg): "))
        k = float(input("Potassium (K) content in soil (mg/kg): "))
        ph = float(input("pH value of soil: "))
        
        # Get weather data
        temperature = float(input("Temperature (°C): "))
        humidity = float(input("Humidity (%): "))
        rainfall = float(input("Rainfall (mm): "))
        
        # Get location information (optional)
        print("\nLocation information (optional):")
        region = input("Region/Area name: ")
        
        try:
            latitude = float(input("Latitude (optional, press Enter to skip): ") or "0")
            longitude = float(input("Longitude (optional, press Enter to skip): ") or "0")
        except ValueError:
            latitude, longitude = 0, 0
    
    except ValueError as e:
        print(f"\nError: Invalid input. Please enter numeric values.")
        return False
    
    # Prepare input data
    input_data = {
        'N': n, 'P': p, 'K': k, 'temperature': temperature,
        'humidity': humidity, 'ph': ph, 'rainfall': rainfall
    }
    
    # Make prediction
    print("\nPredicting suitable crops...")
    prediction = crop_system.predict_crop(input_data)
    
    if prediction is None:
        print("Error making prediction.")
        return False
    
    # Add location information if provided
    if region or (latitude != 0 and longitude != 0):
        location = {
            'region': region,
            'latitude': latitude,
            'longitude': longitude
        }
        prediction['location'] = location
    
    # Print prediction results
    print("\nPrediction Results:")
    print(f"Recommended Crop: {prediction['predicted_crop']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    
    print("\nAlternative Recommendations:")
    for i, rec in enumerate(prediction['recommendations'], 1):
        print(f"{i}. {rec['crop']} (Probability: {rec['probability']:.2%})")
    
    # Generate report if requested
    if args.report:
        output_file = args.output or f"crop_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        print(f"\nGenerating detailed report: {output_file}")
        report = crop_system.generate_report(prediction, output_file)
        print("\nReport Preview:")
        print("-------------------")
        print(report[:500] + "..." if len(report) > 500 else report)
        print("-------------------")
    
    return True


def main():
    """Main function to handle command-line interface."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "train":
        success = train_model(args)
    elif args.command == "predict":
        success = predict_from_location(args)
    elif args.command == "manual":
        success = predict_from_manual_input(args)
    else:
        parser.print_help()
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())