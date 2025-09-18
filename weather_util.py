#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weather Data Utility

This module provides functions to fetch weather data from external APIs
for use with the crop prediction system.
"""

import os
import json
import requests
from datetime import datetime

# Default API key (replace with your own)
DEFAULT_API_KEY = "YOUR_API_KEY_HERE"

def get_weather_data(latitude, longitude, api_key=None):
    """Fetch current weather data for a location using OpenWeatherMap API.
    
    Args:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location
        api_key (str, optional): OpenWeatherMap API key. Defaults to None.
        
    Returns:
        dict: Weather data including temperature, humidity, and rainfall
    """
    # Use provided API key or default
    api_key = api_key or os.environ.get("OPENWEATHERMAP_API_KEY") or DEFAULT_API_KEY
    
    if api_key == "YOUR_API_KEY_HERE":
        print("Warning: Using placeholder API key. Please provide a valid OpenWeatherMap API key.")
        print("You can get a free API key at: https://openweathermap.org/api")
        print("Using simulated weather data instead.")
        return simulate_weather_data(latitude, longitude)
    
    # API endpoint
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
    
    try:
        # Make API request
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant weather information
        weather_data = {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'rainfall': data.get('rain', {}).get('1h', 0) * 24  # Convert hourly to daily (mm)
        }
        
        # If no rainfall data, try to get it from another endpoint (historical data)
        if weather_data['rainfall'] == 0:
            weather_data['rainfall'] = get_historical_rainfall(latitude, longitude, api_key)
        
        return weather_data
    
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        print("Using simulated weather data instead.")
        return simulate_weather_data(latitude, longitude)


def get_historical_rainfall(latitude, longitude, api_key):
    """Get historical rainfall data for the past 5 days.
    
    Args:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location
        api_key (str): OpenWeatherMap API key
        
    Returns:
        float: Average daily rainfall in mm
    """
    # This would normally use the OpenWeatherMap historical API
    # For simplicity, we'll return a simulated value
    return 10.0  # Simulated 10mm average daily rainfall


def simulate_weather_data(latitude, longitude):
    """Generate simulated weather data based on latitude and longitude.
    
    Args:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location
        
    Returns:
        dict: Simulated weather data
    """
    # Simple simulation based on latitude (not scientifically accurate)
    # Equator: hot and humid, Poles: cold and dry
    abs_latitude = abs(latitude)
    
    # Temperature decreases from equator to poles (roughly)
    base_temp = 30 - (abs_latitude / 90) * 40
    
    # Humidity is higher near equator and coastlines (simplified)
    base_humidity = 80 - (abs_latitude / 90) * 40
    
    # Rainfall varies (simplified model)
    base_rainfall = 200 - (abs_latitude / 90) * 150
    
    # Add some randomness
    import random
    random.seed(int(latitude * 100 + longitude * 100))
    
    temperature = base_temp + random.uniform(-5, 5)
    humidity = min(100, max(10, base_humidity + random.uniform(-10, 10)))
    rainfall = max(0, base_rainfall + random.uniform(-50, 50))
    
    # Ensure values are in reasonable ranges
    temperature = round(max(-30, min(50, temperature)), 2)
    humidity = round(humidity, 2)
    rainfall = round(rainfall, 2)
    
    return {
        'temperature': temperature,
        'humidity': humidity,
        'rainfall': rainfall
    }


def get_soil_data(latitude, longitude, api_key=None):
    """Fetch soil data for a location.
    
    Note: This would normally use a soil data API, but for demonstration
    purposes, we'll simulate the data.
    
    Args:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location
        api_key (str, optional): API key for soil data service. Defaults to None.
        
    Returns:
        dict: Soil data including N, P, K, and pH
    """
    # In a real application, this would call an actual soil data API
    # For demonstration, we'll simulate the data
    
    # Simple simulation based on latitude and longitude
    import random
    random.seed(int(latitude * 100 + longitude * 100) + 42)
    
    # Generate simulated soil data
    soil_data = {
        'N': random.randint(20, 100),  # Nitrogen (mg/kg)
        'P': random.randint(15, 60),   # Phosphorus (mg/kg)
        'K': random.randint(15, 50),   # Potassium (mg/kg)
        'ph': round(random.uniform(5.5, 7.5), 1)  # pH value
    }
    
    return soil_data


def get_location_info(address):
    """Get latitude and longitude from an address using geocoding.
    
    Args:
        address (str): Address or location name
        
    Returns:
        dict: Location information including latitude and longitude
    """
    # This would normally use a geocoding API like Google Maps or Nominatim
    # For demonstration, we'll use some predefined locations
    
    # Sample locations (lowercase for case-insensitive matching)
    locations = {
        "new delhi": {"latitude": 28.6139, "longitude": 77.2090, "region": "Northern India"},
        "mumbai": {"latitude": 19.0760, "longitude": 72.8777, "region": "Western India"},
        "bangalore": {"latitude": 12.9716, "longitude": 77.5946, "region": "Southern India"},
        "kolkata": {"latitude": 22.5726, "longitude": 88.3639, "region": "Eastern India"},
        "chennai": {"latitude": 13.0827, "longitude": 80.2707, "region": "Southern India"},
        "hyderabad": {"latitude": 17.3850, "longitude": 78.4867, "region": "Southern India"},
        "pune": {"latitude": 18.5204, "longitude": 73.8567, "region": "Western India"},
        "ahmedabad": {"latitude": 23.0225, "longitude": 72.5714, "region": "Western India"},
        "jaipur": {"latitude": 26.9124, "longitude": 75.7873, "region": "Northern India"},
        "lucknow": {"latitude": 26.8467, "longitude": 80.9462, "region": "Northern India"},
    }
    
    # Try to match the address to a known location
    address_lower = address.lower()
    for name, location in locations.items():
        if name in address_lower or address_lower in name:
            return location
    
    # If no match found, return a default location (New Delhi)
    print(f"Location '{address}' not found in database. Using default location (New Delhi).")
    return locations["new delhi"]


def get_complete_data_for_prediction(location_name):
    """Get complete data needed for crop prediction based on location name.
    
    Args:
        location_name (str): Name of the location or address
        
    Returns:
        tuple: Location info, weather data, and soil data
    """
    # Get location coordinates
    location = get_location_info(location_name)
    
    # Get weather data for the location
    weather = get_weather_data(location["latitude"], location["longitude"])
    
    # Get soil data for the location
    soil = get_soil_data(location["latitude"], location["longitude"])
    
    return location, weather, soil


# Example usage
def main():
    """Demonstrate the weather utility functions."""
    # Example location
    location_name = "New Delhi"
    
    print(f"Fetching data for {location_name}...\n")
    
    # Get location information
    location = get_location_info(location_name)
    print("Location Information:")
    print(json.dumps(location, indent=2))
    
    # Get weather data
    weather = get_weather_data(location["latitude"], location["longitude"])
    print("\nWeather Data:")
    print(json.dumps(weather, indent=2))
    
    # Get soil data
    soil = get_soil_data(location["latitude"], location["longitude"])
    print("\nSoil Data:")
    print(json.dumps(soil, indent=2))
    
    print("\nComplete data for crop prediction:")
    location, weather, soil = get_complete_data_for_prediction(location_name)
    
    # Format the data for prediction
    prediction_data = {
        "location": location,
        "weather": weather,
        "soil": soil
    }
    
    print(json.dumps(prediction_data, indent=2))
    
    print("\nTo use this data with the crop prediction system:")
    print("from crop_prediction import CropPredictionSystem")
    print("from weather_util import get_complete_data_for_prediction")
    print("")
    print("# Get data for a location")
    print("location, weather, soil = get_complete_data_for_prediction('New Delhi')")
    print("")
    print("# Initialize and load the crop prediction model")
    print("crop_system = CropPredictionSystem()")
    print("crop_system.load_model('path/to/model.joblib')")
    print("")
    print("# Make prediction")
    print("prediction = crop_system.predict_from_location_weather(location, weather, soil)")
    print("print(prediction['predicted_crop'])")


if __name__ == "__main__":
    main()