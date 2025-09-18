# Crop Prediction System

A machine learning-based system for predicting suitable crops based on location, weather conditions, and soil type. This project provides a comprehensive solution for agricultural planning and decision-making.

## Features

- **Data Loading and Preprocessing**: Load and preprocess agricultural datasets with soil and weather parameters
- **Exploratory Data Analysis**: Visualize and analyze crop distribution and feature correlations
- **Machine Learning Model**: Train a Random Forest classifier to predict suitable crops
- **Prediction Functionality**: Make predictions based on environmental factors
- **Location-based Prediction**: Predict crops using location, weather, and soil data
- **Visualization**: Generate visualizations for model evaluation and feature importance
- **Reporting**: Create detailed reports with prediction results and recommendations

## Installation

1. Clone the repository or download the source code
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv agri_env
   ```
3. Activate the virtual environment:
   - Windows: `agri_env\Scripts\activate`
   - Linux/Mac: `source agri_env/bin/activate`
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

> **Note**: This project requires Python 3.12 or later. If you encounter package compatibility issues, the requirements.txt file uses flexible version specifications (>=) to allow for compatible versions.

## Usage

### Basic Usage

```python
from crop_prediction import CropPredictionSystem

# Initialize the system
crop_system = CropPredictionSystem()

# Load data
crop_system.load_data('crop_data.csv')

# Train the model
crop_system.train_model()

# Save the model
crop_system.save_model()

# Make a prediction
sample_input = {
    'N': 90, 'P': 40, 'K': 35, 
    'temperature': 25, 'humidity': 65, 
    'ph': 6.5, 'rainfall': 150
}
prediction = crop_system.predict(sample_input)
print(prediction)
```

### Demo Script

A demo script is provided to showcase the system's capabilities:

```
python demo.py
```

### Command-line Interface

The system can be used via the command-line interface:

```
# Train a model
python predict_crop.py train --data crop_data.csv

# Predict based on location
python predict_crop.py predict --location "New Delhi, India"

# Predict using manual input
python predict_crop.py manual
```

## Input Data Format

The system expects input data with the following parameters:

- **N**: Nitrogen content in soil (mg/kg)
- **P**: Phosphorus content in soil (mg/kg)
- **K**: Potassium content in soil (mg/kg)
- **temperature**: Temperature in degrees Celsius
- **humidity**: Relative humidity in percentage
- **ph**: pH value of the soil
- **rainfall**: Rainfall in mm

## Location-based Prediction

The system can predict suitable crops based on location:

```python
from crop_prediction import CropPredictionSystem
from weather_util import get_complete_data_for_prediction

# Initialize the system
crop_system = CropPredictionSystem()

# Load a trained model
crop_system.load_model('models/crop_prediction_model.joblib')

# Get data for a location
location = "Mumbai, India"
data = get_complete_data_for_prediction(location)

# Make prediction
prediction = crop_system.predict(data)
print(prediction)
```

## Model Persistence

Trained models are saved in the `models` directory with timestamps:

```python
# Save model
model_path = crop_system.save_model()

# Load model
crop_system.load_model(model_path)
```

## Report Generation

The system can generate detailed reports with prediction results:

```python
# Generate a report
report = crop_system.generate_report(prediction, 'crop_prediction_report.txt')
```

## Dataset

The included sample dataset (`crop_data.csv`) contains the following crops:

- Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas
- Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate
- Banana, Mango, Grapes, Watermelon, Muskmelon
- Apple, Orange, Papaya, Coconut, Cotton
- Jute, Coffee

## Requirements

- Python 3.12+
- NumPy (>=2.0.0)
- Pandas (>=2.0.2)
- Scikit-learn (>=1.3.0)
- Matplotlib (>=3.7.2)
- Seaborn (>=0.12.2)
- Joblib (>=1.3.1)
- Requests (>=2.32.0)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses simulated weather data when API keys are not provided
- The crop recommendation model is based on soil and weather parameters