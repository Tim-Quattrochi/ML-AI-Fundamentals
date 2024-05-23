# ML-AI-Fundamentals

## Introduction

This is a Python application that uses machine learning to predict employee attrition. It uses the Logistic Regression model from the sklearn library.

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Installation

#### For Mac:

1. Open Terminal
2. Create a new virtual environment: `python3 -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate`
4. Install the required packages: `pip install -r requirements.txt`

#### For Windows:

1. Open Command Prompt Or PowerShell
2. Create a new virtual environment: `py -m venv venv`
3. Activate the virtual environment: `.\venv\Scripts\activate`
4. Install the required packages: `pip install -r requirements.txt`

### Usage

1. Ensure that the virtual environment is activated.
2. Run the application: `python app.py`

### Notes

- The application expects a CSV file named 'WA*Fn-UseC*-HR-Employee-Attrition.csv' in the same directory as the script. Make sure this file is present before running the application.
- The application will print various outputs to the console, including the original data, the preprocessed data, the feature importance, and the performance metrics of the model.
- The application will also fine-tune the model parameters using GridSearchCV and print the performance metrics of the best model.
