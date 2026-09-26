# Sydney Housing Price Prediction

This project develops and evaluates machine learning models for predicting residential property prices in selected Sydney suburbs.

## Project Overview

The project uses housing data collected from three Sydney suburbs:

- Abbotsbury
- Acacia Gardens
- Airds

The dataset contains 105 property listings and includes features such as:

- Bedrooms
- Bathrooms
- Car spaces
- Land size
- Total rooms
- Property type
- Suburb
- Sale year
- Sale month
- Days since first sale

## Machine Learning Models

The following regression models were evaluated:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression

Linear Regression achieved the strongest overall performance and was selected as the final prediction model.

## Model Performance

Linear Regression achieved approximately:

- Test R²: 0.848
- Cross-validation R²: 0.810
- Mean Absolute Error: approximately $132,800

Random Forest achieved a test R² of approximately 0.725.

Decision Tree achieved a test R² of approximately 0.172.

## Application

A Gradio interface was created to allow users to enter property characteristics and generate an estimated house price.

The application uses the saved machine learning model and preprocessing files included in this repository.

## Repository Files

- `housing-syd.ipynb` – complete data analysis and machine learning notebook
- `housing_data.xlsx` – housing dataset
- `app.py` – Gradio application
- `linear_regression_model.pkl` – trained Linear Regression model
- `model_columns.pkl` – saved model feature columns
- `first_sale_date.pkl` – saved reference date used during preprocessing
- `requirements.txt` – Python package requirements

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
