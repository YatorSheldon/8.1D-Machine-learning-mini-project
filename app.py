
import streamlit as st
import pandas as pd
import joblib

# Load trained model and supporting files
model = joblib.load("linear_regression_model.pkl")
model_columns = joblib.load("model_columns.pkl")
first_sale_date = joblib.load("first_sale_date.pkl")

st.title("Sydney Housing Price Prediction")
st.write(
    "Enter the property details below to estimate the expected sale price."
)

# User inputs
suburb = st.selectbox(
    "Suburb",
    ["Abbotsbury", "Acacia Gardens", "Airds"]
)

property_type = st.selectbox(
    "Property Type",
    ["House", "Townhouse", "Duplex"]
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=4
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

car_spaces = st.number_input(
    "Car Spaces",
    min_value=0,
    max_value=20,
    value=2
)

land_size = st.number_input(
    "Land Size (m²)",
    min_value=0.0,
    value=500.0
)

sale_date = st.date_input(
    "Expected Sale Date"
)

if st.button("Predict Sale Price"):

    sale_date = pd.Timestamp(sale_date)

    total_rooms = bedrooms + bathrooms
    sale_year = sale_date.year
    sale_month = sale_date.month

    days_since_first_sale = (
        sale_date - first_sale_date
    ).days

    # Start with all model features set to zero
    input_data = {
        column: 0
        for column in model_columns
    }

    # Numeric features
    input_data["Bedrooms"] = bedrooms
    input_data["Bathrooms"] = bathrooms
    input_data["Car_Spaces"] = car_spaces
    input_data["Land_Size_m2"] = land_size
    input_data["Total_Rooms"] = total_rooms
    input_data["Sale_Year"] = sale_year
    input_data["Sale_Month"] = sale_month
    input_data["Days_Since_First_Sale"] = days_since_first_sale

    # Suburb encoding
    if suburb == "Acacia Gardens":
        input_data["Suburb_Acacia Gardens"] = 1

    elif suburb == "Airds":
        input_data["Suburb_Airds"] = 1

    # Abbotsbury is the reference category

    # Property type encoding
    if property_type == "House":
        input_data["Property_Type_House"] = 1

    elif property_type == "Townhouse":
        input_data["Property_Type_Townhouse"] = 1

    # Duplex is the reference category

    input_df = pd.DataFrame(
        [input_data],
        columns=model_columns
    )

    prediction = model.predict(input_df)[0]

    st.success(
        f"Estimated Sale Price: ${prediction:,.0f}"
    )

    st.caption(
        "This prediction is an estimate based on the available dataset and should not be treated as a professional property valuation."
    )
