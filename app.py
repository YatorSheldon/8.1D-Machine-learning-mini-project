import gradio as gr
import pandas as pd
import joblib

# Load trained model and supporting files
linear_model = joblib.load("linear_regression_model.pkl")
model_columns = joblib.load("model_columns.pkl")
first_sale_date = joblib.load("first_sale_date.pkl")


def predict_price(
    suburb,
    property_type,
    bedrooms,
    bathrooms,
    car_spaces,
    land_size,
    sale_date
):
    # Convert sale date to pandas datetime
    sale_date = pd.Timestamp(sale_date)

    # Create derived features
    total_rooms = bedrooms + bathrooms
    sale_year = sale_date.year
    sale_month = sale_date.month
    days_since_first_sale = (sale_date - first_sale_date).days

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

    # Convert input into DataFrame in correct feature order
    input_df = pd.DataFrame(
        [input_data],
        columns=model_columns
    )

    # Generate prediction
    prediction = linear_model.predict(input_df)[0]

    return f"${prediction:,.0f}"


app = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Dropdown(
            ["Abbotsbury", "Acacia Gardens", "Airds"],
            label="Suburb"
        ),
        gr.Dropdown(
            ["House", "Townhouse", "Duplex"],
            label="Property Type"
        ),
        gr.Number(
            label="Bedrooms",
            value=4
        ),
        gr.Number(
            label="Bathrooms",
            value=2
        ),
        gr.Number(
            label="Car Spaces",
            value=2
        ),
        gr.Number(
            label="Land Size (m²)",
            value=500
        ),
        gr.Textbox(
            label="Expected Sale Date",
            value="2026-09-18"
        )
    ],
    outputs=gr.Textbox(
        label="Estimated Sale Price"
    ),
    title="Sydney Housing Price Prediction",
    description=(
        "Enter the property details below to estimate "
        "the expected sale price."
    )
)


if __name__ == "__main__":
    app.launch()
