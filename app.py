import streamlit as st
import pandas as pd
import joblib
from Ames_Function_Transformation import FeatureEngineer

st.set_page_config(page_title="Ames Housing Predictor", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load('ames_housing_model.pkl')


model = load_model()

st.title("Ames Housing Price Predictor")
st.markdown("Estimate house prices based on property features using a trained Machine Learning pipeline.")
st.markdown("---")

with st.form("prediction_form"):
    st.subheader("Property Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Dimensions & Area**")
        gr_liv_area = st.number_input("Above Grade Living Area (sqft)", 300, 5000, 1500, 50)
        lot_area = st.number_input("Lot Area (sqft)", 1000, 50000, 10000, 100)
        total_bsmt_sf = st.number_input("Total Basement Area (sqft)", 0, 3000, 1000, 50)

    with col2:
        st.markdown("**Age & Quality**")
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
        year_built = st.number_input("Year Built", 1870, 2010, 2000, 1)
        year_remod = st.number_input("Year Remodeled", 1950, 2010, 2000, 1)

    with col3:
        st.markdown("**Rooms & Amenities**")
        tot_rooms = st.number_input("Total Rooms Above Grade", 2, 15, 6, 1)
        garage_cars = st.selectbox("Garage Capacity (Cars)", [0, 1, 2, 3, 4, 5], index=2)
        neighborhood = st.selectbox("Neighborhood",
                                    ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes',
                                     'OldTown', 'BrkSide', 'Sawyer', 'Other'])

    st.markdown("---")
    submit_button = st.form_submit_button("Predict Price", use_container_width=True)

if submit_button:
    with st.spinner('Processing data and generating prediction...'):
        input_df = pd.DataFrame([{
            'Gr Liv Area': gr_liv_area,
            'Lot Area': lot_area,
            'Overall Qual': overall_qual,
            'Total Bsmt SF': total_bsmt_sf,
            'Year Built': year_built,
            'Year Remod/Add': year_remod,
            'Garage Cars': garage_cars,
            'TotRms AbvGrd': tot_rooms,
            'Neighborhood': neighborhood,
            'Garage Area': garage_cars * 250.0,
            'Lot Frontage': 70.0,
            '1st Flr SF': gr_liv_area * 0.6,
            'Wood Deck SF': 0.0,
            'Open Porch SF': 0.0,
            'BsmtFin SF 1': 0.0,
            'Mas Vnr Area': 0.0,
            'MS Zoning': 'RL',
            'House Style': '1Story',
            'Bldg Type': '1Fam',
            'Central Air': 'Y',
            'Exter Qual': 'TA',
            'Kitchen Qual': 'TA',
            'Bsmt Qual': 'TA',
            'Full Bath': 2,
            'Fireplaces': 0
        }])

        try:
            prediction = model.predict(input_df)[0]
            st.success("Prediction Successful!")
            st.metric(label="Estimated Fair Price (USD)", value=f"${prediction:,.0f}")
        except Exception as e:
            st.error("An error occurred during prediction.")
            st.expander("Error Details").write(str(e))