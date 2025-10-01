import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Read CSV file from raw GitHub link
data_url = "https://raw.githubusercontent.com/anukalpr/California-House-Price-Prediction-ML-project/main/california.csv"
data = pd.read_csv(data_url)

# Convert categorical data to numerical data
le = LabelEncoder()
data['ocean_proximity'] = le.fit_transform(data['ocean_proximity'])

# Split the dataset
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
train_features = train_set.drop("median_house_value", axis=1)
train_label = train_set["median_house_value"].copy()
test_features = test_set.drop("median_house_value", axis=1)
test_label = test_set["median_house_value"].copy()

# Pipeline
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])
train_features = my_pipeline.fit_transform(train_features)

# Model
model = DecisionTreeRegressor()
model.fit(train_features, train_label)

# Streamlit UI
st.title("California Housing Society")
st.header("Predicting California House Prices")
st.write("Shape of dataset", data.shape)
st.write("Shape of training set", train_set.shape)
st.write("Shape of testing set", test_set.shape)
st.write("Shape of training features", train_features.shape)
st.write("Shape of training_label", train_label.shape)

Menu = st.sidebar.radio("Menu", ["Home", "Prediction Price", "Evaluation of Model"])

if Menu == "Home":
    st.image("https://raw.githubusercontent.com/anukalpr/California-House-Price-Prediction-ML-project/main/california_image.jpg", use_column_width=True)
    st.write("Welcome to the California House Price Prediction App!")

elif Menu == "Prediction Price":
    st.subheader("Predict House Price")
    # Input sliders for features (example for few features)
    longitude = st.slider("Longitude", float(data.longitude.min()), float(data.longitude.max()))
    latitude = st.slider("Latitude", float(data.latitude.min()), float(data.latitude.max()))
    housing_median_age = st.slider("Housing Median Age", int(data.housing_median_age.min()), int(data.housing_median_age.max()))
    
    # Create input DataFrame
    input_df = pd.DataFrame([[longitude, latitude, housing_median_age]], columns=["longitude", "latitude", "housing_median_age"])
    input_transformed = my_pipeline.transform(input_df)
    prediction = model.predict(input_transformed)
    st.write("Predicted House Price: ", prediction[0])

elif Menu == "Evaluation of Model":
    st.subheader("Model Evaluation")
    test_features_transformed = my_pipeline.transform(test_features)
    predictions = model.predict(test_features_transformed)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(test_label, predictions)
    rmse = np.sqrt(mse)
    st.write("Root Mean Squared Error (RMSE) on Test Set:", rmse)
