import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV file
data = pd.read_csv("california.csv")

# Convert categorical data to numerical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data.ocean_proximity = le.fit_transform(data.ocean_proximity)

# Split the dataset into training data and testing data
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# Splitting into features and labels
train_features = train_set.drop("median_house_value", axis=1)
train_label = train_set["median_house_value"].copy()
test_features = test_set.drop("median_house_value", axis=1)
test_label = test_set["median_house_value"].copy()

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('std_scalor', StandardScaler())])
train_features = my_pipeline.fit_transform(train_features)

# Selecting desired model for California house price for training data set
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(train_features, train_label)
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(train_features, train_label)


# Title of this Project
st.title("California Housing Society")

# Image
st.image("Streamlitlogo.png", width=200)

# Header
st.header("Predicting California House Prices")

# Title
st.write("Shape of dataset", data.shape, "|", "Shape of training set", train_set.shape, "|", "Shape of Testing Set", test_set.shape)
st.write("Shape of training features", train_features.shape, "|", "Shape of training_label", train_label.shape)

Menu = st.sidebar.radio("Menu", ["Home", "Prediction Price", "Evaluation of Model"])
if Menu == "Home":
    st.image("calihouse.jpeg", width=400)
    st.header("Tabular Data of California Housing")
    if st.checkbox("Tabular Data"):
        st.table(data.head(100))
    
    st.header("Statistical summary of a DataFrame")
    if st.checkbox("Statistics"):
        st.table(data.describe())

    st.header("Correlation Coefficient")
    if st.checkbox("Correlation"):
        corr_matrix = data.corr()
        correlation = corr_matrix["median_house_value"].sort_values(ascending=False)
        st.write(correlation)

if Menu == "Prediction Price":
    st.header("Prediction of California House Prices")
    input_features = []
    for feature in train_set.drop("median_house_value", axis=1).columns:
        value = st.number_input(f"Enter value for {feature}", value=0.0)
        input_features.append(value)
    input_data = np.array(input_features).reshape(1, -1)
    input_data = my_pipeline.transform(input_data)
    prediction = model.predict(input_data)
    st.write(f"Predicted House Value: ${prediction[0]:,.2f}")

if Menu == "Evaluation of Model":
    st.header("Evaluation of Model")
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error
    # Predictions
    prediction = model.predict(train_features)

    # Root Mean Squared Error (RMSE)
    mse = mean_squared_error(train_label, prediction)
    rmse = np.sqrt(mse)
    st.write(f"Root Mean Squared Error: {rmse}")

    # R-squared (R2):- Values range from 0 to 1, where 1 means perfect prediction.
    r2 = r2_score(train_label, prediction)
    st.write(f"R-squared: {r2}")

