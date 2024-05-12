import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Set custom theme
def set_theme():
    st.markdown(
        """
        <style>
        .css-1v3fvcr {
            background-color: #f0f0f0;
        }
        .css-1aumxhk {
            color: #333333;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_theme()

# Load data from CSV
@st.cache_data
def load_data():
    data = pd.read_csv("finalTrain.csv")
    data.dropna(inplace=True)

    # Drop irrelevant columns
    data.drop(['ID', 'Delivery_person_ID', 'Order_Date', 'Time_Orderd', 'Time_Order_picked'], axis=1, inplace=True)
    
    return data

df = load_data()

# Split the data
X = df.drop('Time_taken (min)', axis=1)
y = df['Time_taken (min)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing pipeline
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append regression model to preprocessing pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Streamlit app
st.title('Delivery Time Prediction')

# Inputs for prediction
age = st.slider('Delivery Person Age', min_value=18, max_value=65, value=30)
ratings = st.slider('Delivery Person Ratings', min_value=0.0, max_value=5.0, value=4.2)
vehicle_condition = st.slider('Vehicle Condition', min_value=0, max_value=2, value=1)
multiple_deliveries = st.selectbox('Multiple Deliveries', options=[0, 1, 2, 3, 4])
latitude = st.number_input('Restaurant Latitude', format="%.6f")
longitude = st.number_input('Restaurant Longitude', format="%.6f")
delivery_latitude = st.number_input('Delivery Location Latitude', format="%.6f")
delivery_longitude = st.number_input('Delivery Location Longitude', format="%.6f")

# Select boxes for categorical data
weather = st.selectbox('Weather Conditions', options=df['Weather_conditions'].unique())
road_traffic = st.selectbox('Road Traffic Density', options=df['Road_traffic_density'].unique())
order_type = st.selectbox('Type of Order', options=df['Type_of_order'].unique())
vehicle_type = st.selectbox('Type of Vehicle', options=df['Type_of_vehicle'].unique())
festival = st.selectbox('Festival', options=df['Festival'].unique())
city = st.selectbox('City', options=df['City'].unique())

if st.button('Predict Delivery Time'):
    # Prepare input data
    input_data = pd.DataFrame({
        'Delivery_person_Age': [age],
        'Delivery_person_Ratings': [ratings],
        'Restaurant_latitude': [latitude],
        'Restaurant_longitude': [longitude],
        'Delivery_location_latitude': [delivery_latitude],
        'Delivery_location_longitude': [delivery_longitude],
        'Weather_conditions': [weather],
        'Road_traffic_density': [road_traffic],
        'Vehicle_condition': [vehicle_condition],
        'Type_of_order': [order_type],
        'Type_of_vehicle': [vehicle_type],
        'multiple_deliveries': [multiple_deliveries],
        'Festival': [festival],
        'City': [city]
    })

    # Make prediction
    prediction = model.predict(input_data)
    st.write(f'Predicted Delivery Time: {prediction[0]:.2f} minutes')

    # Show prediction visualization
    show_prediction_visualization(model, input_data)

# Function to show prediction visualization
def show_prediction_visualization(model, input_data):
    # Generate scenarios for visualization
    features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
                'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude',
                'Weather_conditions', 'Road_traffic_density', 'Vehicle_condition', 'Type_of_order',
                'Type_of_vehicle', 'multiple_deliveries', 'Festival', 'City']
    scenarios = generate_scenarios(input_data[features])

    # Predict delivery times for scenarios
    predictions = model.predict(scenarios)

    # Plot bar chart
    fig, ax = plt.subplots()
    ax.bar(range(len(predictions)), predictions, align='center')
    ax.set_xticks(range(len(predictions)))
    ax.set_xticklabels([f'Scenario {i+1}' for i in range(len(predictions))])
    ax.set_ylabel('Predicted Delivery Time (min)')
    ax.set_xlabel('Scenarios')
    ax.set_title('Predicted Delivery Time for Different Scenarios')
    st.pyplot(fig)

# Function to generate scenarios for prediction visualization
def generate_scenarios(input_data):
    scenarios = []
    for i in range(len(input_data)):
        scenario = input_data.iloc[0].copy()
        for col in input_data.columns:
            # Modify each feature slightly for different scenarios
            if pd.api.types.is_numeric_dtype(input_data[col]):
                scenario[col] += np.random.uniform(low=-1.0, high=1.0)
            else:
                scenario[col] = input_data[col].values[0]  # Keep categorical features unchanged
        scenarios.append(scenario)
    return pd.DataFrame(scenarios)

