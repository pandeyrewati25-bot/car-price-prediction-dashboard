import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

# ---------------- PAGE ---------------- #
st.set_page_config(page_title="Car Price AI", layout="wide")

# ---------------- UI ---------------- #
st.markdown("""
<style>

.stApp {
    background-image: url("https://images.unsplash.com/photo-1503376780353-7e6692767b70");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.85);
    z-index: 0;
}

.block-container {
    position: relative;
    z-index: 1;
}

.glass {
    background: rgba(255,255,255,0.10);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    color: white;
}

h1, h2, h3 {
    color: white !important;
}

label {
    color: #e6e6e6 !important;
}

.stButton>button {
    background: linear-gradient(90deg, #ff3c3c, #ff914d);
    color: white;
    width: 100%;
    height: 3em;
    font-size: 18px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADING (CHANGED) ---------------- #
st.title("🚗 Car Price Prediction")
st.markdown("### AI-based Machine Learning Model for Predicting Car Prices")

# ---------------- DATA ---------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("carprice.csv")
    df = df.dropna().drop_duplicates()

    le_fuel = LabelEncoder()
    le_loc = LabelEncoder()
    le_engine = LabelEncoder()

    df['fuel-type'] = le_fuel.fit_transform(df['fuel-type'])
    df['engine-location'] = le_loc.fit_transform(df['engine-location'])
    df['engine-type'] = le_engine.fit_transform(df['engine-type'])

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    return df, le_fuel, le_loc, le_engine

df, le_fuel, le_loc, le_engine = load_data()

# ---------------- FEATURES ---------------- #
FEATURES = df.drop("price", axis=1).columns.tolist()

X = df[FEATURES]
y = df["price"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ---------------- INPUT ---------------- #
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("📊 Car Details Input")

    fuel_type = st.selectbox("Fuel Type", ["gas", "diesel"])
    engine_location = st.selectbox("Engine Location", ["front", "rear"])
    engine_type = st.selectbox("Engine Type", sorted(df['engine-type'].unique()))

    horsepower = st.slider("Horsepower", 50, 300, 100)
    peak_rpm = st.slider("Peak RPM", 1000, 8000, 5000)
    city_mpg = st.slider("City MPG", 5, 60, 25)
    highway_mpg = st.slider("Highway MPG", 5, 70, 30)

    st.markdown("</div>", unsafe_allow_html=True)

# encode
fuel_type = le_fuel.transform([fuel_type])[0]
engine_location = le_loc.transform([engine_location])[0]

# ---------------- PREDICTION ---------------- #
with col2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("💰 Prediction Result")

    if st.button("Predict Price 🚀"):

        input_dict = {
            "fuel-type": fuel_type,
            "engine-location": engine_location,
            "engine-type": engine_type,
            "horsepower": horsepower,
            "peak-rpm": peak_rpm,
            "city-mpg": city_mpg,
            "highway-mpg": highway_mpg
        }

        input_data = pd.DataFrame([input_dict])

        for col in FEATURES:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[FEATURES]

        prediction = model.predict(input_data)[0]

        st.success(f"💰 Estimated Price: ${prediction:,.2f}")
        st.info("✔ AI Model: Random Forest Regressor")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- GRAPH (JUPYTER STYLE) ---------------- #
st.markdown("## 📊 Feature Importance (Jupyter Style)")

importances = model.feature_importances_

fig, ax = plt.subplots(figsize=(10,5))

ax.barh(FEATURES, importances, color="royalblue")

ax.set_title("Feature Importance")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")

plt.gca().invert_yaxis()

st.pyplot(fig)

# ---------------- FOOTER ---------------- #
st.markdown("""
---
<center style='color:#ddd'>
🚗 Built with Streamlit | Car Price Prediction System
</center>
""", unsafe_allow_html=True)