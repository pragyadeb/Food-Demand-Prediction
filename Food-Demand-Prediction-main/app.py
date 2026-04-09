import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Food Demand Dashboard", layout="wide")

# =========================
# THEME
# =========================
theme = st.sidebar.radio("🌗 Theme", ["Dark", "Light"])

if theme == "Dark":
    bg_color = "#0B0F19"
    text_color = "#E6EAF2"
else:
    bg_color = "#FFFFFF"
    text_color = "#111111"

st.markdown(f"""
<style>
.stApp {{
background-color:{bg_color};
color:{text_color};
}}
</style>
""", unsafe_allow_html=True)

# =========================
# NAVIGATION
# =========================
page = st.sidebar.selectbox(
"📌 Navigate",
["Dashboard","Prediction","Analytics","Brand Signup"]
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():

    data = pd.read_csv("train.csv")

    data = data.sort_values(by="week")

    data["price_diff"] = data["base_price"] - data["checkout_price"]

    # Map center_id → Indian cities
    city_map = {
    10:"Delhi",
    11:"Mumbai",
    12:"Bangalore",
    13:"Hyderabad",
    14:"Chennai",
    15:"Kolkata",
    16:"Pune",
    17:"Ahmedabad",
    18:"Jaipur",
    19:"Lucknow",
    20:"Chandigarh",
    21:"Indore"
    }

    data["city"] = data["center_id"].map(city_map)

    # Food categories
    np.random.seed(42)

    data["food_category"] = np.random.choice(
    ["Veg","Non-Veg","Vegan"],
    size=len(data)
    )

    return data

data = load_data()

# =========================
# SIDEBAR FILTERS
# =========================
food_category = st.sidebar.selectbox(
"🍽 Food Category",
["All","Veg","Non-Veg","Vegan"]
)

city = st.sidebar.selectbox(
"🏙 City",
["All"] + sorted(data["city"].dropna().unique())
)

# =========================
# FILTER DATA
# =========================
filtered_data = data.copy()

if food_category != "All":
    filtered_data = filtered_data[
    filtered_data["food_category"] == food_category
    ]

if city != "All":
    filtered_data = filtered_data[
    filtered_data["city"] == city
    ]

# =========================
# LOAD MODEL
# =========================
model = load_model("gru_model.h5",compile=False)
scaler = joblib.load("scaler.pkl")

# =========================
# DASHBOARD PAGE
# =========================
if page == "Dashboard":

    st.title("📊 Food Demand Dashboard")

    col1,col2 = st.columns(2)

    col1.metric(
    "Total Orders",
    int(filtered_data["num_orders"].sum())
    )

    col2.metric(
    "Average Orders",
    int(filtered_data["num_orders"].mean())
    )

    # DEMAND TREND
    st.subheader("📈 Demand Trend")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
    y=filtered_data["num_orders"],
    mode="lines",
    name="Orders"
    ))

    fig.update_layout(
    height=300,
    plot_bgcolor=bg_color,
    paper_bgcolor=bg_color,
    font=dict(color=text_color)
    )

    st.plotly_chart(fig,use_container_width=True)

    # DEMAND BY CITY
    st.subheader("🏙 Demand by City")

    city_orders = filtered_data.groupby("city")["num_orders"].sum()

    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
    x=city_orders.index,
    y=city_orders.values
    ))

    fig2.update_layout(
    plot_bgcolor=bg_color,
    paper_bgcolor=bg_color,
    font=dict(color=text_color)
    )

    st.plotly_chart(fig2,use_container_width=True)

    # CATEGORY PIE
    st.subheader("🍽 Category Distribution")

    cat_orders = filtered_data.groupby("food_category")["num_orders"].sum()

    pie = go.Figure(data=[go.Pie(
    labels=cat_orders.index,
    values=cat_orders.values
    )])

    pie.update_layout(
    plot_bgcolor=bg_color,
    paper_bgcolor=bg_color,
    font=dict(color=text_color)
    )

    st.plotly_chart(pie,use_container_width=True)

# =========================
# PREDICTION PAGE
# =========================
elif page == "Prediction":

    st.title("🔮 Predict Demand")

    food_type = st.selectbox(
    "Food Category",
    ["Veg","Non-Veg","Vegan"]
    )

    price = st.slider(
    "Checkout Price",
    100,500,200
    )

    base = st.slider(
    "Base Price",
    100,600,250
    )

    if st.button("Predict"):

        price_diff = base - price

        inp = np.array([[100,price,price_diff]])

        inp = scaler.transform(inp)

        inp = inp.reshape(1,1,3)

        pred = model.predict(inp,verbose=0)

        prediction = int(pred[0][0]*1000)

        st.success(f"Predicted {food_type} Orders: {prediction}")

        labels = ["Predicted","Remaining"]

        values = [prediction,max(1000-prediction,0)]

        pie = go.Figure(data=[go.Pie(labels=labels,values=values)])

        pie.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color)
        )

        st.plotly_chart(pie,use_container_width=True)

# =========================
# ANALYTICS PAGE
# =========================
elif page == "Analytics":

    st.title("📊 Analytics")

    st.subheader("Actual vs Predicted")

    sample = filtered_data[
    ["num_orders","checkout_price","price_diff"]
    ].head(100)

    scaled = scaler.transform(sample)

    X_test = scaled.reshape(len(scaled),1,3)

    preds = model.predict(X_test,verbose=0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
    y=sample["num_orders"],
    mode="lines",
    name="Actual"
    ))

    fig.add_trace(go.Scatter(
    y=preds.flatten()*1000,
    mode="lines",
    name="Predicted"
    ))

    fig.update_layout(
    height=350,
    plot_bgcolor=bg_color,
    paper_bgcolor=bg_color,
    font=dict(color=text_color)
    )

    st.plotly_chart(fig,use_container_width=True)

# =========================
# BRAND SIGNUP PAGE
# =========================
elif page == "Brand Signup":

    st.title("🏪 Brand Partnership Signup")

    st.write("Brands can register to join the food demand prediction platform.")

    with st.form("brand_signup"):

        brand = st.text_input("Brand Name")

        email = st.text_input("Business Email")

        city_choice = st.selectbox(
        "City",
        sorted(data["city"].dropna().unique())
        )

        category = st.selectbox(
        "Food Category",
        ["Veg","Non-Veg","Vegan"]
        )

        contact = st.text_input("Contact Number")

        submit = st.form_submit_button("Submit")

        if submit:

            new_entry = pd.DataFrame({
            "Brand":[brand],
            "Email":[email],
            "City":[city_choice],
            "Category":[category],
            "Contact":[contact]
            })

            try:
                existing = pd.read_csv("brand_signups.csv")
                updated = pd.concat([existing,new_entry],ignore_index=True)
            except:
                updated = new_entry

            updated.to_csv("brand_signups.csv",index=False)

            st.success("✅ Signup Successful! Our team will contact you soon.")