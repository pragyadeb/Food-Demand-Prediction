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
# SESSION NAVIGATION
# =========================
if "page" not in st.session_state:
    st.session_state.page = "📊 Dashboard"

page = st.sidebar.radio(
    "📌 Navigation",
    ["📊 Dashboard", "🔮 Prediction", "📈 Analytics", "🏪 Brand Signup"],
    index=["📊 Dashboard", "🔮 Prediction", "📈 Analytics", "🏪 Brand Signup"].index(st.session_state.page)
)

st.session_state.page = page

# =========================
# THEME
# =========================
theme = st.sidebar.radio("🌗 Theme", ["Dark", "Light"])

if theme == "Dark":
    bg_color = "#0B0F19"
    text_color = "#E6EAF2"
    card_color = "#111827"
else:
    bg_color = "#FFFFFF"
    text_color = "#111111"
    card_color = "#F3F4F6"

# =========================
# CSS
# =========================
st.markdown(f"""
<style>
.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}
.card {{
    background-color: {card_color};
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    data = pd.read_csv("train.csv")
    data = data.sort_values(by="week")

    data["price_diff"] = data["base_price"] - data["checkout_price"]

    city_map = {
        10:"Delhi",11:"Mumbai",12:"Bangalore",13:"Hyderabad",
        14:"Chennai",15:"Kolkata",16:"Pune",17:"Ahmedabad",
        18:"Jaipur",19:"Lucknow",20:"Chandigarh",21:"Indore"
    }

    data["city"] = data["center_id"].map(city_map)

    np.random.seed(42)
    data["food_category"] = np.random.choice(
        ["Veg","Non-Veg","Vegan"], size=len(data)
    )

    return data

data = load_data()

# =========================
# FILTERS
# =========================
food_category = st.sidebar.selectbox(
    "🍽 Food Category",
    ["All","Veg","Non-Veg","Vegan"]
)

city = st.sidebar.selectbox(
    "🏙 City",
    ["All"] + sorted(data["city"].dropna().unique())
)

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
@st.cache_resource
def load_ml():
    model = load_model("gru_model.h5", compile=False)
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_ml()

# =========================
# DASHBOARD
# =========================
if page == "📊 Dashboard":

    st.title("📊 Food Demand Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f'<div class="card"><h4>Total Orders</h4><h2>{int(filtered_data["num_orders"].sum())}</h2></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="card"><h4>Average Orders</h4><h2>{int(filtered_data["num_orders"].mean())}</h2></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="card"><h4>Max Orders</h4><h2>{int(filtered_data["num_orders"].max())}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Demand Trend")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=filtered_data["num_orders"],
            mode="lines",
            line=dict(color="#6366F1", width=3)
        ))

        fig.update_layout(
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏙 Demand by City")

        city_orders = filtered_data.groupby("city")["num_orders"].sum()

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=city_orders.index,
            y=city_orders.values,
            marker=dict(color="#8B5CF6")
        ))

        fig2.update_layout(
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color)
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🍽 Category Distribution")

    cat_orders = filtered_data.groupby("food_category")["num_orders"].sum()

    pie = go.Figure(data=[go.Pie(
        labels=cat_orders.index,
        values=cat_orders.values,
        hole=0.4
    )])

    pie.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color)
    )

    st.plotly_chart(pie, use_container_width=True)

# =========================
# PREDICTION
# =========================
elif page == "🔮 Prediction":

    st.title("🔮 Demand Prediction")

    price = st.slider("Checkout Price", 100, 500, 200)
    base = st.slider("Base Price", 100, 600, 250)

    if st.button("🚀 Predict"):

        price_diff = base - price

        inp = np.array([[100, price, price_diff]])
        inp = scaler.transform(inp)
        inp = inp.reshape(1,1,3)

        pred = model.predict(inp, verbose=0)
        prediction = int(pred[0][0]*1000)

        st.success(f"📊 Predicted Orders: {prediction}")

# =========================
# ANALYTICS
# =========================
elif page == "📈 Analytics":

    st.title("📈 Analytics")

    sample = filtered_data[["num_orders","checkout_price","price_diff"]].head(100)

    scaled = scaler.transform(sample)
    X_test = scaled.reshape(len(scaled),1,3)

    preds = model.predict(X_test, verbose=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=sample["num_orders"], name="Actual"))
    fig.add_trace(go.Scatter(y=preds.flatten()*1000, name="Predicted"))

    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color)
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# SIGNUP + REDIRECT
# =========================
elif page == "🏪 Brand Signup":

    st.title("🏪 Brand Signup")

    with st.form("signup"):
        brand = st.text_input("Brand Name")
        email = st.text_input("Email")
        contact = st.text_input("Contact")

        submit = st.form_submit_button("Submit")

        if submit:
            new_entry = pd.DataFrame({
                "Brand":[brand],
                "Email":[email],
                "Contact":[contact]
            })

            try:
                existing = pd.read_csv("brand_signups.csv")
                updated = pd.concat([existing,new_entry],ignore_index=True)
            except:
                updated = new_entry

            updated.to_csv("brand_signups.csv", index=False)

            st.success("✅ Signup Successful! Redirecting...")

            # 🔥 REDIRECT
            st.session_state.page = "📊 Dashboard"
            st.rerun()