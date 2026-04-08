# =========================
# IMPORT LIBRARIES
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model
import joblib

# =========================
# LOAD MODEL + SCALER
# =========================
model = load_model("gru_model_food_type.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Food Demand Dashboard", layout="wide")

# =========================
# THEME TOGGLE
# =========================
theme = st.sidebar.radio("🌗 Theme", ["Dark", "Light"])

if theme == "Dark":
    bg_color = "#0E1117"
    text_color = "white"
    card_color = "#1c1f26"
else:
    bg_color = "white"
    text_color = "black"
    card_color = "#f0f2f6"

chart_height = 520
chart_margin = dict(l=60, r=60, t=70, b=60)

# =========================
# TITLE
# =========================
st.markdown(
    f"<h1 style='color:{text_color};'>🍽️ Food Demand Dashboard</h1>",
    unsafe_allow_html=True
)

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("train.csv")

np.random.seed(42)

if "city" not in data.columns:
    data["city"] = np.random.choice(["Delhi", "Mumbai", "Bangalore"], len(data))

if "food_type" not in data.columns:
    data["food_type"] = np.random.choice(["Veg", "Non-Veg", "Vegan"], len(data))

# =========================
# SESSION STATE
# =========================
if "data_store" not in st.session_state:
    st.session_state.data_store = data.copy()

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("🔍 Filters")

city = st.sidebar.selectbox(
    "City",
    st.session_state.data_store["city"].unique()
)

food = st.sidebar.selectbox(
    "Food Type",
    ["All"] + list(st.session_state.data_store["food_type"].unique())
)

# =========================
# FILTER DATA
# =========================
if food == "All":
    filtered = st.session_state.data_store[
        st.session_state.data_store["city"] == city
    ]
else:
    filtered = st.session_state.data_store[
        (st.session_state.data_store["city"] == city) &
        (st.session_state.data_store["food_type"] == food)
    ]

filtered = filtered.copy()
filtered["price_diff"] = filtered["base_price"] - filtered["checkout_price"]

# =========================
# KPI CARDS
# =========================
col1, col2, col3 = st.columns(3)

def kpi(title, value):
    st.markdown(f"""
    <div style="
        background:{card_color};
        padding:20px;
        border-radius:10px;
        text-align:center;
        color:{text_color}">
        <h4>{title}</h4>
        <h2>{value}</h2>
    </div>
    """, unsafe_allow_html=True)

with col1:
    kpi("📦 Total Orders", int(filtered["num_orders"].sum()))

with col2:
    kpi("📈 Avg Orders", int(filtered["num_orders"].mean()))

with col3:
    kpi("💰 Avg Price", int(filtered["checkout_price"].mean()))

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📉 Line Chart", "📊 Bar Chart", "🥧 Pie Chart", "📄 Data"]
)

# =========================
# LINE CHART
# =========================
with tab1:

    plot_data = filtered.copy().reset_index()

    # Reduce noise
    plot_data["group"] = plot_data.index // 1500
    clean_data = plot_data.groupby("group")["num_orders"].mean().reset_index()

    # Smooth actual data
    clean_data["actual_smooth"] = clean_data["num_orders"].rolling(
        window=3, center=True, min_periods=1
    ).mean()

    # Predicted line (shifted comparison)
    clean_data["predicted"] = clean_data["actual_smooth"].shift(-1)
    clean_data["predicted"] = clean_data["predicted"].fillna(clean_data["actual_smooth"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=clean_data["group"] * 1500,
        y=clean_data["actual_smooth"],
        mode="lines",
        name="Actual Orders",
        line=dict(color="#F5F5F5", width=2, shape="spline")
    ))

    fig.add_trace(go.Scatter(
        x=clean_data["group"] * 1500,
        y=clean_data["predicted"],
        mode="lines",
        name="Predicted Orders",
        line=dict(color="#F6E27F", width=2, shape="spline")
    ))

    fig.update_layout(
        title=f"📈 Orders Over Time — {city} ({food})",
        xaxis_title="Time",
        yaxis_title="Orders",
        height=chart_height,
        margin=chart_margin,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.18)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.18)"),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# BAR CHART
# =========================
with tab2:

    bar_fig = px.bar(
        filtered.head(20),
        x="meal_id" if "meal_id" in filtered.columns else filtered.index,
        y="num_orders",
        color="food_type",
        title="Orders by Meal"
    )

    bar_fig.update_layout(
        height=chart_height,
        margin=chart_margin,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color)
    )

    st.plotly_chart(bar_fig, use_container_width=True)

# =========================
# PIE CHART
# =========================
with tab3:

    pie_data = filtered["food_type"].value_counts().reset_index()
    pie_data.columns = ["Food Type", "Count"]

    pie_fig = px.pie(
        pie_data,
        names="Food Type",
        values="Count",
        title="Food Distribution"
    )

    pie_fig.update_layout(
        height=chart_height,
        margin=chart_margin,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color)
    )

    st.plotly_chart(pie_fig, use_container_width=True)

# =========================
# DATA TABLE
# =========================
with tab4:
    st.dataframe(filtered.tail(50))

# =========================
# PREDICTION INPUT
# =========================
st.sidebar.header("🤖 Predict New Demand")

checkout_price = st.sidebar.slider("Checkout Price", 50, 2000, 500)
base_price = st.sidebar.slider("Base Price", 50, 2000, 700)

price_diff = base_price - checkout_price

if st.sidebar.button("Predict"):

    food_type_for_pred = food if food != "All" else "Veg"

    inp = np.array([[0, checkout_price, base_price, price_diff, 0]])

    inp_scaled = scaler.transform(inp)
    inp_scaled = inp_scaled.reshape((1, 1, 5))

    pred = model.predict(inp_scaled)

    temp = np.zeros((1, 5))
    temp[0, 0] = pred[0][0]

    result = scaler.inverse_transform(temp)[0][0]

    st.success(f"📊 Predicted Orders ({food_type_for_pred}): {int(result)}")

    new_row = {
        "num_orders": int(result),
        "checkout_price": checkout_price,
        "base_price": base_price,
        "meal_id": np.random.randint(1000, 2000),
        "city": city,
        "food_type": food_type_for_pred
    }

    st.session_state.data_store = pd.concat(
        [st.session_state.data_store, pd.DataFrame([new_row])],
        ignore_index=True
    )

    st.rerun()

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 Built with Streamlit + GRU Model")