import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    SimpleRNN,
    Dropout,
    Dense
)

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Electricity Consumption Forecasting",
    page_icon="⚡",
    layout="wide"
)

# =====================================
# BASE DIR
# =====================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# =====================================
# REBUILD MODEL
# =====================================

@st.cache_resource
def load_model():

    model = Sequential()

    model.add(
        SimpleRNN(
            64,
            return_sequences=True,
            input_shape=(24,1)
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        SimpleRNN(32)
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        Dense(1)
    )

    model.load_weights(
        os.path.join(
            BASE_DIR,
            "energy_weights.weights.h5"
        )
    )

    scaler = joblib.load(
        os.path.join(
            BASE_DIR,
            "scaler.pkl"
        )
    )

    return model, scaler

model, scaler = load_model()

# =====================================
# TITLE
# =====================================

st.title(
    "⚡ Electricity Consumption Forecasting"
)

st.markdown(
    "### SimpleRNN-Based Energy Forecasting Dashboard"
)

st.divider()

# =====================================
# INPUTS
# =====================================

st.sidebar.header(
    "Last 24 Hours Consumption"
)

values = []

for i in range(24):

    value = st.sidebar.number_input(
        f"Hour {i+1}",
        min_value=0.0,
        value=1000.0
    )

    values.append(value)

# =====================================
# PREDICT
# =====================================

if st.button(
    "🚀 Forecast Next Hour"
):

    data = np.array(
        values
    ).reshape(-1,1)

    scaled = scaler.transform(
        data
    )

    X = scaled.reshape(
        1,
        24,
        1
    )

    pred = model.predict(
        X,
        verbose=0
    )

    next_hour = scaler.inverse_transform(
        pred
    )[0][0]

    current = values[-1]

    change = next_hour - current

    change_pct = (
        change/current
    ) * 100

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Current Usage",
            f"{current:.2f}"
        )

    with c2:

        st.metric(
            "Forecast Usage",
            f"{next_hour:.2f}"
        )

    with c3:

        st.metric(
            "% Change",
            f"{change_pct:.2f}%"
        )

    st.divider()

    if change_pct > 10:

        st.error(
            "🔥 Peak Demand Expected"
        )

    elif change_pct > 5:

        st.warning(
            "⚠ Increased Demand"
        )

    else:

        st.success(
            "✅ Stable Demand"
        )

    chart_data = values.copy()

    chart_data.append(
        next_hour
    )

    fig = plt.figure(
        figsize=(10,4)
    )

    plt.plot(
        range(1,26),
        chart_data,
        marker="o"
    )

    plt.title(
        "Electricity Forecast"
    )

    plt.xlabel(
        "Hour"
    )

    plt.ylabel(
        "Consumption"
    )

    plt.grid(True)

    st.pyplot(fig)

    st.subheader(
        "📊 Analytics"
    )

    a1, a2, a3 = st.columns(3)

    with a1:

        st.info(
            f"Average: {np.mean(values):.2f}"
        )

    with a2:

        st.info(
            f"Maximum: {np.max(values):.2f}"
        )

    with a3:

        st.info(
            f"Minimum: {np.min(values):.2f}"
        )

st.markdown("---")

st.caption(
    "⚡ Electricity Consumption Forecasting | SimpleRNN"
)
