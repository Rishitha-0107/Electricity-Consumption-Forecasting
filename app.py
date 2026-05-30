import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Electricity Consumption Forecasting",
    page_icon="⚡",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.metric-card{
    background:#f8fafc;
    padding:15px;
    border-radius:12px;
    border:1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(
        os.path.join(
            BASE_DIR,
            "energy_rnn_model.h5"
        ),
        compile=False
    )

    scaler = joblib.load(
        os.path.join(
            BASE_DIR,
            "scaler.pkl"
        )
    )

    return model, scaler

try:

    model, scaler = load_artifacts()

except Exception as e:

    st.error(
        f"Error loading model: {e}"
    )

    st.stop()

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<h1 style='text-align:center;color:#2563eb;'>
⚡ Electricity Consumption Forecasting System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h4 style='text-align:center;color:gray;'>
SimpleRNN-Based Energy Demand Prediction Platform
</h4>
""", unsafe_allow_html=True)

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header(
    "📊 Previous 24 Hours Consumption"
)

st.sidebar.caption(
    "Enter electricity usage for the last 24 hours"
)

hourly_values = []

for i in range(24):

    value = st.sidebar.number_input(
        f"Hour {i+1}",
        min_value=0.0,
        value=1000.0,
        step=10.0
    )

    hourly_values.append(value)

# ==========================================
# FORECAST BUTTON
# ==========================================

if st.button(
    "🚀 Forecast Next Hour",
    use_container_width=True
):

    try:

        values = np.array(
            hourly_values
        ).reshape(-1,1)

        scaled_values = scaler.transform(
            values
        )

        X = scaled_values.reshape(
            1,
            24,
            1
        )

        prediction = model.predict(
            X,
            verbose=0
        )

        next_hour = scaler.inverse_transform(
            prediction
        )[0][0]

        current = hourly_values[-1]

        change = (
            next_hour - current
        )

        percent_change = (
            change / current
        ) * 100

        # ==================================
        # KPI DASHBOARD
        # ==================================

        st.subheader(
            "📈 Energy Forecast Dashboard"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Current Usage",
                f"{current:.0f}"
            )

        with c2:

            st.metric(
                "Forecast Usage",
                f"{next_hour:.0f}"
            )

        with c3:

            st.metric(
                "Change",
                f"{change:.0f}"
            )

        with c4:

            st.metric(
                "% Change",
                f"{percent_change:.2f}%"
            )

        st.divider()

        # ==================================
        # ALERTS
        # ==================================

        if percent_change > 10:

            st.error(
                "🔥 Peak Consumption Expected"
            )

        elif percent_change > 5:

            st.warning(
                "⚠ Increased Demand Forecast"
            )

        else:

            st.success(
                "✅ Stable Energy Demand"
            )

        # ==================================
        # FORECAST CHART
        # ==================================

        chart_data = hourly_values.copy()

        chart_data.append(
            next_hour
        )

        fig = plt.figure(
            figsize=(12,5)
        )

        plt.plot(
            range(1,26),
            chart_data,
            marker="o"
        )

        plt.title(
            "Energy Consumption Forecast"
        )

        plt.xlabel(
            "Hour"
        )

        plt.ylabel(
            "Consumption"
        )

        plt.grid(True)

        st.pyplot(fig)

        # ==================================
        # ANALYTICS
        # ==================================

        st.subheader(
            "📊 Consumption Analytics"
        )

        a1, a2, a3, a4 = st.columns(4)

        with a1:

            st.info(
                f"Average: {np.mean(hourly_values):.2f}"
            )

        with a2:

            st.info(
                f"Maximum: {np.max(hourly_values):.2f}"
            )

        with a3:

            st.info(
                f"Minimum: {np.min(hourly_values):.2f}"
            )

        with a4:

            peak_hour = (
                np.argmax(hourly_values) + 1
            )

            st.info(
                f"Peak Hour: {peak_hour}"
            )

        # ==================================
        # DEMAND CATEGORY
        # ==================================

        st.subheader(
            "⚡ Demand Assessment"
        )

        avg_usage = np.mean(
            hourly_values
        )

        if avg_usage > 5000:

            st.error(
                "High Consumption Zone"
            )

        elif avg_usage > 2500:

            st.warning(
                "Moderate Consumption Zone"
            )

        else:

            st.success(
                "Low Consumption Zone"
            )

        # ==================================
        # AI INSIGHTS
        # ==================================

        st.subheader(
            "🤖 AI Insights"
        )

        insights = []

        if percent_change > 10:

            insights.append(
                "Potential peak demand period approaching."
            )

        if peak_hour >= 18:

            insights.append(
                "Evening demand spike detected."
            )

        if avg_usage > 3000:

            insights.append(
                "Overall consumption is above normal."
            )

        if len(insights) == 0:

            insights.append(
                "Energy demand appears stable."
            )

        for item in insights:

            st.info(item)

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

# ==========================================
# INFO SECTION
# ==========================================

st.divider()

st.subheader(
    "🏭 Project Information"
)

st.write("""
**Model:** SimpleRNN

**Input:** Previous 24 Hours Electricity Consumption

**Output:** Next Hour Forecast

**Applications:**
- Smart Grid Monitoring
- Energy Demand Forecasting
- Peak Load Management
- Utility Planning
- Power Distribution Optimization
""")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "⚡ Electricity Consumption Forecasting | Deep Learning SimpleRNN Model"
)
