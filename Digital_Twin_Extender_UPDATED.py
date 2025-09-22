import streamlit as st
import sqlite3
import requests
import json
import pandas as pd
from dotenv import load_dotenv
import os
import random
import time
from openai import AzureOpenAI
from datetime import datetime
import matplotlib.pyplot as plt

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
DB_PATH = os.getenv("DB_PATH", "twin_config.db")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-raj"

# -----------------------------
# Azure OpenAI Client
# -----------------------------
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# -----------------------------
# Utility: get configs from DB
# -----------------------------
def get_configs():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    rows = c.execute("SELECT * FROM twin_configs ORDER BY created_at DESC").fetchall()
    conn.close()
    return rows

# -----------------------------
# Utility: simulate telemetry
# -----------------------------
def simulate_data(twin_type):
    if twin_type == "Well":
        return {
            "pressure": round(random.uniform(1000, 1400), 2),
            "temperature": round(random.uniform(85, 110), 2),
            "flow_rate": round(random.uniform(700, 1000), 2),
            "status": "Simulated"
        }
    elif twin_type == "Pump":
        return {
            "rpm": round(random.uniform(1700, 2000), 2),
            "vibration": round(random.uniform(1.0, 1.5), 2),
            "amp_load": round(random.uniform(70, 80), 2),
            "status": "Simulated"
        }
    else:
        return {"status": "Simulated", "message": f"No model for twin type: {twin_type}"}

# -----------------------------
# Utility: fetch live data from API
# -----------------------------
def fetch_telemetry(config):
    headers = {}
    if config[4] == "Bearer Token":
        headers = {"Authorization": f"Bearer {config[5]}"}
    elif config[4] == "API Key":
        headers = {"x-api-key": config[5]}
    elif config[4] == "Basic Auth":
        headers = {"Authorization": f"Basic {config[5]}"}

    try:
        health = requests.get(f"{config[3]}{config[6]}", headers=headers, timeout=5)
        health.raise_for_status()
        telemetry = requests.get(f"{config[3]}{config[7]}", headers=headers, timeout=5)
        telemetry.raise_for_status()
        return telemetry.json(), None
    except Exception as e:
        return None, str(e)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Digital Twin Extender", layout="wide")
st.title("🔌 Digital Twin Extender")

configs = get_configs()
if configs:
    options = [f"{row[0]} - {row[1]} ({row[2]})" for row in configs]
    selected = st.sidebar.selectbox("Select a Digital Twin", options)
    selected_id = int(selected.split(" - ")[0])
    selected_config = next(row for row in configs if row[0] == selected_id)
else:
    st.warning("No digital twin configurations available.")
    st.stop()

tabs = st.tabs([
    "1. Status",
    "2. Live Trend",
    "3. Simulated Forecast",
    "4. GenAI Advisor",
    "5. Dashboard",
    "6. Chat with Twin",
    "7. Actuation Control",
    "8. Scenario Simulator",
    "9. Anomaly Detector"
])

# -----------------------------
# Tab 1 - Status
# -----------------------------
with tabs[0]:
    st.subheader(f"🛰️ Telemetry Status: {selected_config[1]}")
    data, error = fetch_telemetry(selected_config)
    if data:
        st.success("✅ Live Data from API")
        st.json(data)
    elif selected_config[8]:
        st.warning(f"⚠️ Live data failed: {error}")
        st.info("🔁 Showing fallback simulated data")
        st.json(simulate_data(selected_config[2]))
    else:
        st.error(f"❌ Failed to fetch data and fallback is disabled. Reason: {error}")

# -----------------------------
# Tab 2 - Live Trend
# -----------------------------
with tabs[1]:
    st.subheader("📈 Live Telemetry Trend")

    trend_type = st.selectbox("Metric to Simulate", ["pressure", "temperature", "rpm", "flow_rate"])
    chart_data = []
    timestamps = []

    for _ in range(20):
        val = simulate_data(selected_config[2]).get(trend_type, random.uniform(0, 1))
        chart_data.append(val)
        timestamps.append(datetime.now().strftime("%H:%M:%S"))
        time.sleep(0.1)

    fig, ax = plt.subplots()
    ax.plot(timestamps, chart_data, marker="o", linestyle="-")
    ax.set_title(f"Simulated {trend_type.capitalize()} Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(f"{trend_type.capitalize()}")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)

    st.pyplot(fig)

# -----------------------------
# Tab 3 - Simulated Forecast
# -----------------------------
with tabs[2]:
    st.subheader("🔮 24h Forecast Simulation")
    trend_type = st.selectbox("Metric to Forecast", ["pressure", "temperature", "rpm", "flow_rate"])

    current_data = simulate_data(selected_config[2])
    base_value = current_data.get(trend_type, random.uniform(0, 1))

    forecast_data = []
    for h in range(24):
        delta = random.uniform(-5, 5)
        base_value = max(0, base_value + delta)
        forecast_data.append({"hour": h, trend_type: round(base_value, 2)})

    df_forecast = pd.DataFrame(forecast_data)
    st.markdown("### Forecast Table")
    st.dataframe(df_forecast)
    st.markdown("### Forecast Chart")
    st.line_chart(df_forecast.set_index("hour"), use_container_width=True)

# -----------------------------
# Tab 4 - GenAI Advisor (Azure OpenAI)
# -----------------------------
with tabs[3]:
    st.subheader("🧠 GenAI Assistant")
    user_question = st.text_input("Ask something about the twin...", "Why is the pressure fluctuating?")
    twin_data = simulate_data(selected_config[2])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Get AI Insight"):
        prompt = [
            {"role": "system", "content": f"You are a digital twin advisor focused on {selected_config[13]}."},
            {"role": "user", "content": f"The current telemetry is: {json.dumps(twin_data)}. Question: {user_question}"}
        ]
        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=prompt,
                max_tokens=200
            )
            reply = response.choices[0].message.content.strip()
            st.session_state.chat_history.append((user_question, reply))
        except Exception as e:
            st.error(f"Error from Azure OpenAI: {str(e)}")

    for user, ai in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {user}")
        st.markdown(f"**Twin:** {ai}")

# -----------------------------
# Tab 5 - Dashboard
# -----------------------------
with tabs[4]:
    st.subheader(f"📊 Twin Dashboard: {selected_config[1]}")
    telemetry_data, err = fetch_telemetry(selected_config)
    if not telemetry_data:
        st.warning("Using simulated data due to fetch failure.")
        telemetry_data = simulate_data(selected_config[2])

    st.markdown("### Telemetry Snapshot")
    st.json(telemetry_data)

    df_metrics = pd.DataFrame.from_dict(telemetry_data, orient="index", columns=["Value"]).reset_index()
    df_metrics.columns = ["Metric", "Value"]

    st.markdown("### Key Indicators")
    cols = st.columns(min(4, len(df_metrics)))
    for i, row in df_metrics.iterrows():
        value = row["Value"]
        status_color = "🟢" if isinstance(value, (int, float)) and value > 0 else "🔴"
        cols[i % len(cols)].metric(label=row["Metric"], value=value, delta=status_color)

    numeric_metrics = df_metrics[df_metrics["Value"].apply(lambda x: isinstance(x, (int, float)))]
    if not numeric_metrics.empty:
        st.markdown("### Value Distribution")
        chart_df = pd.DataFrame({
            "Metric": numeric_metrics["Metric"],
            "Value": numeric_metrics["Value"]
        }).set_index("Metric")
        st.bar_chart(chart_df, use_container_width=True)

# -----------------------------
# Tab 6 - Chat with Twin (Azure OpenAI)
# -----------------------------
with tabs[5]:
    st.subheader(f"💬 Chat with Digital Twin: {selected_config[1]}")

    if "chat_twin_history" not in st.session_state:
        st.session_state.chat_twin_history = []

    question = st.text_input("Ask your question:", placeholder="e.g., What if temperature rises by 10%?")
    twin_data = simulate_data(selected_config[2])
    telemetry_str = json.dumps(twin_data)

    if st.button("Submit Question"):
        prompt = [
            {"role": "system", "content": f"You are a {selected_config[13]} assistant with access to real-time digital twin telemetry."},
            {"role": "user", "content": f"Current telemetry data: {telemetry_str}"},
            {"role": "user", "content": f"Question: {question}"}
        ]
        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=prompt,
                max_tokens=200
            )
            reply = response.choices[0].message.content.strip()
            st.session_state.chat_twin_history.append((question, reply))
        except Exception as e:
            st.error(f"❌ Azure OpenAI Error: {str(e)}")

    if st.session_state.chat_twin_history:
        st.markdown("### Chat History")
        for q, a in reversed(st.session_state.chat_twin_history[-10:]):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Twin:** {a}")

# -----------------------------
# Tab 7 - Actuation Control
# -----------------------------
with tabs[6]:
    st.subheader(f"🛠️ Actuation Control: {selected_config[1]}")
    if not selected_config[8]:
        st.warning("⚠️ Twin is running in fallback simulation mode. Actuation is disabled.")
    else:
        command = st.text_input("Enter Command", placeholder="e.g., set_rpm=1800")
        if st.button("Send Command"):
            try:
                headers = {}
                if selected_config[4] == "Bearer Token":
                    headers = {"Authorization": f"Bearer {selected_config[5]}"}
                elif selected_config[4] == "API Key":
                    headers = {"x-api-key": selected_config[5]}
                elif selected_config[4] == "Basic Auth":
                    headers = {"Authorization": f"Basic {selected_config[5]}"}

                url = f"{selected_config[3]}{selected_config[8]}"
                response = requests.post(url, headers=headers, json={"command": command}, timeout=5)

                if response.status_code == 200:
                    st.success(f"✅ Command sent successfully: {response.text}")
                else:
                    st.error(f"❌ Failed to send command. Status code: {response.status_code}")
            except Exception as e:
                st.error(f"⚠️ Error sending command: {str(e)}")

# -----------------------------
# Tab 8 - Scenario Simulator (Azure OpenAI)
# -----------------------------
with tabs[7]:
    st.subheader(f"🔮 Scenario Simulator: {selected_config[1]}")

    scenario_metric = st.selectbox("Metric to Modify", ["pressure", "temperature", "rpm", "flow_rate"])
    base_data = fetch_telemetry(selected_config)[0] or simulate_data(selected_config[2])
    current_value = base_data.get(scenario_metric, 0)
    delta_percent = st.slider("Change by (%)", -50, 50, 10)
    simulated_value = round(current_value * (1 + delta_percent / 100), 2)

    modified_data = base_data.copy()
    modified_data[scenario_metric] = simulated_value

    st.markdown("### 📊 Simulated Telemetry")
    st.json(modified_data)

    st.markdown("### 🧠 GenAI Advisory")
    try:
        prompt = [
            {"role": "system", "content": "You are a senior process control engineer analyzing digital twin telemetry."},
            {"role": "user", "content": f"Base telemetry data: {json.dumps(base_data)}"},
            {"role": "user", "content": f"Modified scenario data: {json.dumps(modified_data)}"}
        ]
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=prompt,
            max_tokens=300
        )
        scenario_reply = response.choices[0].message.content.strip()
        st.markdown(f"**Twin Advisor:** {scenario_reply}")
    except Exception as e:
        st.error(f"GenAI error: {str(e)}")

# -----------------------------
# Tab 9 - Anomaly Detector (Azure OpenAI)
# -----------------------------
with tabs[8]:
    st.subheader(f"⚠️ Telemetry Anomaly Detector: {selected_config[1]}")
    telemetry = fetch_telemetry(selected_config)[0] or simulate_data(selected_config[2])

    st.markdown("### 🔍 Current Telemetry")
    st.json(telemetry)

    thresholds = {"pressure": 1400, "temperature": 105, "rpm": 2000, "vibration": 1.4, "flow_rate": 950, "amp_load": 78}

    anomalies = []
    for key, limit in thresholds.items():
        if key in telemetry and isinstance(telemetry[key], (int, float)) and telemetry[key] > limit:
            anomalies.append((key, telemetry[key], limit))
            st.error(f"{key.capitalize()} = {telemetry[key]} exceeds threshold {limit}")

    if anomalies:
        st.markdown("### 🧠 GenAI Analysis of Anomalies")
        try:
            prompt = [
                {"role": "system", "content": f"You are a {selected_config[13]} digital twin anomaly analyzer."},
                {"role": "user", "content": f"Telemetry: {json.dumps(telemetry)}"},
                {"role": "user", "content": f"Anomalies: {json.dumps(anomalies)}"}
            ]
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=prompt,
                max_tokens=200
            )
            st.markdown(f"**Twin Anomaly Advisor:** {response.choices[0].message.content.strip()}")
        except Exception as e:
            st.error(f"GenAI error: {str(e)}")
    else:
        st.success("✅ No anomalies detected.")
