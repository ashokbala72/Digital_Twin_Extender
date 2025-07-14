import streamlit as st
import sqlite3
import requests
import json
import pandas as pd
from dotenv import load_dotenv
import os
import random
import time
from openai import OpenAI

# Load environment variables
load_dotenv()
DB_PATH = os.getenv("DB_PATH", "twin_config.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Utility: get configs from DB
def get_configs():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    rows = c.execute("SELECT * FROM twin_configs ORDER BY created_at DESC").fetchall()
    conn.close()
    return rows

# Utility: simulate telemetry
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

# Utility: fetch live data from API
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

# Streamlit UI
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


# Tab 1 - Status
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

# Tab 2 - Live Trend
from datetime import datetime
import matplotlib.pyplot as plt

with tabs[1]:  # Tab 2 - Live Trend
    st.subheader("📈 Live Telemetry Trend")

    trend_type = st.selectbox("Metric to Simulate", ["pressure", "temperature", "rpm", "flow_rate"])
    chart_data = []
    timestamps = []

    # Simulate 20 real-time readings with timestamps
    for _ in range(20):
        val = simulate_data(selected_config[2]).get(trend_type, random.uniform(0, 1))
        chart_data.append(val)
        timestamps.append(datetime.now().strftime("%H:%M:%S"))
        time.sleep(0.1)

    # Plot with axis labels using Matplotlib
    fig, ax = plt.subplots()
    ax.plot(timestamps, chart_data, marker="o", linestyle="-")
    ax.set_title(f"Simulated {trend_type.capitalize()} Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(f"{trend_type.capitalize()}")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)

    st.pyplot(fig)


# Tab 3 - Simulated Forecast
with tabs[2]:  # Tab 3 - Simulated Forecast
    st.subheader("🔮 24h Forecast Simulation")

    # Select metric to forecast
    trend_type = st.selectbox("Metric to Forecast", ["pressure", "temperature", "rpm", "flow_rate"])

    # Get starting value from current simulation
    current_data = simulate_data(selected_config[2])
    base_value = current_data.get(trend_type, random.uniform(0, 1))

    # Generate 24-hour forecast with drifting logic
    forecast_data = []
    for h in range(24):
        delta = random.uniform(-5, 5)
        base_value = max(0, base_value + delta)  # Keep value non-negative
        forecast_data.append({"hour": h, trend_type: round(base_value, 2)})

    # Convert to DataFrame
    df_forecast = pd.DataFrame(forecast_data)

    # Display table and chart
    st.markdown("### Forecast Table")
    st.dataframe(df_forecast)

    st.markdown("### Forecast Chart")
    st.line_chart(df_forecast.set_index("hour"), use_container_width=True)


# Tab 4 - GenAI Advisor
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
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=prompt,
                max_tokens=200
            )
            reply = response.choices[0].message.content.strip()
            st.session_state.chat_history.append((user_question, reply))
        except Exception as e:
            st.error(f"Error from OpenAI: {str(e)}")

    for i, (user, ai) in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**You:** {user}")
        st.markdown(f"**Twin:** {ai}")

# Tab 5 - Placeholder
with tabs[4]:  # Dashboard Tab
    st.subheader(f"📊 Twin Dashboard: {selected_config[1]}")

    # Try fetching live telemetry or fallback to simulation
    telemetry_data, err = fetch_telemetry(selected_config)
    if not telemetry_data:
        st.warning("Using simulated data due to fetch failure.")
        telemetry_data = simulate_data(selected_config[2])

    # Show raw data
    st.markdown("### Telemetry Snapshot")
    st.json(telemetry_data)

    # Extract and format metrics
    df_metrics = pd.DataFrame.from_dict(telemetry_data, orient="index", columns=["Value"]).reset_index()
    df_metrics.columns = ["Metric", "Value"]

    # Display KPIs
    st.markdown("### Key Indicators")
    cols = st.columns(min(4, len(df_metrics)))
    for i, row in df_metrics.iterrows():
        value = row["Value"]
        status_color = "🟢" if isinstance(value, (int, float)) and value > 0 else "🔴"
        cols[i % len(cols)].metric(label=row["Metric"], value=value, delta=status_color)

    # Trend chart if numerical
    numeric_metrics = df_metrics[df_metrics["Value"].apply(lambda x: isinstance(x, (int, float)))]
    if not numeric_metrics.empty:
        st.markdown("### Value Distribution")
        chart_df = pd.DataFrame({
            "Metric": numeric_metrics["Metric"],
            "Value": numeric_metrics["Value"]
        }).set_index("Metric")
        st.bar_chart(chart_df, use_container_width=True)
    else:
        st.info("No numeric metrics available for charting.")

with tabs[5]:  # Tab 6 - Chat with Twin
    st.subheader(f"💬 Chat with Digital Twin: {selected_config[1]}")

    # Initialize memory
    if "chat_twin_history" not in st.session_state:
        st.session_state.chat_twin_history = []

    # Get user question
    question = st.text_input("Ask your question:", placeholder="e.g., What if temperature rises by 10%?")
    twin_data = simulate_data(selected_config[2])  # Or use real data if available
    telemetry_str = json.dumps(twin_data)

    if st.button("Submit Question"):
        prompt = [
            {"role": "system", "content": f"You are a {selected_config[13]} assistant with access to real-time digital twin telemetry."},
            {"role": "user", "content": f"Current telemetry data: {telemetry_str}"},
            {"role": "user", "content": f"Question: {question}"}
        ]
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=prompt,
                max_tokens=200
            )
            reply = response.choices[0].message.content.strip()
            st.session_state.chat_twin_history.append((question, reply))
        except Exception as e:
            st.error(f"❌ OpenAI Error: {str(e)}")

    # Show chat history
    if st.session_state.chat_twin_history:
        st.markdown("### Chat History")
        for q, a in reversed(st.session_state.chat_twin_history[-10:]):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Twin:** {a}")


with tabs[6]:  # Tab 7 - Actuation Control
    st.subheader(f"🛠️ Actuation Control: {selected_config[1]}")

    if not selected_config[8]:  # If fallback mode enabled
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

                url = f"{selected_config[3]}{selected_config[8]}"  # API base + actuation endpoint
                response = requests.post(url, headers=headers, json={"command": command}, timeout=5)

                if response.status_code == 200:
                    st.success(f"✅ Command sent successfully: {response.text}")
                else:
                    st.error(f"❌ Failed to send command. Status code: {response.status_code}")
            except Exception as e:
                st.error(f"⚠️ Error sending command: {str(e)}")

        # Helpful example commands
        st.markdown("""
### 💡 Example Commands by Twin Type
| Twin Type  | Example Command              | Description                            |
|------------|------------------------------|----------------------------------------|
| Pump       | `set_rpm=1800`               | Sets pump RPM to 1800                  |
| Pump       | `start` / `stop`             | Starts or stops the pump               |
| Well       | `open_valve=A1`              | Opens valve A1                         |
| Well       | `choke=65`                   | Sets choke opening to 65%              |
| Pipeline   | `divert_flow=section_3`      | Diverts flow to section 3              |
| Pipeline   | `set_pressure_limit=1200`    | Updates pressure threshold             |
| Refinery   | `shutdown_unit=Furnace1`     | Initiates safe shutdown for a unit     |
| Any        | `diagnostics_mode=on`        | Puts the twin into diagnostics mode    |
""")



with tabs[7]:  # Tab 8 - Scenario Simulator
    st.subheader(f"🔮 Scenario Simulator: {selected_config[1]}")
    st.markdown("Simulate what-if scenarios by adjusting telemetry inputs.")

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
            {
                "role": "system",
                "content": (
                    "You are a senior process control engineer responsible for analyzing telemetry and simulation "
                    "data from industrial digital twins. Your goal is to identify the impact of changes in operational "
                    "parameters like pressure, rpm, or flow rate on system safety, efficiency, and reliability. Always "
                    "provide specific engineering insights, causal relationships, and recommended actions. Assume the twin "
                    "is for a critical energy asset (e.g., pump, pipeline, or well)."
                )
            },
            {"role": "user", "content": f"Base telemetry data: {json.dumps(base_data)}"},
            {
                "role": "user",
                "content": (
                    f"Modified scenario data: {json.dumps(modified_data)}. "
                    "Please compare the two sets, explain the likely cause and impact of the changes, "
                    "and suggest operational or safety recommendations."
                )
            }
        ]
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            max_tokens=300
        )
        scenario_reply = response.choices[0].message.content.strip()
        st.markdown(f"**Twin Advisor:** {scenario_reply}")
    except Exception as e:
        st.error(f"GenAI error: {str(e)}")



with tabs[8]:  # Tab 9 - Anomaly Detector
    st.subheader(f"⚠️ Telemetry Anomaly Detector: {selected_config[1]}")
    telemetry = fetch_telemetry(selected_config)[0] or simulate_data(selected_config[2])

    st.markdown("### 🔍 Current Telemetry")
    st.json(telemetry)

    thresholds = {
        "pressure": 1400,
        "temperature": 105,
        "rpm": 2000,
        "vibration": 1.4,
        "flow_rate": 950,
        "amp_load": 78
    }

    st.markdown("### 🚨 Detected Anomalies")
    anomalies = []
    for key, limit in thresholds.items():
        if key in telemetry and isinstance(telemetry[key], (int, float)) and telemetry[key] > limit:
            anomalies.append((key, telemetry[key], limit))
            st.error(f"{key.capitalize()} = {telemetry[key]} exceeds threshold of {limit}")

    if anomalies:
        st.markdown("### 🧠 GenAI Analysis of Anomalies")
        try:
            prompt = [
                {"role": "system", "content": f"You are a {selected_config[13]} digital twin anomaly analyzer."},
                {"role": "user", "content": f"Telemetry: {json.dumps(telemetry)}"},
                {"role": "user", "content": f"Anomalies: {json.dumps(anomalies)}. What is the likely cause or risk?"}
            ]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=prompt,
                max_tokens=200
            )
            st.markdown(f"**Twin Anomaly Advisor:** {response.choices[0].message.content.strip()}")
        except Exception as e:
            st.error(f"GenAI error: {str(e)}")
    else:
        st.success("✅ No anomalies detected based on defined thresholds.")