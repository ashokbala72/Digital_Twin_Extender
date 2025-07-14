
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()
DB_PATH = os.getenv("DB_PATH", "twin_config.db")

# DB connection
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

# Auto-upgrade schema (add missing columns if needed)
columns_to_add = {
    "actuation_endpoint": "TEXT",
    "model_coverage": "INTEGER",
    "data_confidence": "INTEGER",
    "enable_scenarios": "INTEGER",
    "advisor_profile": "TEXT"
}

for column, col_type in columns_to_add.items():
    try:
        c.execute(f"ALTER TABLE twin_configs ADD COLUMN {column} {col_type}")
    except sqlite3.OperationalError:
        pass  # Column already exists

# Streamlit UI
st.set_page_config(page_title="Digital Twin Configurator", layout="wide")
st.title("🛠️ Digital Twin Configurator")

with st.form("config_form"):
    st.subheader("🔧 Enter Digital Twin Connection Details")

    twin_name = st.text_input("Twin Name", "Shell Wellhead DT")
    twin_type = st.selectbox("Twin Type", ["Well", "Pump", "Refinery", "Pipeline", "Other"])
    api_base_url = st.text_input("API Base URL", "https://api.shell.com/dt/wellhead1")
    auth_type = st.selectbox("Auth Type", ["Bearer Token", "Basic Auth", "API Key"])
    default_auth = os.getenv("DEFAULT_AUTH_TOKEN", "")
    auth_credential = st.text_input("Auth Credential", default_auth, type="password")
    heartbeat_endpoint = st.text_input("Heartbeat Endpoint", "/health")
    telemetry_endpoint = st.text_input("Telemetry Endpoint", "/data/latest")
    actuation_endpoint = st.text_input("Actuation Endpoint", "/actuate")
    enable_fallback = st.checkbox("Enable Fallback Simulation", value=True)
    enable_scenarios = st.checkbox("Enable Scenario Simulation", value=True)
    model_coverage = st.slider("Model Coverage (%)", 0, 100, 85)
    data_confidence = st.slider("Data Confidence (%)", 0, 100, 90)
    advisor_profile = st.selectbox("GenAI Advisory Mode", ["Default", "Safety-Focused", "Efficiency-Oriented", "Explainability"])
    notes = st.text_area("Notes", "Testing wellhead twin")

    submitted = st.form_submit_button("💾 Save Configuration")

    if submitted:
        c.execute("""
        INSERT INTO twin_configs (
            twin_name, twin_type, api_base_url, auth_type, auth_credential,
            heartbeat_endpoint, telemetry_endpoint, actuation_endpoint,
            enable_fallback, enable_scenarios, model_coverage, data_confidence,
            advisor_profile, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            twin_name, twin_type, api_base_url, auth_type, auth_credential,
            heartbeat_endpoint, telemetry_endpoint, actuation_endpoint,
            int(enable_fallback), int(enable_scenarios), model_coverage, data_confidence,
            advisor_profile, notes
        ))
        conn.commit()
        st.success(f"✅ Configuration for '{twin_name}' saved successfully!")

# View configs
st.subheader("📋 Saved Configurations")

rows = c.execute("SELECT * FROM twin_configs ORDER BY created_at DESC").fetchall()
if rows:
    df = pd.DataFrame(rows, columns=[
        "ID", "Twin Name", "Twin Type", "API Base URL", "Auth Type", "Auth Credential",
        "Heartbeat Endpoint", "Telemetry Endpoint", "Actuation Endpoint",
        "Enable Fallback", "Enable Scenarios", "Model Coverage", "Data Confidence",
        "Advisor Profile", "Notes", "Created At"
    ])
    st.dataframe(df.drop(columns=["Auth Credential"]), use_container_width=True)
else:
    st.info("No configurations saved yet.")

conn.close()
