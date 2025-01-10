import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
from prophet import Prophet

# --- Page Config ---
st.set_page_config(page_title="WeShip Express Dashboard", layout="wide")
st.title("📦 WeShip Express Fulfillment Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("Wines_Til_Sold_Out.csv")
    df['Profit'] = df['BEF_Cost'] - df['CI_Net Charge Amount']
    df['Profit_Positive'] = df['Profit'].apply(lambda x: max(x, 1))  # Ensure positive values for map size
    return df

df = load_data()

# --- Load ZIP Code Coordinates Dataset ---
@st.cache_data
def load_zip_coordinates():
    zip_file_path = '/Users/jashandeepsinghkaleka/Downloads/simplemaps_uszips_basicv1.86/uszips.csv'
    zip_df = pd.read_csv(zip_file_path)
    zip_df['zip'] = zip_df['zip'].astype(str).str.zfill(5)  # Ensure ZIP codes are 5 digits
    return zip_df

zip_df = load_zip_coordinates()

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")
selected_customer = st.sidebar.selectbox("Select Customer", df["BEF_Customer Name"].unique())
selected_carrier = st.sidebar.multiselect("Select Carrier(s)", df["CI_Carrier Name"].unique(), default=df["CI_Carrier Name"].unique())
selected_date_range = st.sidebar.date_input("Select Date Range", [])

filtered_df = df[
    (df["BEF_Customer Name"] == selected_customer) &
    (df["CI_Carrier Name"].isin(selected_carrier))
]

# --- Summary Metrics ---
st.header(f"Executive Summary for {selected_customer}")

total_revenue = filtered_df["BEF_Cost"].sum()
total_cost = filtered_df["CI_Net Charge Amount"].sum()
total_profit = filtered_df["Profit"].sum()
total_shipments = filtered_df.shape[0]

revenue_per_shipment = total_revenue / total_shipments if total_shipments > 0 else 0
cost_per_shipment = total_cost / total_shipments if total_shipments > 0 else 0
profit_per_shipment = total_profit / total_shipments if total_shipments > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${total_revenue:,.2f}")
col2.metric("Total Cost", f"${total_cost:,.2f}")
col3.metric("Total Profit", f"${total_profit:,.2f}")
col4.metric("Total Shipments", f"{total_shipments:,}")

col1, col2, col3, _ = st.columns(4)
col1.metric("Revenue per Shipment", f"${revenue_per_shipment:,.2f}")
col2.metric("Cost per Shipment", f"${cost_per_shipment:,.2f}")
col3.metric("Profit per Shipment", f"${profit_per_shipment:,.2f}")

# --- Profit Breakdown by Carrier ---
st.subheader("💰 Profit, Revenue, and Cost Breakdown by Carrier")

# Group by Carrier and calculate Revenue, Cost, and Profit
carrier_breakdown = filtered_df.groupby("CI_Carrier Name").agg({
    "BEF_Cost": "sum",  # Revenue
    "CI_Net Charge Amount": "sum",  # Cost
    "Profit": "sum"
}).reset_index()

# Rename columns for clarity
carrier_breakdown.columns = ["Carrier", "Revenue", "Cost", "Profit"]

# Plot Stacked Bar Chart
fig_carrier_breakdown = px.bar(
    carrier_breakdown,
    x="Carrier",
    y=["Revenue", "Cost", "Profit"],
    title="Profit, Revenue, and Cost Breakdown by Carrier",
    text_auto=True,
    labels={"value": "Amount ($)", "Carrier": "Carrier Name"},
    barmode="group",  # Grouped bars for better clarity
    color_discrete_sequence=px.colors.qualitative.Vivid
)
st.plotly_chart(fig_carrier_breakdown)


# --- Merge Coordinates with Shipment Data ---
df['CI_Recipient Zip Code'] = df['CI_Recipient Zip Code'].astype(str).str.zfill(5)
df = df.merge(zip_df[['zip', 'lat', 'lng']], left_on='CI_Recipient Zip Code', right_on='zip', how='left')

# --- Plot Recipient Map ---
st.subheader("📦 Recipient Locations")
fig_recipient_map = px.scatter_mapbox(
    df,
    lat="lat",
    lon="lng",
    color="CI_Carrier Name",
    size="Profit_Positive",
    hover_data={
        "Profit": ":.2f",
        "CI_Number of Pieces": True,
        "CI_Actual Weight Amount": True
    },
    mapbox_style="carto-positron",
    title="Recipient Locations (USA)",
    zoom=3,
    center={"lat": 37.0902, "lon": -95.7129}  # Centered on the USA
)
st.plotly_chart(fig_recipient_map)

# --- Time Series Analysis ---
st.subheader("📈 Revenue, Cost, and Profit Over Time")
filtered_df['BEF_Invoice Date'] = pd.to_datetime(filtered_df['BEF_Invoice Date'])

# Aggregate by date for time series analysis
time_series = filtered_df.groupby('BEF_Invoice Date').agg({
    'BEF_Cost': 'sum',
    'CI_Net Charge Amount': 'sum',
    'Profit': 'sum'
}).reset_index()

# Plot time series chart
fig_time_series = px.line(
    time_series,
    x='BEF_Invoice Date',
    y=['BEF_Cost', 'CI_Net Charge Amount', 'Profit'],
    labels={'value': 'Amount ($)', 'BEF_Invoice Date': 'Date'},
    title="Revenue, Cost, and Profit Over Time"
)
st.plotly_chart(fig_time_series)

# --- Cost Breakdown Visualization ---
st.subheader("💸 Cost Breakdown by Charge Type")

# Select relevant charge columns
charge_columns = [
    'CI_Charge_Service', 'CI_Charge_Fuel', 'CI_Charge_Late_Fee', 'CI_Charge_Discounts',
    'CI_Charge_Residential_Total', 'CI_Charge_Delivery_Type', 'CI_Charge_Pickup', 'CI_Charge_DAS',
    'CI_Charge_Handling', 'CI_Charge_Signature', 'CI_Charge_Declared_Value', 'CI_Charge_Labels',
    'CI_Charge_Weight', 'CI_Charge_Other', 'CI_Charge_Sales_Tax'
]

# Aggregate costs by month
filtered_df['YearMonth'] = filtered_df['BEF_Invoice Date'].dt.to_period('M')
monthly_cost_breakdown = filtered_df.groupby('YearMonth')[charge_columns].sum().reset_index()
monthly_cost_breakdown['YearMonth'] = monthly_cost_breakdown['YearMonth'].astype(str)

# Stacked Bar Chart for Monthly Cost Breakdown
fig_cost_breakdown = px.bar(
    monthly_cost_breakdown,
    x='YearMonth',
    y=charge_columns,
    title="Monthly Cost Breakdown by Charge Type",
    labels={'value': 'Amount ($)', 'YearMonth': 'Year-Month'},
    color_discrete_sequence=px.colors.qualitative.Set3
)
st.plotly_chart(fig_cost_breakdown)

# Total Cost Breakdown by Charge Type (Pie Chart)
total_cost_breakdown = filtered_df[charge_columns].sum().reset_index()
total_cost_breakdown.columns = ['Charge Type', 'Amount']


# --- AI Insights ---
st.header("🤖 AI Insights & Suggestions")
st.write("Ask the AI for shipment insights or profit optimization strategies.")

# Sample Question and Answer
st.subheader("💡 Sample Question")
sample_question = "What insights can we derive from the data to improve profitability and reduce delivery delays?"
st.write(f"**Question:** {sample_question}")

sample_answer = """
Based on historical data analysis:
1. **Carrier Performance**: Certain carriers have consistently higher profit margins but also higher delays. Adjusting carrier contracts can optimize both profitability and on-time deliveries.
2. **High-Volume Customers**: Focus on customers with high shipment volumes and profit margins to offer loyalty discounts and increase retention.
3. **Seasonal Trends**: Profitability spikes during specific months. Align marketing and operations to capitalize on these trends.
4. **Zone-Based Pricing**: Implement zone-based pricing adjustments to optimize costs in regions with lower margins.
"""
st.write(f"**Answer:** {sample_answer}")

# User Input Question
user_query = st.text_input("Enter your question for the AI:")

if user_query:
    st.write(f"**Question:** {user_query}")
    st.write("**Answer:** This is a prototype demo. In a real implementation, the AI would provide a detailed, data-driven answer.")
    st.write("**Answer:** This is a prototype demo. In a real implementation, the AI would provide a detailed, data-driven answer.")

# --- Customer Report ---
st.subheader(f"📋 Detailed Report for {selected_customer}")
st.write(filtered_df.head(10))

# --- Download Option ---
st.download_button(
    label="📥 Download Customer Report",
    data=filtered_df.to_csv(index=False),
    file_name=f"{selected_customer}_report.csv",
    mime="text/csv",
)
