import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
from prophet import Prophet
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import gc
import traceback
import psutil
import os

# Set page config first before any other Streamlit commands
st.set_page_config(page_title="WeShip Express Dashboard", layout="wide")

# Function to get current memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024  # Convert to MB
    return memory_mb

# Function to check if memory usage is too high
def check_memory_usage():
    memory_mb = get_memory_usage()
    # Streamlit cloud has a 1GB memory limit
    memory_limit_mb = 1000  
    
    if memory_mb > memory_limit_mb * 0.9:  # Warning at 90% usage
        st.warning(f"High memory usage detected: {memory_mb:.1f}MB / {memory_limit_mb}MB")
        return True
    return False

# --- Load Data in Background ---
@st.cache_data
def load_background_data():
    try:
        # Log initial memory usage
        st.write(f"Initial memory usage: {get_memory_usage():.1f}MB")

        # Establish connection to Snowflake
        def connectToSnowflake():
            try:
                connect = snowflake.connector.connect(
                    account=st.secrets["snowflake"]["account"],
                    user=st.secrets["snowflake"]["user"], 
                    password=st.secrets["snowflake"]["password"],
                    role=st.secrets["snowflake"]["role"],
                    warehouse=st.secrets["snowflake"]["warehouse"],
                    database=st.secrets["snowflake"]["database"],
                    schema=st.secrets["snowflake"]["schema"]
                )
                return connect
            except Exception as e:
                st.error(f"Failed to connect to Snowflake: {str(e)}")
                return None

        conn = connectToSnowflake()
        if conn is None:
            return pd.DataFrame(), pd.DataFrame()
            
        cur = conn.cursor()

        try:
            # Query data from Snowflake
            query = """
                SELECT *
                FROM JOINED_BEF_X_CARRIER_INVOICE_V2 
                WHERE "NetSuite_Customer Name" = 'Wines ''til Sold Out'
                AND _merge = 'both'
            """
            cur.execute(query)
            df = cur.fetch_pandas_all()
            df['Profit'] = df['BEF_Cost'] - df['CI_Net Charge Amount']
            df['Profit_Positive'] = df['Profit'].apply(lambda x: max(x, 1))
            
            # Log memory after loading main dataframe
            st.write(f"Memory after loading main data: {get_memory_usage():.1f}MB")
            
            # Query zip code data
            zip_query = """
                SELECT ZIP, LAT, LNG 
                FROM ZIP_CODES
            """
            cur.execute(zip_query)
            zip_df = cur.fetch_pandas_all()
            zip_df['ZIP'] = zip_df['ZIP'].astype(str).str.zfill(5)
            
            # Log final memory usage
            st.write(f"Final memory usage: {get_memory_usage():.1f}MB")
            
            return df, zip_df

        except Exception as e:
            st.error(f"Error querying data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
            
        finally:
            cur.close()
            conn.close()
            gc.collect() # Force garbage collection

    except Exception as e:
        st.error(f"Error loading data: {str(e)}\n{traceback.format_exc()}")
        return pd.DataFrame(), pd.DataFrame()

# Start loading data in background
try:
    background_data, zip_df = load_background_data()
    if background_data.empty or zip_df.empty:
        st.error("Failed to load required data. Please try refreshing the page.")
        st.stop()
        
    # Check memory usage after loading data
    if check_memory_usage():
        st.warning("Consider reducing the data size or optimizing memory usage")
        
except Exception as e:
    st.error(f"Critical error loading data: {str(e)}")
    st.stop()

# --- Login Screen ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        try:
            if st.session_state["username"].lower() == st.secrets["login"]["username"] and st.session_state["password"] == st.secrets["login"]["password"]:
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # Don't store password
                del st.session_state["username"]  # Don't store username
            else:
                st.session_state["password_correct"] = False
        except Exception as e:
            st.error(f"Login error: {str(e)}")
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show centered login form
        try:
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                col_a, col_b, col_c = st.columns([1,3,1])
                with col_b:
                    st.image('Weshipcirclelogo_Logo_Full_Color_RGB.svg', width=250, use_container_width=True)
                st.markdown("<h1 style='text-align: center;'>Login</h1>", unsafe_allow_html=True)
                st.text_input("Username", key="username")
                st.text_input("Password", type="password", key="password")
                st.button("Login", on_click=password_entered, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying login form: {str(e)}")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show centered error form
        try:
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                col_a, col_b, col_c = st.columns([1,3,1])
                with col_b:
                    st.image('Weshipcirclelogo_Logo_Full_Color_RGB.svg', width=100, use_container_width=True)
                st.markdown("<h1 style='text-align: center;'>Login</h1>", unsafe_allow_html=True)
                st.text_input("Username", key="username")
                st.text_input("Password", type="password", key="password")
                st.error("😕 User not known or password incorrect")
                st.button("Login", on_click=password_entered, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying error form: {str(e)}")
        return False
    else:
        # Password correct
        return True

if check_password():
    try:
        st.title("📦 WeShip Express Fulfillment Dashboard")

        # Use the pre-loaded data
        df = background_data.copy() # Create copy to avoid modifying cached data

        # --- Sidebar Filters ---
        st.sidebar.header("Filter Options")
        
        # Wrap widget creation in try-except blocks
        try:
            selected_customer = st.sidebar.selectbox("Select Customer", df["BEF_Customer Name"].unique())
        except Exception as e:
            st.sidebar.error(f"Error loading customers: {str(e)}")
            selected_customer = None
            
        try:
            selected_carrier = st.sidebar.multiselect("Select Carrier(s)", df["CI_Carrier Name"].unique(), default=df["CI_Carrier Name"].unique())
        except Exception as e:
            st.sidebar.error(f"Error loading carriers: {str(e)}")
            selected_carrier = []
        
        try:
            # Convert BEF_Invoice Date to datetime if not already
            df['BEF_Invoice Date'] = pd.to_datetime(df['BEF_Invoice Date'])
            
            # Date range selector with min/max dates from data
            min_date = df['BEF_Invoice Date'].min().date()
            max_date = df['BEF_Invoice Date'].max().date()
            selected_date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        except Exception as e:
            st.sidebar.error(f"Error with date selection: {str(e)}")
            selected_date_range = []

        # Apply all filters including date range
        try:
            if len(selected_date_range) == 2:
                start_date, end_date = selected_date_range
                filtered_df = df[
                    (df["BEF_Customer Name"] == selected_customer) &
                    (df["CI_Carrier Name"].isin(selected_carrier)) &
                    (df["BEF_Invoice Date"].dt.date >= start_date) &
                    (df["BEF_Invoice Date"].dt.date <= end_date)
                ]
            else:
                filtered_df = df[
                    (df["BEF_Customer Name"] == selected_customer) &
                    (df["CI_Carrier Name"].isin(selected_carrier))
                ]

            # Clear memory after filtering
            gc.collect()

            # Log memory usage after filtering
            st.write(f"Memory after filtering: {get_memory_usage():.1f}MB")

        except Exception as e:
            st.error(f"Error filtering data: {str(e)}")
            filtered_df = pd.DataFrame()

        # --- Summary Metrics ---
        st.header(f"Executive Summary for {selected_customer}")

        try:
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
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")

        # --- Profit Breakdown by Carrier ---
        st.subheader("💰 Profit, Revenue, and Cost Breakdown by Carrier")

        try:
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
            
            # Clear memory after plotting
            del fig_carrier_breakdown
            gc.collect()

            # Log memory usage after carrier breakdown
            st.write(f"Memory after carrier breakdown: {get_memory_usage():.1f}MB")

        except Exception as e:
            st.error(f"Error creating carrier breakdown: {str(e)}")

        # Comment out memory-intensive sections for debugging
        """
        # --- Merge Coordinates with Shipment Data ---
        try:
            df['CI_Recipient Zip Code'] = df['CI_Recipient Zip Code'].astype(str).str.zfill(5)
            df = df.merge(zip_df[['ZIP', 'LAT', 'LNG']], left_on='CI_Recipient Zip Code', right_on='ZIP', how='left')

            # --- Plot Recipient Map ---
            st.subheader("📦 Recipient Locations")
            fig_recipient_map = px.scatter_mapbox(
                df,
                lat="LAT",
                lon="LNG",
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
            
            # Clear memory after plotting
            del fig_recipient_map
            gc.collect()

        except Exception as e:
            st.error(f"Error creating recipient map: {str(e)}")

        # --- Time Series Analysis ---
        st.subheader("📈 Revenue, Cost, and Profit Over Time")
        
        try:
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
            
            # Clear memory after plotting
            del fig_time_series
            gc.collect()

        except Exception as e:
            st.error(f"Error creating time series analysis: {str(e)}")

        # --- Volume Analysis by Month ---
        st.subheader("📊 Monthly Volume Analysis & Forecast")
        
        try:
            # Prepare monthly volume data
            filtered_df['YearMonth'] = pd.to_datetime(filtered_df['BEF_Invoice Date']).dt.to_period('M')
            monthly_volume = filtered_df.groupby('YearMonth').size().reset_index()
            monthly_volume.columns = ['YearMonth', 'Volume']
            monthly_volume['YearMonth'] = monthly_volume['YearMonth'].astype(str)
            monthly_volume['YearMonth'] = pd.to_datetime(monthly_volume['YearMonth'])

            # Prepare data for Prophet
            prophet_df = monthly_volume.copy()
            prophet_df.columns = ['ds', 'y']
            
            # Create and fit Prophet model with stricter bounds
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                        growth='logistic')
            prophet_df['cap'] = prophet_df['y'].max() * 1.3  # Reduced upper bound
            prophet_df['floor'] = prophet_df['y'].min() * 0.8  # Set floor to 80% of minimum historical value
            model.fit(prophet_df)
            
            # Create future dates for 2025
            future_dates = model.make_future_dataframe(periods=12, freq='M')
            future_dates['cap'] = prophet_df['cap'].iloc[0]
            future_dates['floor'] = prophet_df['floor'].iloc[0]
            forecast = model.predict(future_dates)
            
            # Ensure no negative values in forecast
            forecast['yhat'] = forecast['yhat'].clip(lower=0)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
            
            # Create combined visualization with improved spacing
            fig_combined = px.bar(
                monthly_volume,
                x='YearMonth',
                y='Volume',
                title="Monthly Shipment Volume & Forecast through 2025",
                labels={'YearMonth': 'Month', 'Volume': 'Number of Shipments'}
            )
            
            # Add volume numbers above bars with increased spacing
            fig_combined.add_scatter(
                x=monthly_volume['YearMonth'],
                y=monthly_volume['Volume'] + monthly_volume['Volume'].max() * 0.05,  # Position text higher above bars
                mode='text',
                text=monthly_volume['Volume'],
                textposition='top center',
                showlegend=False
            )
            
            # Add trend line for historical data
            fig_combined.add_scatter(
                x=monthly_volume['YearMonth'],
                y=monthly_volume['Volume'].rolling(window=3).mean(),
                mode='lines',
                name='3-Month Moving Average',
                line=dict(color='red', width=2)
            )
            
            # Add forecast line with improved text positioning
            fig_combined.add_scatter(
                x=forecast['ds'][len(monthly_volume):],  # Only show forecast for future dates
                y=forecast['yhat'][len(monthly_volume):].round(),
                mode='lines+text',
                name='Forecast',
                text=forecast['yhat'][len(monthly_volume):].round().astype(int),
                textposition='top center',
                textfont=dict(size=10),
                line=dict(color='blue', dash='dash', width=2)
            )
            
            # Add confidence interval
            fig_combined.add_scatter(
                x=forecast['ds'][len(monthly_volume):],  # Only show forecast for future dates
                y=forecast['yhat_upper'][len(monthly_volume):],
                mode='lines',
                name='Confidence Interval',
                line=dict(color='lightblue', width=1),
                showlegend=True
            )
            
            fig_combined.add_scatter(
                x=forecast['ds'][len(monthly_volume):],  # Only show forecast for future dates
                y=forecast['yhat_lower'][len(monthly_volume):],
                mode='lines',
                line=dict(color='lightblue', width=1),
                fill='tonexty',
                showlegend=False
            )
            
            # Update layout for better readability and autoscaling
            fig_combined.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=600,  # Increased height for better spacing
                margin=dict(t=100, b=50),  # Increased margins
                yaxis=dict(
                    autorange=True,  # Enable autoscaling
                    rangemode='tozero'  # Start y-axis from zero
                )
            )
            
            st.plotly_chart(fig_combined)
            
            # Clear memory after plotting
            del fig_combined, model, forecast
            gc.collect()

        except Exception as e:
            st.error(f"Error creating volume analysis: {str(e)}")

        # --- Cost Breakdown Visualization ---
        st.subheader("💸 Cost Breakdown by Charge Type")

        try:
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
            
            # Clear memory after plotting
            del fig_cost_breakdown
            gc.collect()

        except Exception as e:
            st.error(f"Error creating cost breakdown: {str(e)}")

        # --- AI Insights ---
        st.header("🤖 AI Insights & Suggestions")
        st.write("Ask the AI for shipment insights or profit optimization strategies.")

        # Sample Question and Answer
        st.subheader("💡 Sample Question")
        sample_question = "What insights can we derive from the data to improve profitability and reduce delivery delays?"
        st.write(f"**Question:** {sample_question}")

        sample_answer = \"\"\"
        Based on historical data analysis:
        1. **Carrier Performance**: Certain carriers have consistently higher profit margins but also higher delays. Adjusting carrier contracts can optimize both profitability and on-time deliveries.
        2. **High-Volume Customers**: Focus on customers with high shipment volumes and profit margins to offer loyalty discounts and increase retention.
        3. **Seasonal Trends**: Profitability spikes during specific months. Align marketing and operations to capitalize on these trends.
        4. **Zone-Based Pricing**: Implement zone-based pricing adjustments to optimize costs in regions with lower margins.
        \"\"\"
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
        """

        # Final memory cleanup and logging
        gc.collect()
        st.write(f"Final memory usage: {get_memory_usage():.1f}MB")

    except Exception as e:
        st.error(f"Critical dashboard error: {str(e)}\n{traceback.format_exc()}")
