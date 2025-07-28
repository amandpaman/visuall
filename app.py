import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Telecom Data Analyzer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .filter-section {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .status-green { color: #28a745; font-weight: bold; }
    .status-red { color: #dc3545; font-weight: bold; }
    .status-yellow { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_excel_data(uploaded_file):
    """Load and cache Excel data"""
    try:
        # Try to read all sheets
        excel_file = pd.ExcelFile(uploaded_file)
        sheets = {}
        for sheet_name in excel_file.sheet_names:
            sheets[sheet_name] = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        return sheets
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None

def clean_numeric_columns(df, columns):
    """Clean and convert columns to numeric"""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def parse_dates(df, date_columns):
    """Parse date columns"""
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def get_safe_column_values(df, column_name, default_value='N/A'):
    """Safely get unique values from a column"""
    if column_name in df.columns:
        return sorted(df[column_name].dropna().unique().tolist())
    else:
        return [default_value]

def safe_column_access(df, column_name, default_value=None):
    """Safely access a column, return default if column doesn't exist"""
    if column_name in df.columns:
        return df[column_name]
    else:
        return pd.Series([default_value] * len(df))

def get_rag_color(status):
    """Get color for RAG status"""
    if pd.isna(status):
        return 'gray'
    status = str(status).upper()
    if 'GREEN' in status or 'CONNECTED' in status:
        return 'green'
    elif 'RED' in status or 'NOT CONNECTED' in status:
        return 'red'
    elif 'YELLOW' in status or 'ON HOLD' in status:
        return 'orange'
    else:
        return 'blue'

# Main App
def main():
    st.title("📡 Advanced Telecom Data Analyzer")
    st.markdown("---")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📂 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=['xlsx', 'xls', 'xlsm'],
            help="Upload your telecom data Excel file"
        )
        
        if uploaded_file is not None:
            sheets = load_excel_data(uploaded_file)
            
            if sheets:
                # Sheet selection
                sheet_names = list(sheets.keys())
                selected_sheet = st.selectbox("Select Sheet", sheet_names)
                df = sheets[selected_sheet].copy()
                
                st.success(f"✅ Loaded {len(df)} records from '{selected_sheet}'")
                st.info(f"📊 {len(df.columns)} columns available")
            else:
                st.error("❌ Failed to load Excel file")
                return
        else:
            st.info("👆 Please upload an Excel file to begin analysis")
            return
    
    # Data preprocessing
    with st.spinner("Processing data..."):
        # Display available columns for debugging
        with st.expander("📋 Available Columns in Your Data", expanded=False):
            st.write("**Column Names:**")
            col_df = pd.DataFrame({
                'Column Name': df.columns.tolist(),
                'Data Type': [str(dtype) for dtype in df.dtypes],
                'Non-Null Count': [df[col].notna().sum() for col in df.columns],
                'Sample Value': [str(df[col].iloc[0]) if len(df) > 0 and df[col].notna().any() else 'N/A' for col in df.columns]
            })
            st.dataframe(col_df, use_container_width=True)
        
        # Identify numeric columns for CAPEX and financial data
        numeric_cols = ['OTC', 'ARC', 'OB VALUE', 'LM CAPEX', 'NETWORK CAPEX', 
                       'CPE CAPEX', 'OFFNET OTC', 'OFFNET ARC', 'POLE CAPEX',
                       'TRANSPORTATION CAPEX', 'GI STRIP CAPEX', 'MATERIAL CAPEX',
                       'ON HOLD DAYS', 'INCLUDING ON HOLD', 'EXCLUDING ON HOLD',
                       'OVERALL AGING', 'LATTITUDE', 'LONGITUDE']
        
        df = clean_numeric_columns(df, numeric_cols)
        
        # Parse date columns
        date_cols = ['WO START DATE', 'WO COMMISSION DATE', 'CUSTOMER ACCEPTANCE DATE']
        df = parse_dates(df, date_cols)
    
    # Main dashboard
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🗺️ Geographic", "💰 Financial", "⏱️ Timeline", "🔧 Technical", "📈 Advanced Analytics"
    ])
    
    # Tab 1: Overview Dashboard
    with tab1:
        st.header("📊 Executive Dashboard")
        
        # Filters
        with st.expander("🔍 Filters", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                zones = ['All'] + get_safe_column_values(df, 'ZONE')
                selected_zone = st.selectbox("Zone", zones)
                
            with col2:
                circles = ['All'] + get_safe_column_values(df, 'A END CIRCLE')
                selected_circle = st.selectbox("Circle", circles)
                
            with col3:
                statuses = ['All'] + get_safe_column_values(df, 'CONNECTED STATUS')
                selected_status = st.selectbox("Connection Status", statuses)
                
            with col4:
                customers = ['All'] + get_safe_column_values(df, 'CUSTOMER NAME')
                selected_customer = st.selectbox("Customer", customers)
        
        # Apply filters
        filtered_df = df.copy()
        if selected_zone != 'All' and 'ZONE' in df.columns:
            filtered_df = filtered_df[filtered_df['ZONE'] == selected_zone]
        if selected_circle != 'All' and 'A END CIRCLE' in df.columns:
            filtered_df = filtered_df[filtered_df['A END CIRCLE'] == selected_circle]
        if selected_status != 'All' and 'CONNECTED STATUS' in df.columns:
            filtered_df = filtered_df[filtered_df['CONNECTED STATUS'] == selected_status]
        if selected_customer != 'All' and 'CUSTOMER NAME' in df.columns:
            filtered_df = filtered_df[filtered_df['CUSTOMER NAME'] == selected_customer]
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_orders = len(filtered_df)
            st.metric("Total Orders", f"{total_orders:,}")
        
        with col2:
            if 'CONNECTED STATUS' in filtered_df.columns:
                connected = len(filtered_df[filtered_df['CONNECTED STATUS'].str.contains('Connected', na=False)])
                connection_rate = (connected / total_orders * 100) if total_orders > 0 else 0
                st.metric("Connected", f"{connected:,}", f"{connection_rate:.1f}%")
            else:
                st.metric("Connected", "N/A", "Data not available")
        
        with col3:
            if any(col in filtered_df.columns for col in ['LM CAPEX', 'NETWORK CAPEX', 'CPE CAPEX', 'POLE CAPEX']):
                capex_columns = [col for col in ['LM CAPEX', 'NETWORK CAPEX', 'CPE CAPEX', 'POLE CAPEX'] if col in filtered_df.columns]
                total_capex = filtered_df[capex_columns].sum().sum()
                st.metric("Total CAPEX", f"₹{total_capex:,.0f}")
            else:
                st.metric("Total CAPEX", "N/A", "Data not available")
        
        with col4:
            if 'OVERALL AGING' in filtered_df.columns:
                avg_aging = filtered_df['OVERALL AGING'].mean()
                st.metric("Avg Aging (Days)", f"{avg_aging:.0f}" if not pd.isna(avg_aging) else "N/A")
            else:
                st.metric("Avg Aging (Days)", "N/A", "Data not available")
        
        with col5:
            if 'CONNECTED STATUS' in filtered_df.columns:
                on_hold = len(filtered_df[filtered_df['CONNECTED STATUS'].str.contains('Hold', na=False)])
                st.metric("On Hold", f"{on_hold:,}")
            else:
                st.metric("On Hold", "N/A", "Data not available")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Status Distribution
            if 'CONNECTED STATUS' in filtered_df.columns:
                status_counts = filtered_df['CONNECTED STATUS'].value_counts()
                fig_status = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Connection Status Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_status, use_container_width=True)
            else:
                st.warning("CONNECTED STATUS column not found")
        
        with col2:
            # Zone-wise Orders
            if 'ZONE' in filtered_df.columns:
                zone_counts = filtered_df['ZONE'].value_counts()
                fig_zone = px.bar(
                    x=zone_counts.index,
                    y=zone_counts.values,
                    title="Orders by Zone",
                    labels={'x': 'Zone', 'y': 'Order Count'},
                    color=zone_counts.values,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_zone, use_container_width=True)
            else:
                st.warning("ZONE column not found")
        
        # RAG Status Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            if 'RAG STATUS' in filtered_df.columns:
                rag_counts = filtered_df['RAG STATUS'].value_counts()
                fig_rag = px.bar(
                    x=rag_counts.index,
                    y=rag_counts.values,
                    title="RAG Status Distribution",
                    labels={'x': 'RAG Status', 'y': 'Count'},
                    color=rag_counts.index,
                    color_discrete_map={
                        'GREEN': 'green',
                        'RED': 'red',
                        'YELLOW': 'orange',
                        'CAPTIVE': 'blue'
                    }
                )
                st.plotly_chart(fig_rag, use_container_width=True)
            else:
                st.warning("RAG STATUS column not found")
        
        with col2:
            # Customer-wise Orders
            if 'CUSTOMER NAME' in filtered_df.columns:
                customer_counts = filtered_df['CUSTOMER NAME'].value_counts().head(10)
                fig_customer = px.bar(
                    x=customer_counts.values,
                    y=customer_counts.index,
                    orientation='h',
                    title="Top 10 Customers by Order Count",
                    labels={'x': 'Order Count', 'y': 'Customer'},
                    color=customer_counts.values,
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig_customer, use_container_width=True)
            else:
                st.warning("CUSTOMER NAME column not found")
    
    # Tab 2: Geographic Analysis
    with tab2:
        st.header("🗺️ Geographic Analysis")
        
        # Map filters
        col1, col2, col3 = st.columns(3)
        with col1:
            available_color_cols = [col for col in ['CONNECTED STATUS', 'RAG STATUS', 'ZONE', 'A END CIRCLE', 'CUSTOMER NAME'] if col in filtered_df.columns]
            if available_color_cols:
                map_color_by = st.selectbox("Color Points By", available_color_cols)
            else:
                map_color_by = None
                st.warning("No suitable columns found for coloring")
                
        with col2:
            available_size_cols = [col for col in ['OB VALUE', 'LM CAPEX', 'OVERALL AGING'] if col in filtered_df.columns]
            if available_size_cols:
                map_size_by = st.selectbox("Size Points By", ['None'] + available_size_cols)
            else:
                map_size_by = 'None'
                
        with col3:
            show_only_connected = st.checkbox("Show Only Valid Coordinates")
        
        # Prepare map data
        map_df = filtered_df.copy()
        
        # Check if coordinate columns exist
        if 'LATTITUDE' not in map_df.columns or 'LONGITUDE' not in map_df.columns:
            st.error("Latitude and Longitude columns not found in the data. Geographic analysis is not available.")
            st.info("Required columns: 'LATTITUDE', 'LONGITUDE'")
            return
            
        if show_only_connected:
            map_df = map_df.dropna(subset=['LATTITUDE', 'LONGITUDE'])
            map_df = map_df[(map_df['LATTITUDE'] != 0) & (map_df['LONGITUDE'] != 0)]
        
        if len(map_df) > 0 and map_color_by:
            # Create hover data - only include columns that exist
            hover_cols = ['CIRCUIT ID', 'CUSTOMER NAME', 'AREA', 'CONNECTED STATUS']
            available_hover_cols = [col for col in hover_cols if col in map_df.columns]
            
            # Create map
            fig_map = px.scatter_mapbox(
                map_df,
                lat='LATTITUDE',
                lon='LONGITUDE',
                color=map_color_by,
                size=map_size_by if map_size_by != 'None' else None,
                hover_data=available_hover_cols,
                title=f"Geographic Distribution of Orders",
                mapbox_style='open-street-map',
                height=600,
                zoom=6
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
            # Geographic summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Orders with Coordinates", len(map_df))
            with col2:
                if 'AREA' in map_df.columns:
                    unique_areas = map_df['AREA'].nunique()
                    st.metric("Unique Areas", unique_areas)
                else:
                    st.metric("Unique Areas", "N/A")
            with col3:
                if 'DISTRICT' in map_df.columns:
                    unique_districts = map_df['DISTRICT'].nunique()
                    st.metric("Unique Districts", unique_districts)
                else:
                    st.metric("Unique Districts", "N/A")
        else:
            st.warning("No valid geographic coordinates found in filtered data or no color column available")
    
    # Tab 3: Financial Analysis
    with tab3:
        st.header("💰 Financial Analysis")
        
        # CAPEX Analysis
        capex_cols = ['LM CAPEX', 'NETWORK CAPEX', 'CPE CAPEX', 'POLE CAPEX', 
                     'TRANSPORTATION CAPEX', 'GI STRIP CAPEX', 'MATERIAL CAPEX']
        
        available_capex_cols = [col for col in capex_cols if col in filtered_df.columns]
        
        if available_capex_cols:
            # Total CAPEX by type
            capex_totals = filtered_df[available_capex_cols].sum()
            capex_totals = capex_totals[capex_totals > 0]  # Remove zero values
            
            col1, col2 = st.columns(2)
            
            with col1:
                if len(capex_totals) > 0:
                    fig_capex = px.pie(
                        values=capex_totals.values,
                        names=capex_totals.index,
                        title="CAPEX Distribution by Type",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig_capex, use_container_width=True)
            
            with col2:
                # CAPEX by Zone
                zone_capex = filtered_df.groupby('ZONE')[available_capex_cols].sum().sum(axis=1)
                zone_capex = zone_capex[zone_capex > 0]
                
                if len(zone_capex) > 0:
                    fig_zone_capex = px.bar(
                        x=zone_capex.index,
                        y=zone_capex.values,
                        title="Total CAPEX by Zone",
                        labels={'x': 'Zone', 'y': 'Total CAPEX (₹)'},
                        color=zone_capex.values,
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_zone_capex, use_container_width=True)
            
            # Financial metrics table
            st.subheader("📊 Financial Summary")
            financial_summary = pd.DataFrame({
                'CAPEX Type': available_capex_cols,
                'Total Amount (₹)': [filtered_df[col].sum() for col in available_capex_cols],
                'Average per Order (₹)': [filtered_df[col].mean() for col in available_capex_cols],
                'Max Amount (₹)': [filtered_df[col].max() for col in available_capex_cols],
                'Orders with Value': [filtered_df[col].notna().sum() for col in available_capex_cols]
            })
            financial_summary = financial_summary[financial_summary['Total Amount (₹)'] > 0]
            st.dataframe(financial_summary, use_container_width=True)
        
        # OTC and ARC Analysis
        if 'OTC' in filtered_df.columns and 'ARC' in filtered_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                otc_data = filtered_df['OTC'].dropna()
                if len(otc_data) > 0:
                    fig_otc = px.histogram(
                        otc_data,
                        title="OTC Distribution",
                        nbins=20,
                        labels={'value': 'OTC Amount', 'count': 'Frequency'}
                    )
                    st.plotly_chart(fig_otc, use_container_width=True)
            
            with col2:
                arc_data = filtered_df['ARC'].dropna()
                if len(arc_data) > 0:
                    fig_arc = px.histogram(
                        arc_data,
                        title="ARC Distribution",
                        nbins=20,
                        labels={'value': 'ARC Amount', 'count': 'Frequency'}
                    )
                    st.plotly_chart(fig_arc, use_container_width=True)
    
    # Tab 4: Timeline Analysis
    with tab4:
        st.header("⏱️ Timeline & Aging Analysis")
        
        # Aging Analysis
        aging_cols = ['ON HOLD DAYS', 'INCLUDING ON HOLD', 'EXCLUDING ON HOLD', 'OVERALL AGING']
        available_aging_cols = [col for col in aging_cols if col in filtered_df.columns]
        
        if available_aging_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                # Aging distribution
                aging_data = filtered_df['OVERALL AGING'].dropna()
                if len(aging_data) > 0:
                    fig_aging = px.histogram(
                        aging_data,
                        title="Overall Aging Distribution",
                        nbins=30,
                        labels={'value': 'Days', 'count': 'Order Count'}
                    )
                    st.plotly_chart(fig_aging, use_container_width=True)
            
            with col2:
                # Aging by status
                aging_by_status = filtered_df.groupby('CONNECTED STATUS')['OVERALL AGING'].mean()
                aging_by_status = aging_by_status.dropna()
                
                if len(aging_by_status) > 0:
                    fig_aging_status = px.bar(
                        x=aging_by_status.index,
                        y=aging_by_status.values,
                        title="Average Aging by Connection Status",
                        labels={'x': 'Status', 'y': 'Average Days'},
                        color=aging_by_status.values,
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_aging_status, use_container_width=True)
        
        # Timeline Analysis with dates
        date_cols_available = [col for col in date_cols if col in filtered_df.columns]
        
        if date_cols_available:
            st.subheader("📅 Timeline Analysis")
            
            # Order timeline
            timeline_df = filtered_df.copy()
            
            for col in date_cols_available:
                timeline_df[col] = pd.to_datetime(timeline_df[col])
            
            # Orders over time
            if 'WO START DATE' in timeline_df.columns:
                monthly_orders = timeline_df.groupby(timeline_df['WO START DATE'].dt.to_period('M')).size()
                
                if len(monthly_orders) > 0:
                    fig_timeline = px.line(
                        x=monthly_orders.index.astype(str),
                        y=monthly_orders.values,
                        title="Orders Over Time (Monthly)",
                        labels={'x': 'Month', 'y': 'Order Count'}
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
        
        # WSLA/OSLA Analysis
        if 'WSLA/OSLA' in filtered_df.columns:
            wsla_counts = filtered_df['WSLA/OSLA'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                fig_wsla = px.pie(
                    values=wsla_counts.values,
                    names=wsla_counts.index,
                    title="WSLA/OSLA Distribution"
                )
                st.plotly_chart(fig_wsla, use_container_width=True)
    
    # Tab 5: Technical Analysis
    with tab5:
        st.header("🔧 Technical Analysis")
        
        # Device and Equipment Analysis
        technical_cols = ['LM SERVICE PROVIDER', 'LM TYPE', 'LM MEDIA', 'DEVICE TYPE', 
                         'DEVICE MAKE', 'DEVICE MODEL', 'MANAGED FLAVOUR']
        
        available_tech_cols = [col for col in technical_cols if col in filtered_df.columns]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'DEVICE MAKE' in filtered_df.columns:
                device_makes = filtered_df['DEVICE MAKE'].value_counts().head(10)
                fig_devices = px.bar(
                    x=device_makes.values,
                    y=device_makes.index,
                    orientation='h',
                    title="Top Device Makes",
                    labels={'x': 'Count', 'y': 'Device Make'}
                )
                st.plotly_chart(fig_devices, use_container_width=True)
        
        with col2:
            if 'LM SERVICE PROVIDER' in filtered_df.columns:
                providers = filtered_df['LM SERVICE PROVIDER'].value_counts()
                fig_providers = px.pie(
                    values=providers.values,
                    names=providers.index,
                    title="Last Mile Service Providers"
                )
                st.plotly_chart(fig_providers, use_container_width=True)
        
        # Technical summary table
        if available_tech_cols:
            st.subheader("📋 Technical Configuration Summary")
            
            tech_summary = {}
            for col in available_tech_cols:
                unique_values = filtered_df[col].nunique()
                most_common = filtered_df[col].mode().iloc[0] if len(filtered_df[col].mode()) > 0 else 'N/A'
                tech_summary[col] = {
                    'Unique Values': unique_values,
                    'Most Common': most_common,
                    'Data Availability': f"{filtered_df[col].notna().sum()}/{len(filtered_df)}"
                }
            
            tech_df = pd.DataFrame(tech_summary).T
            st.dataframe(tech_df, use_container_width=True)
    
    # Tab 6: Advanced Analytics
    with tab6:
        st.header("📈 Advanced Analytics")
        
        # Correlation Analysis
        st.subheader("🔗 Correlation Analysis")
        
        numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) >= 2:
            selected_vars = st.multiselect(
                "Select variables for correlation analysis",
                numeric_columns,
                default=numeric_columns[:5] if len(numeric_columns) >= 5 else numeric_columns
            )
            
            if len(selected_vars) >= 2:
                corr_matrix = filtered_df[selected_vars].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    title="Correlation Matrix",
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        
        # Performance Analysis
        st.subheader("⚡ Performance Metrics")
        
        # Create performance score
        if 'OVERALL AGING' in filtered_df.columns and 'CONNECTED STATUS' in filtered_df.columns:
            performance_df = filtered_df.copy()
            
            # Calculate performance score (lower aging = better performance)
            max_aging = performance_df['OVERALL AGING'].max()
            performance_df['Performance Score'] = 100 - (performance_df['OVERALL AGING'] / max_aging * 100)
            
            # Performance by circle
            perf_by_circle = performance_df.groupby('A END CIRCLE')['Performance Score'].mean().sort_values(ascending=False)
            
            if len(perf_by_circle) > 0:
                fig_perf = px.bar(
                    x=perf_by_circle.index,
                    y=perf_by_circle.values,
                    title="Performance Score by Circle",
                    labels={'x': 'Circle', 'y': 'Performance Score'},
                    color=perf_by_circle.values,
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_perf, use_container_width=True)
        
        # Data Export
        st.subheader("💾 Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Filtered Data"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"telecom_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Export Summary Report"):
                summary_data = {
                    'Metric': ['Total Orders', 'Connected Orders', 'Average Aging', 'Total CAPEX'],
                    'Value': [
                        len(filtered_df),
                        len(filtered_df[filtered_df['CONNECTED STATUS'].str.contains('Connected', na=False)]),
                        filtered_df['OVERALL AGING'].mean() if 'OVERALL AGING' in filtered_df.columns else 0,
                        filtered_df[available_capex_cols].sum().sum() if available_capex_cols else 0
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Summary",
                    data=csv,
                    file_name=f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            st.info(f"📋 Current Filter: {len(filtered_df):,} of {len(df):,} records")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            📡 Advanced Telecom Data Analyzer | Built with Streamlit & Plotly
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
