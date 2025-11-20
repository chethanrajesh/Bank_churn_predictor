# src/streamlit_dashboard.py
"""
AUTOMATED 5-PAGE STREAMLIT DASHBOARD
Beautiful visualizations of churn prediction data
"""

import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Bank Churn Prediction Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Page background */
.reportview-container, .main {
    background-color: #f8fbff;
}

/* Global text color */
[data-testid="stAppViewContainer"] {
    color: #0f172a;
}

/* Headings */
h1, h2, h3 {
    color: #0b62d6;
    font-weight: 600;
}

/* Sidebar background and title */
div[data-testid="stSidebar"] {
    background-color: #f0f5ff;
    padding-top: 12px;
}
div[data-testid="stSidebar"] .css-1d391kg {  /* sidebar title selector may vary by Streamlit version */
    color: #0b62d6 !important;
}

/* Option-menu / radio items (generic) */
.stRadio button, .stRadio label, .stRadio div {
    color: #0b62d6 !important;
}

/* Buttons */
.stButton>button {
    background-color: #0b62d6;
    color: white;
    border-radius: 8px;
    padding: 6px 12px;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #1748d9;
}

/* Metric styling */
div[data-testid="stMetricValue"] {
    color: #062c7a;
    font-weight: 700;
}

/* Cards / small panels */
.card {
    background: white;
    border-radius: 10px;
    padding: 12px;
    box-shadow: 0 1px 4px rgba(11,98,214,0.06);
}

/* Tables / dataframes: subtle rounded corners */
.stDataFrameContainer .dataframe, .stDataFrame {
    border-radius: 8px;
    overflow: hidden;
}

/* Expander header color */
.streamlit-expanderHeader {
    color: #0b62d6 !important;
    font-weight: 600;
}

/* Plot background neutral (for Plotly) */
.main .plotly-graph-div .plotly {
    background: transparent !important;
}

/* Small utility: hide default hamburger menu and footer if you want */
#MainMenu { visibility: visible; }   /* change to hidden to hide main menu */
footer { visibility: visible; }      /* change to hidden to hide Streamlit footer */
</style>
""", unsafe_allow_html=True)

class ChurnDashboard:
    def __init__(self):
        self.data_dir = Path("../powerbi/data")
        self.load_data()
        
    def load_data(self):
        """Load all data files"""
        try:
            self.predictions_df = pd.read_csv(self.data_dir / "predictions_data.csv")
            self.model_metrics_df = pd.read_csv(self.data_dir / "model_metrics.csv")
            self.shap_df = pd.read_csv(self.data_dir / "shap_values_summary.csv")
            self.driver_df = pd.read_csv(self.data_dir / "driver_breakdown.csv")
            self.segments_df = pd.read_csv(self.data_dir / "customer_segments.csv")
            self.individual_shap_df = pd.read_csv(self.data_dir / "individual_shap_values.csv")
            
            # Calculate additional metrics
            self.total_customers = len(self.predictions_df)
            self.high_risk_count = len(self.predictions_df[self.predictions_df['risk_segment'] == 'High Risk'])
            self.avg_churn_prob = self.predictions_df['churn_probability'].mean()
            self.total_expected_loss = self.predictions_df['expected_loss'].sum()
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    def page_executive_overview(self):
        """Page 1: Executive Overview"""
        st.title("🏦 Executive Overview")
        st.markdown("### Bank Churn Prediction Dashboard")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Customers",
                value=f"{self.total_customers:,}",
                delta=f"{self.high_risk_count} high risk"
            )
        
        with col2:
            st.metric(
                label="High Risk Customers",
                value=f"{self.high_risk_count}",
                delta=f"{(self.high_risk_count/self.total_customers*100):.1f}%"
            )
        
        with col3:
            auc_score = self.model_metrics_df[
                self.model_metrics_df['metric_name'] == 'AUC Score'
            ]['metric_value'].iloc[0] if not self.model_metrics_df.empty else 0
            st.metric(
                label="Model AUC Score",
                value=f"{auc_score:.3f}",
                delta="Excellent" if auc_score > 0.8 else "Good"
            )
        
        with col4:
            st.metric(
                label="Expected Loss",
                value=f"${self.total_expected_loss:,.0f}",
                delta=f"${(self.total_expected_loss * 0.3):,.0f} savings potential"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Distribution Pie Chart
            risk_counts = self.predictions_df['risk_segment'].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Customer Risk Distribution",
                color=risk_counts.index,
                color_discrete_map={
                    'High Risk': '#FF6B6B',
                    'Medium Risk': '#FFD166', 
                    'Low Risk': '#06D6A0'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Churn Probability Distribution
            fig_hist = px.histogram(
                self.predictions_df,
                x='churn_probability',
                nbins=50,
                title="Churn Probability Distribution",
                color_discrete_sequence=['#118AB2']
            )
            fig_hist.add_vline(x=self.avg_churn_prob, line_dash="dash", line_color="red",
                             annotation_text=f"Avg: {self.avg_churn_prob:.2f}")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Top Drivers Summary
        st.subheader("🎯 Top Churn Drivers")
        top_drivers = self.shap_df.nlargest(5, 'shap_importance')
        
        for idx, row in top_drivers.iterrows():
            driver_col, importance_col = st.columns([3, 1])
            with driver_col:
                st.write(f"**{row['feature_name']}**")
            with importance_col:
                st.metric("Importance", f"{row['shap_importance']:.4f}")
    
    def page_risk_analysis(self):
        """Page 2: Risk Analysis - Heavily Optimized for 100k+ Data"""
        st.title("🔍 Risk Analysis")
        st.markdown("### Customer Segmentation & Risk Profiling")
        
        # Pre-calculate everything needed for this page
        with st.spinner("Optimizing data for analysis..."):
            # Use pre-aggregated data for initial display
            @st.cache_data(ttl=3600)
            def get_aggregated_data(_df):
                # Pre-aggregate risk distribution
                risk_dist = _df['risk_segment'].value_counts().reset_index()
                risk_dist.columns = ['risk_segment', 'count']
                
                # Pre-calculate summary stats
                summary_stats = _df.groupby('risk_segment').agg({
                    'churn_probability': ['mean', 'median', 'std'],
                    'expected_loss': ['sum', 'mean']
                }).round(4)
                
                # Flatten column names
                summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
                summary_stats = summary_stats.reset_index()
                
                return risk_dist, summary_stats
            
            risk_dist, summary_stats = get_aggregated_data(self.predictions_df)
        
        # Quick filters in sidebar for better performance
        st.sidebar.subheader("🔧 Risk Analysis Filters")
        
        # Use session state to avoid re-filtering on every interaction
        if 'risk_filters' not in st.session_state:
            st.session_state.risk_filters = {
                'selected_risk': ['High Risk', 'Medium Risk'],
                'sample_size': 1000
            }
        
        # Simple binary filter for high risk only
        show_only_high_risk = st.sidebar.checkbox(
            "Show Only High Risk Customers", 
            value=False,
            help="Focus on highest risk segment for better performance"
        )
        
        # Sample size control
        sample_size = st.sidebar.select_slider(
            "Visualization Sample Size",
            options=[500, 1000, 2000, 5000, 10000],
            value=1000,
            help="Smaller samples load faster"
        )
        
        # Apply filters
        if show_only_high_risk:
            filtered_df = self.predictions_df[self.predictions_df['risk_segment'] == 'High Risk']
            if len(filtered_df) > sample_size:
                filtered_df = filtered_df.sample(n=sample_size, random_state=42)
        else:
            # Sample from full dataset
            if len(self.predictions_df) > sample_size:
                filtered_df = self.predictions_df.sample(n=sample_size, random_state=42)
            else:
                filtered_df = self.predictions_df
        
        st.sidebar.info(f"Showing {len(filtered_df):,} customers")
        
        # Main content - use tabs to avoid loading all charts at once
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Risk Distribution", "🚨 High Risk Details"])
        
        with tab1:
            self._render_overview_tab(filtered_df, risk_dist, summary_stats)
        
        with tab2:
            self._render_distribution_tab(filtered_df)
        
        with tab3:
            self._render_high_risk_tab(filtered_df)

    def _render_overview_tab(self, filtered_df, risk_dist, summary_stats):
        """Render overview tab with fast-loading content"""
        st.subheader("📊 Quick Overview")
        
        # KPI Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(filtered_df)
            st.metric("Customers in View", f"{total_customers:,}")
        
        with col2:
            high_risk_pct = len(filtered_df[filtered_df['risk_segment'] == 'High Risk']) / len(filtered_df) * 100
            st.metric("High Risk %", f"{high_risk_pct:.1f}%")
        
        with col3:
            avg_churn = filtered_df['churn_probability'].mean()
            st.metric("Avg Churn Probability", f"{avg_churn:.2%}")
        
        with col4:
            total_loss = filtered_df['expected_loss'].sum()
            st.metric("Total Expected Loss", f"${total_loss:,.0f}")
        
        # Fast charts - use pre-aggregated data
        col1, col2 = st.columns(2)
        
        with col1:
            # Simple bar chart from pre-aggregated data
            fig_bar = px.bar(
                risk_dist,
                x='risk_segment',
                y='count',
                title="Customer Distribution by Risk Segment",
                color='risk_segment',
                color_discrete_map={
                    'High Risk': '#FF6B6B',
                    'Medium Risk': '#FFD166',
                    'Low Risk': '#06D6A0'
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Box plot for churn probability distribution (faster than scatter)
            fig_box = px.box(
                filtered_df,
                x='risk_segment',
                y='churn_probability',
                title="Churn Probability Distribution by Risk",
                color='risk_segment',
                color_discrete_map={
                    'High Risk': '#FF6B6B',
                    'Medium Risk': '#FFD166',
                    'Low Risk': '#06D6A0'
                }
            )
            st.plotly_chart(fig_box, use_container_width=True)

    def _render_distribution_tab(self, filtered_df):
        """Render distribution analysis tab"""
        st.subheader("📈 Risk Distribution Analysis")
        
        # Use hexbin or density plot instead of scatter for large data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Churn Probability vs Expected Loss**")
            
            # For large datasets, use a 2D histogram
            if len(filtered_df) > 1000:
                fig_hist = px.density_heatmap(
                    filtered_df,
                    x='churn_probability',
                    y='expected_loss',
                    title="Risk Density Heatmap",
                    nbinsx=20,
                    nbinsy=20,
                    color_continuous_scale='Viridis'
                )
            else:
                # For smaller samples, use scatter with opacity
                fig_hist = px.scatter(
                    filtered_df,
                    x='churn_probability',
                    y='expected_loss',
                    color='risk_segment',
                    opacity=0.6,
                    title="Churn Probability vs Expected Loss",
                    color_discrete_map={
                        'High Risk': '#FF6B6B',
                        'Medium Risk': '#FFD166',
                        'Low Risk': '#06D6A0'
                    }
                )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.markdown("**Cumulative Risk Distribution**")
            
            # Create cumulative distribution
            sorted_probs = np.sort(filtered_df['churn_probability'])
            cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
            
            fig_cumulative = go.Figure()
            fig_cumulative.add_trace(go.Scatter(
                x=sorted_probs,
                y=cumulative,
                mode='lines',
                name='Cumulative Distribution',
                line=dict(color='#118AB2', width=3)
            ))
            
            fig_cumulative.update_layout(
                title="Cumulative Churn Probability Distribution",
                xaxis_title="Churn Probability",
                yaxis_title="Cumulative Proportion",
                showlegend=False
            )
            st.plotly_chart(fig_cumulative, use_container_width=True)

    def _render_high_risk_tab(self, filtered_df):
        """Render high risk customer details"""
        st.subheader("🚨 High Risk Customer Analysis")
        
        # Get high risk customers
        high_risk_df = filtered_df[filtered_df['risk_segment'] == 'High Risk']
        
        if high_risk_df.empty:
            st.info("No high-risk customers in current view")
            return
        
        # Top 20 high risk customers by default
        top_high_risk = high_risk_df.nlargest(20, 'churn_probability')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Top High Risk Customers**")
            
            # Simple table without heavy formatting
            display_data = top_high_risk[[
                'customer_id', 'churn_probability', 'expected_loss'
            ]].round(4)
            
            st.dataframe(
                display_data,
                use_container_width=True,
                height=400
            )
        
        with col2:
            st.markdown("**High Risk Summary**")
            
            st.metric(
                "Total High Risk", 
                f"{len(high_risk_df):,}",
                delta=f"{(len(high_risk_df)/len(filtered_df)*100):.1f}%"
            )
            
            avg_high_risk_prob = high_risk_df['churn_probability'].mean()
            st.metric("Avg Probability", f"{avg_high_risk_prob:.2%}")
            
            total_high_risk_loss = high_risk_df['expected_loss'].sum()
            st.metric("Total Loss at Risk", f"${total_high_risk_loss:,.0f}")
            
            # Quick actions
            st.markdown("**💡 Quick Actions**")
            if st.button("📥 Export High Risk List", key="export_high_risk"):
                csv = high_risk_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="high_risk_customers.csv",
                    mime="text/csv",
                    key="download_high_risk"
                )
    
    def page_driver_analysis(self):
        """Page 3: Driver Analysis"""
        st.title("🎯 Driver Analysis")
        st.markdown("### Feature Importance & Churn Drivers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top Features by SHAP Importance
            top_n = st.slider("Number of top features to show", 5, 20, 10)
            top_features = self.shap_df.nlargest(top_n, 'shap_importance')
            
            fig_bar = px.bar(
                top_features,
                x='shap_importance',
                y='feature_name',
                orientation='h',
                title=f"Top {top_n} Features by SHAP Importance",
                color='shap_importance',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Driver Category Breakdown
            if not self.driver_df.empty:
                # Check what columns are available in driver_df
                available_columns = self.driver_df.columns.tolist()
                st.write(f"Available columns: {available_columns}")  # Debug info
                
                # Use available importance column
                importance_column = 'normalized_importance' if 'normalized_importance' in self.driver_df.columns else 'absolute_importance'
                
                fig_treemap = px.treemap(
                    self.driver_df,
                    path=['driver_category'],
                    values=importance_column,
                    title="Driver Category Importance",
                    color=importance_column,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_treemap, use_container_width=True)
        
        # Waterfall Chart for Feature Impact
        st.subheader("📊 Feature Impact Analysis")
        waterfall_data = self.shap_df.nlargest(15, 'shap_importance').sort_values('shap_importance')
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="Feature Impact",
            orientation="h",
            y=waterfall_data['feature_name'],
            x=waterfall_data['shap_importance'],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig_waterfall.update_layout(
            title="Feature Impact Waterfall Chart",
            showlegend=False
        )
        
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # Driver Details
        st.subheader("🔍 Driver Category Details")
        
        # Debug: Show actual column names
        st.write(f"Driver DataFrame columns: {self.driver_df.columns.tolist()}")
        st.write(f"Driver DataFrame sample:")
        st.dataframe(self.driver_df.head())
        
        for _, driver in self.driver_df.iterrows():
            with st.expander(f"{driver['driver_category']} - Importance: {self.get_driver_importance(driver):.4f}"):
                # Display available metrics safely
                if 'feature_count' in driver and pd.notna(driver['feature_count']):
                    st.write(f"**Features in category:** {driver['feature_count']}")
                
                if 'normalized_importance' in driver and pd.notna(driver['normalized_importance']):
                    st.write(f"**Normalized Importance:** {driver['normalized_importance']:.4f}")
                
                if 'absolute_importance' in driver and pd.notna(driver['absolute_importance']):
                    st.write(f"**Absolute Importance:** {driver['absolute_importance']:.4f}")
                
                if 'shap_importance' in driver and pd.notna(driver['shap_importance']):
                    st.write(f"**SHAP Importance:** {driver['shap_importance']:.4f}")
                
                # Show top features if available
                if 'top_features' in driver and pd.notna(driver['top_features']):
                    st.write(f"**Top features:** {driver['top_features']}")
                elif 'features' in driver and pd.notna(driver['features']):
                    st.write(f"**Features:** {driver['features']}")

    def get_driver_importance(self, driver):
        """Helper method to get importance value from available columns"""
        if 'normalized_importance' in driver and pd.notna(driver['normalized_importance']):
            return driver['normalized_importance']
        elif 'absolute_importance' in driver and pd.notna(driver['absolute_importance']):
            return driver['absolute_importance']
        elif 'shap_importance' in driver and pd.notna(driver['shap_importance']):
            return driver['shap_importance']
        elif 'total_importance' in driver and pd.notna(driver['total_importance']):
            return driver['total_importance']
        else:
            return 0.0
        
    def page_individual_analysis(self):
        """Page 4: Individual Analysis - Improved Layout"""
        st.title("👤 Individual Customer Analysis")
        st.markdown("### Deep Dive into Customer-Level Insights & Risk Drivers")
        
        # Customer Selection Section
        st.subheader("🔍 Customer Selection")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            customer_id = st.selectbox(
                "Select Customer ID",
                options=self.predictions_df['customer_id'].unique(),
                help="Choose a customer to analyze their churn risk profile"
            )
        
        # Get customer data
        customer_data = self.predictions_df[
            self.predictions_df['customer_id'] == customer_id
        ].iloc[0]
        
        customer_shap = self.individual_shap_df[
            self.individual_shap_df['customer_id'] == customer_id
        ]
        
        # Customer Profile Header
        st.markdown("---")
        st.subheader("📊 Customer Profile Summary")
        
        # Main KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_color = "#FF6B6B" if customer_data['risk_segment'] == 'High Risk' else "#FFD166" if customer_data['risk_segment'] == 'Medium Risk' else "#06D6A0"
            st.markdown(
                f"""
                <div style="background: {risk_color}20; padding: 20px; border-radius: 10px; border-left: 4px solid {risk_color};">
                    <h4 style="margin: 0; color: {risk_color};">Churn Probability</h4>
                    <h2 style="margin: 5px 0; color: {risk_color};">{customer_data['churn_probability']:.1%}</h2>
                    <p style="margin: 0; font-size: 0.9em;">{customer_data['risk_segment']} Risk</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div style="background: #118AB220; padding: 20px; border-radius: 10px; border-left: 4px solid #118AB2;">
                    <h4 style="margin: 0; color: #118AB2;">Expected Loss</h4>
                    <h2 style="margin: 5px 0; color: #118AB2;">${customer_data['expected_loss']:,.0f}</h2>
                    <p style="margin: 0; font-size: 0.9em;">Potential revenue loss</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            segment_avg = self.predictions_df[
                self.predictions_df['risk_segment'] == customer_data['risk_segment']
            ]['churn_probability'].mean()
            comparison = "Above" if customer_data['churn_probability'] > segment_avg else "Below"
            comparison_color = "#FF6B6B" if comparison == "Above" else "#06D6A0"
            
            st.markdown(
                f"""
                <div style="background: {comparison_color}20; padding: 20px; border-radius: 10px; border-left: 4px solid {comparison_color};">
                    <h4 style="margin: 0; color: {comparison_color};">Segment Comparison</h4>
                    <h2 style="margin: 5px 0; color: {comparison_color};">{comparison}</h2>
                    <p style="margin: 0; font-size: 0.9em;">Segment avg: {segment_avg:.1%}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col4:
            region_info = customer_data['region'] if 'region' in customer_data else "N/A"
            st.markdown(
                """
                <div style="background: #6A4C9320; padding: 20px; border-radius: 10px; border-left: 4px solid #6A4C93;">
                    <h4 style="margin: 0; color: #6A4C93;">Customer Details</h4>
                    <p style="margin: 5px 0; font-size: 1.1em;"><strong>ID:</strong> {}</p>
                    <p style="margin: 0; font-size: 0.9em;"><strong>Region:</strong> {}</p>
                </div>
                """.format(customer_id, region_info),
                unsafe_allow_html=True
            )
        
        # Risk Drivers Section
        st.markdown("---")
        st.subheader("🎯 Risk Driver Analysis")
        
        if not customer_shap.empty:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Top Risk Drivers
                st.markdown("#### 🚨 Top Risk Drivers")
                positive_drivers = customer_shap[customer_shap['shap_value'] > 0].nlargest(5, 'shap_value')
                
                if not positive_drivers.empty:
                    for idx, (_, driver) in enumerate(positive_drivers.iterrows()):
                        intensity = min(90 + (idx * 20), 100)
                        st.markdown(
                            f"""
                            <div style="background: linear-gradient(90deg, #FF6B6B{intensity} 0%, #FFFFFF 100%); 
                                        padding: 12px; margin: 8px 0; border-radius: 8px; border-left: 4px solid #FF6B6B;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-weight: bold;">{driver['feature_name']}</span>
                                    <span style="color: #FF6B6B; font-weight: bold;">+{driver['shap_value']:.4f}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No significant risk drivers identified")
                
                # Protective Factors
                st.markdown("#### 🛡️ Protective Factors")
                negative_drivers = customer_shap[customer_shap['shap_value'] < 0].nsmallest(5, 'shap_value')
                
                if not negative_drivers.empty:
                    for idx, (_, driver) in enumerate(negative_drivers.iterrows()):
                        intensity = min(90 + (idx * 20), 100)
                        st.markdown(
                            f"""
                            <div style="background: linear-gradient(90deg, #06D6A0{intensity} 0%, #FFFFFF 100%); 
                                        padding: 12px; margin: 8px 0; border-radius: 8px; border-left: 4px solid #06D6A0;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-weight: bold;">{driver['feature_name']}</span>
                                    <span style="color: #06D6A0; font-weight: bold;">{driver['shap_value']:.4f}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No significant protective factors identified")
            
            with col2:
                # Feature Impact Visualization
                st.markdown("#### 📊 Feature Impact Overview")
                top_drivers = customer_shap.nlargest(10, 'absolute_shap')
                
                fig = px.bar(
                    top_drivers,
                    x='shap_value',
                    y='feature_name',
                    orientation='h',
                    title=f"Top Feature Impacts for Customer {customer_id}",
                    color='shap_value',
                    color_continuous_scale='RdBu_r',
                    range_color=[-abs(top_drivers['shap_value'].max()), abs(top_drivers['shap_value'].max())]
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    xaxis_title="SHAP Value Impact",
                    yaxis_title="Features",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=2)
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ No individual SHAP data available for this customer")
        
        # Comparison & Insights Section
        st.markdown("---")
        st.subheader("📈 Comparative Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Gauge Chart
            st.markdown("#### 🎯 Risk Level Comparison")
            segment_avg = self.predictions_df[
                self.predictions_df['risk_segment'] == customer_data['risk_segment']
            ]['churn_probability'].mean()
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = customer_data['churn_probability'],
                delta = {'reference': segment_avg, 'relative': False},
                gauge = {
                    'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#FF6B6B" if customer_data['risk_segment'] == 'High Risk' else "#FFD166"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 0.3], 'color': "#06D6A0"},
                        {'range': [0.3, 0.7], 'color': "#FFD166"},
                        {'range': [0.7, 1], 'color': "#FF6B6B"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': segment_avg
                    }
                }
            ))
            
            fig_gauge.update_layout(
                height=300,
                title={
                    'text': f"vs {customer_data['risk_segment']} Segment Average",
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Key Insights
            st.markdown("#### 💡 Key Insights & Recommendations")
            
            # Generate dynamic insights
            insights = []
            
            if not customer_shap.empty:
                top_driver = customer_shap.nlargest(1, 'absolute_shap').iloc[0]
                insights.append(f"⚠️ **Primary Driver**: {top_driver['feature_name']} contributes most to churn risk")
                
                if top_driver['shap_value'] > 0.1:
                    insights.append("🚨 **High Impact**: Consider immediate intervention for the primary risk factor")
                elif top_driver['shap_value'] > 0.05:
                    insights.append("⚠️ **Medium Impact**: Monitor closely and consider proactive measures")
                else:
                    insights.append("✅ **Low Impact**: Current risk level is manageable with standard retention")
            
            if customer_data['churn_probability'] > 0.7:
                insights.append("🎯 **Priority Action**: This customer requires immediate retention efforts")
            elif customer_data['churn_probability'] > 0.4:
                insights.append("📊 **Watchlist**: Monitor behavior and consider targeted offers")
            else:
                insights.append("💚 **Stable**: Maintain current relationship management")
            
            # Display insights in cards
            for i, insight in enumerate(insights):
                st.markdown(
                    f"""
                    <div style="background: black; padding: 12px; margin: 8px 0; border-radius: 8px; 
                                border-left: 4px solid {'#FF6B6B' if '🚨' in insight else '#FFD166' if '⚠️' in insight else '#06D6A0'};">
                        {insight}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Detailed Feature Breakdown
        st.markdown("---")
        st.subheader("🔍 Detailed Feature Breakdown")
        
        if not customer_shap.empty:
            # Add some metrics above the table
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_positive_impact = customer_shap[customer_shap['shap_value'] > 0]['shap_value'].sum()
                st.metric("Total Risk Impact", f"+{total_positive_impact:.4f}")
            
            with col2:
                total_negative_impact = customer_shap[customer_shap['shap_value'] < 0]['shap_value'].sum()
                st.metric("Total Protective Impact", f"{total_negative_impact:.4f}")
            
            with col3:
                net_impact = customer_shap['shap_value'].sum()
                st.metric("Net Impact", f"{net_impact:.4f}")
            
            # Enhanced data table
            display_df = customer_shap[['feature_name', 'shap_value', 'absolute_shap']].copy()
            display_df = display_df.sort_values('absolute_shap', ascending=False)
            display_df['impact_type'] = display_df['shap_value'].apply(
                lambda x: 'Risk Driver' if x > 0 else 'Protective Factor'
            )
            display_df['shap_value'] = display_df['shap_value'].round(4)
            display_df['absolute_shap'] = display_df['absolute_shap'].round(4)
            
            # Style the dataframe
            def color_impact(val):
                color = '#FF6B6B' if val > 0 else '#06D6A0'
                return f'color: {color}; font-weight: bold;'
            
            styled_df = display_df.style.applymap(color_impact, subset=['shap_value'])
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=400,
                column_config={
                    "feature_name": "Feature Name",
                    "shap_value": "SHAP Value",
                    "absolute_shap": "Absolute Impact",
                    "impact_type": "Impact Type"
                }
            )
            
            # Download option
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Feature Analysis",
                data=csv,
                file_name=f"customer_{customer_id}_feature_analysis.csv",
                mime="text/csv"
            )
        
        else:
            st.info("Individual feature analysis requires SHAP values. Please ensure individual SHAP data is available.")

        # Action Plan Section
        st.markdown("---")
        st.subheader("🎯 Recommended Action Plan")
        
        # Dynamic action plan based on risk level
        risk_level = customer_data['risk_segment']
        churn_prob = customer_data['churn_probability']
        
        if risk_level == 'High Risk':
            st.error("**🚨 IMMEDIATE ACTION REQUIRED**")
            actions = [
                "Personalized retention call within 24 hours",
                "Special offer: 20% discount on premium services",
                "Account review with senior relationship manager",
                "Expedited complaint resolution process"
            ]
            color = "#FF6B6B"
        elif risk_level == 'Medium Risk':
            st.warning("**⚠️ PROACTIVE MONITORING**")
            actions = [
                "Personalized email with service benefits",
                "Offer: 15% loyalty discount",
                "Quarterly account review",
                "Customer satisfaction survey"
            ]
            color = "#FFD166"
        else:
            st.success("**✅ MAINTENANCE & GROWTH**")
            actions = [
                "Regular relationship check-in",
                "Cross-sell appropriate products",
                "Loyalty program enrollment",
                "Quarterly newsletter"
            ]
            color = "#06D6A0"
        
        # Display action plan
        for i, action in enumerate(actions):
            st.markdown(
                f"""
                <div style="background: {color}20; padding: 12px; margin: 8px 0; border-radius: 8px; 
                            border-left: 4px solid {color}; display: flex; align-items: center;">
                    <span style="margin-right: 10px; font-weight: bold; color: {color};">{i+1}.</span>
                    <span>{action}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    def page_business_impact(self):
        """Page 5: Business Impact"""
        st.title("💰 Business Impact")
        st.markdown("### Financial Analysis & ROI Projections")
        
        # Financial KPIs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_loss = self.predictions_df['expected_loss'].sum()
            st.metric(
                "Total Expected Loss",
                f"${total_loss:,.0f}",
                delta=f"${(total_loss * 0.3):,.0f} savings potential"
            )
        
        with col2:
            high_risk_loss = self.predictions_df[
                self.predictions_df['risk_segment'] == 'High Risk'
            ]['expected_loss'].sum()
            st.metric(
                "High Risk Segment Loss",
                f"${high_risk_loss:,.0f}",
                delta=f"{(high_risk_loss/total_loss*100):.1f}% of total"
            )
        
        with col3:
            avg_customer_value = total_loss / self.total_customers
            st.metric(
                "Avg Customer Value at Risk",
                f"${avg_customer_value:,.0f}",
            )
        
        # Financial Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Expected Loss by Risk Segment
            loss_by_segment = self.predictions_df.groupby('risk_segment')['expected_loss'].sum().reset_index()
            fig_pie = px.pie(
                loss_by_segment,
                values='expected_loss',
                names='risk_segment',
                title="Expected Loss Distribution by Risk Segment",
                color='risk_segment',
                color_discrete_map={
                    'High Risk': '#FF6B6B',
                    'Medium Risk': '#FFD166',
                    'Low Risk': '#06D6A0'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # ROI Projection
            retention_rates = {
                'High Risk': 0.3,  # 30% retention with intervention
                'Medium Risk': 0.5, # 50% retention
                'Low Risk': 0.7     # 70% retention
            }
            
            savings_data = []
            for segment in ['High Risk', 'Medium Risk', 'Low Risk']:
                segment_loss = self.predictions_df[
                    self.predictions_df['risk_segment'] == segment
                ]['expected_loss'].sum()
                savings = segment_loss * retention_rates[segment]
                savings_data.append({
                    'Segment': segment,
                    'Potential Savings': savings
                })
            
            savings_df = pd.DataFrame(savings_data)
            fig_bar = px.bar(
                savings_df,
                x='Segment',
                y='Potential Savings',
                title="Potential Savings from Retention Programs",
                color='Segment',
                color_discrete_map={
                    'High Risk': '#FF6B6B',
                    'Medium Risk': '#FFD166',
                    'Low Risk': '#06D6A0'
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Regional Analysis (if available)
        if 'region' in self.predictions_df.columns:
            st.subheader("🌍 Regional Impact Analysis")
            
            regional_impact = self.predictions_df.groupby('region').agg({
                'expected_loss': 'sum',
                'customer_id': 'count',
                'churn_probability': 'mean'
            }).reset_index()
            
            regional_impact.columns = ['Region', 'Total Loss', 'Customer Count', 'Avg Churn Probability']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_region = px.bar(
                    regional_impact,
                    x='Region',
                    y='Total Loss',
                    title="Expected Loss by Region",
                    color='Total Loss',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_region, use_container_width=True)
            
            with col2:
                fig_scatter = px.scatter(
                    regional_impact,
                    x='Customer Count',
                    y='Total Loss',
                    size='Avg Churn Probability',
                    color='Region',
                    title="Regional Impact: Customers vs Loss",
                    hover_data=['Region', 'Total Loss', 'Customer Count']
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Action Recommendations
        st.subheader("🎯 Recommended Actions")
        
        high_risk_customers = self.predictions_df[self.predictions_df['risk_segment'] == 'High Risk']
        
        if not high_risk_customers.empty:
            st.warning(f"**Immediate Action Required:** {len(high_risk_customers)} high-risk customers identified")
            st.write("**Recommended interventions:**")
            st.write("1. Personal retention offers for high-risk segment")
            st.write("2. Proactive customer service outreach")
            st.write("3. Product feature education campaigns")
            st.write(f"4. Expected ROI: ${(high_risk_loss * 0.3):,.0f} potential savings")

    def run_dashboard(self):
        """Run the complete dashboard"""
        # Sidebar navigation
        st.sidebar.title("🏦 Navigation")
        page = st.sidebar.radio(
            "Go to",
            [
                "Executive Overview",
                "Risk Analysis", 
                "Driver Analysis",
                "Individual Analysis",
                "Business Impact"
            ]
        )
        
        # Display selected page
        if page == "Executive Overview":
            self.page_executive_overview()
        elif page == "Risk Analysis":
            self.page_risk_analysis()
        elif page == "Driver Analysis":
            self.page_driver_analysis()
        elif page == "Individual Analysis":
            self.page_individual_analysis()
        elif page == "Business Impact":
            self.page_business_impact()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.info(
            f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"**Total Customers:** {self.total_customers:,}\n"
            f"**High Risk:** {self.high_risk_count} ({(self.high_risk_count/self.total_customers*100):.1f}%)"
        )

# Run the dashboard
if __name__ == "__main__":
    dashboard = ChurnDashboard()
    dashboard.run_dashboard()