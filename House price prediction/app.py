import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
# Assuming these are custom modules you have created.
# Make sure the files 'data_processor.py', 'model_trainer.py', and 'utils.py'
# are in the same directory as this script.
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from utils import format_currency, create_prediction_explanation

# Page configuration
st.set_page_config(
    page_title="Hyderabad House Price Predictor",
        layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None

def load_and_process_data():
    """Load and process the house data"""
    try:
        # Load the CSV file
        data_file = "attached_assets/Hyderabad_House_Data_1755501086395.csv"
        
        if not os.path.exists(data_file):
            st.error(f"Data file not found: {data_file}")
            st.info("Please ensure the data file is in the correct location.")
            return None
            
        processor = DataProcessor()
        processed_data = processor.load_and_clean_data(data_file)
        
        if processed_data is None or processed_data.empty:
            st.error("Failed to process the data. Please check the data file format.")
            return None
            
        return processed_data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def train_model(data):
    """Train the machine learning model"""
    try:
        trainer = ModelTrainer()
        model, metrics, feature_names = trainer.train_model(data)
        
        return model, metrics, feature_names
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None

def main():
    # Title and description
    st.title("Hyderabad House Price Predictor")
    st.markdown("""
    Welcome to the intelligent house price prediction system for Hyderabad! 
    This application uses advanced machine learning to estimate property values based on key features.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    # Removed "Help" from the page selection
    page = st.sidebar.selectbox("Choose a page:", 
                                  ["Price Prediction", "Model Performance", "Data Insights"])
    
    # Load data if not already loaded
    if st.session_state.processed_data is None:
        with st.spinner("Loading and processing data..."):
            st.session_state.processed_data = load_and_process_data()
    
    # Train model if not already trained
    if not st.session_state.model_trained and st.session_state.processed_data is not None:
        with st.spinner("Training machine learning model..."):
            model, metrics, feature_names = train_model(st.session_state.processed_data)
            if model is not None:
                st.session_state.model = model
                st.session_state.model_metrics = metrics
                st.session_state.feature_names = feature_names
                st.session_state.model_trained = True
                st.success("Model trained successfully!")
    
    if st.session_state.processed_data is None:
        st.error("Unable to load data. Please check the data file.")
        return
    
    # Page routing
    if page == "Price Prediction":
        show_prediction_page()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Data Insights":
        show_data_insights()
    # The 'elif page == "Help":' block has been removed.

def show_prediction_page():
    """Main prediction interface"""
    st.header("HYD Predict House Price")
    
    if not st.session_state.model_trained:
        st.warning("Model is still training. Please wait...")
        return
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Property Details")
        
        # Bedroom selection
        bedrooms = st.selectbox(
            "Number of Bedrooms:",
            options=["1 BHK", "2 BHK", "3 BHK", "4 BHK", "Studio"],
            index=2,
            help="Select the type of accommodation"
        )
        
        # Bathroom selection
        bathrooms = st.number_input(
            "Number of Bathrooms:",
            min_value=1,
            max_value=5,
            value=2,
            help="Enter the number of bathrooms"
        )
        
        # Furnishing selection
        furnishing = st.selectbox(
            "Furnishing Type:",
            options=["Furnished", "Semi-Furnished", "Unfurnished"],
            index=1,
            help="Select the furnishing status"
        )
        
        # Tenant preference
        tenant_type = st.selectbox(
            "Tenant Preference:",
            options=["Bachelors/Family", "Family", "Bachelors"],
            index=0,
            help="Select preferred tenant type"
        )
    
    with col2:
        st.subheader("Location & Area")
        
        # Get unique localities
        localities = sorted(st.session_state.processed_data['Locality_clean'].unique())
        
        # Locality selection
        locality = st.selectbox(
            "Locality:",
            options=localities,
            help="Select the locality/area"
        )
        
        # Area input
        area = st.number_input(
            "Area (sq ft):",
            min_value=300,
            max_value=5000,
            value=1500,
            step=50,
            help="Enter the property area in square feet"
        )
    
    # Prediction button
    if st.button(" Predict Price", type="primary"):
        try:
            # Prepare input data
            processor = DataProcessor()
            input_data = processor.prepare_prediction_input(
                bedrooms, bathrooms, furnishing, tenant_type, area, locality,
                st.session_state.processed_data
            )
            
            # Make prediction
            prediction = st.session_state.model.predict([input_data])[0]
            
            # Display results
            st.success("Prediction completed!")
            
            # Create results display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Predicted Price",
                    value=format_currency(prediction),
                    help="Estimated monthly rent"
                )
            
            with col2:
                # Calculate confidence interval (approximate)
                confidence_range = prediction * 0.15  # 15% range
                lower_bound = max(0, prediction - confidence_range)
                upper_bound = prediction + confidence_range
                
                st.metric(
                    label="Price Range",
                    value=f"₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}",
                    help="Estimated price range (±15%)"
                )
            
            with col3:
                # Price per sq ft
                price_per_sqft = prediction / area
                st.metric(
                    label="Price per Sq Ft",
                    value=f"₹{price_per_sqft:.0f}",
                    help="Rent per square foot"
                )
            
            # Explanation
            st.subheader(" Prediction Explanation")
            explanation = create_prediction_explanation(
                bedrooms, locality, area, furnishing, prediction
            )
            st.info(explanation)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def show_model_performance():
    """Display model performance metrics"""
    st.header(" Model Performance")
    
    if not st.session_state.model_trained or st.session_state.model_metrics is None:
        st.warning("Model metrics not available. Please train the model first.")
        return
    
    metrics = st.session_state.model_metrics
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="R² Score",
            value=f"{metrics['r2']:.3f}",
            help="Coefficient of determination (higher is better)"
        )
    
    with col2:
        st.metric(
            label="MAE",
            value=f"₹{metrics['mae']:,.0f}",
            help="Mean Absolute Error"
        )
    
    with col3:
        st.metric(
            label="RMSE",
            value=f"₹{metrics['rmse']:,.0f}",
            help="Root Mean Square Error"
        )
    
    with col4:
        st.metric(
            label="CV Score",
            value=f"{metrics['cv_score']:.3f}",
            help="Cross-validation score"
        )
    
    # Feature importance
    if 'feature_importance' in metrics:
        st.subheader(" Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': st.session_state.feature_names,
            'Importance': metrics['feature_importance']
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 10 Most Important Features",
            labels={'Importance': 'Feature Importance', 'Feature': 'Features'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model details
    st.subheader(" Model Details")
    st.info("""
    **Model Type:** Random Forest Regressor
    
    **Training Details:**
    - Algorithm: Random Forest with 100 estimators
    - Cross-validation: 5-fold
    - Features: Bedrooms, Bathrooms, Area, Locality, Furnishing, Tenant Type
    - Target: Monthly Rent (INR)
    
    **Performance Interpretation:**
    - **R² Score:** Explains the proportion of variance in house prices
    - **MAE:** Average absolute difference between predicted and actual prices
    - **RMSE:** Root mean square error, penalizes larger errors more
    - **CV Score:** Average performance across different data splits
    """)

def show_data_insights():
    """Display data insights and visualizations"""
    st.header(" Data Analysis & Insights")
    
    data = st.session_state.processed_data
    
    # Summary statistics
    st.subheader(" Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", len(data))
    
    with col2:
        st.metric("Average Price", format_currency(data['Price_clean'].mean()))
    
    with col3:
        st.metric("Unique Localities", data['Locality_clean'].nunique())
    
    with col4:
        st.metric("Price Range", f"₹{data['Price_clean'].min():,.0f} - ₹{data['Price_clean'].max():,.0f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution by bedroom type
        fig1 = px.box(
            data,
            x='Bedrooms_clean',
            y='Price_clean',
            title="Price Distribution by Bedroom Type",
            labels={'Price_clean': 'Price (INR)', 'Bedrooms_clean': 'Bedroom Type'}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Price by furnishing type
        fig2 = px.violin(
            data,
            x='Furnishing',
            y='Price_clean',
            title="Price Distribution by Furnishing Type",
            labels={'Price_clean': 'Price (INR)', 'Furnishing': 'Furnishing Type'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Top localities by average price
    st.subheader(" Top Localities by Average Price")
    locality_avg = data.groupby('Locality_clean')['Price_clean'].agg(['mean', 'count']).reset_index()
    locality_avg = locality_avg[locality_avg['count'] >= 3]  # At least 3 properties
    locality_avg = locality_avg.sort_values('mean', ascending=False).head(15)
    
    fig3 = px.bar(
        locality_avg,
        x='mean',
        y='Locality_clean',
        orientation='h',
        title="Average Rent by Locality (Top 15)",
        labels={'mean': 'Average Price (INR)', 'Locality_clean': 'Locality'}
    )
    fig3.update_layout(height=600)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Price vs Area scatter plot
    st.subheader("Price vs Area Analysis")
    fig4 = px.scatter(
        data,
        x='Area_clean',
        y='Price_clean',
        color='Bedrooms_clean',
        title="Price vs Area by Bedroom Type",
        labels={'Area_clean': 'Area (sq ft)', 'Price_clean': 'Price (INR)'},
        hover_data=['Locality_clean']
    )
    st.plotly_chart(fig4, use_container_width=True)


# The 'show_help_page' function has been completely removed.

if __name__ == "__main__":
    main()