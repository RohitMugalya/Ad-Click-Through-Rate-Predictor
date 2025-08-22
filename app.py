import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import joblib
import pickle
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Ad Click Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and model
@st.cache_data
def load_data():
    filepath = "advertising.csv"
    data = pd.read_csv(filepath)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.drop('City', axis='columns', inplace=True)
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek  # 0=Mon
    return data

@st.cache_resource
def load_model():
    model = joblib.load('ad_click_model.pkl')
    with open('model_features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

# Load the data and model
data = load_data()
try:
    model, feature_names = load_model()
    model_loaded = True
except:
    st.warning("Model files not found. Please train the model first.")
    model_loaded = False

# Sidebar navigation
st.sidebar.title("Ad Click Prediction Dashboard")
page = st.sidebar.radio("Navigation", ["EDA Dashboard", "Click Prediction"])

# Filters for EDA
if page == "EDA Dashboard":
    st.sidebar.header("Filters")
    click_filter = st.sidebar.selectbox("Clicked on Ad", options=["All", "Yes", "No"])
    gender_filter = st.sidebar.selectbox("Gender", options=["All", "Male", "Female"])

    # Apply filters
    filtered_data = data.copy()
    if click_filter != "All":
        filtered_data = filtered_data[filtered_data['Clicked on Ad'] == (1 if click_filter == "Yes" else 0)]
    if gender_filter != "All":
        filtered_data = filtered_data[filtered_data['Male'] == (1 if gender_filter == "Male" else 0)]

# EDA Dashboard
if page == "EDA Dashboard":
    st.title("Advertising Campaign Analysis Dashboard")
    st.markdown("""
    This dashboard provides insights into user behavior and advertising campaign performance.
    Explore the metrics and visualizations below to understand what drives ad clicks.
    """)

    # KPI cards
    st.header("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)

    total_users = len(filtered_data)
    click_rate = filtered_data['Clicked on Ad'].mean() * 100
    avg_time_spent = filtered_data['Daily Time Spent on Site'].mean()
    avg_internet_usage = filtered_data['Daily Internet Usage'].mean()

    col1.metric("Total Users", f"{total_users:,}")
    col2.metric("Click-Through Rate", f"{click_rate:.2f}%")
    col3.metric("Avg. Time on Site", f"{avg_time_spent:.2f} mins")
    col4.metric("Avg. Internet Usage", f"{avg_internet_usage:.2f} MB")

    # Distribution plots
    st.header("User Demographics Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Age distribution
        fig_age = px.histogram(
            filtered_data, 
            x='Age', 
            nbins=30, 
            title='Age Distribution',
            color_discrete_sequence=['tomato']
        )
        fig_age.update_layout(bargap=0.1)
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        # Area Income distribution
        fig_income = go.Figure()
        fig_income.add_trace(go.Histogram(
            x=filtered_data['Area Income'],
            name='Area Income',
            marker_color='red'
        ))
        fig_income.update_layout(title='Area Income Distribution', bargap=0.1)
        st.plotly_chart(fig_income, use_container_width=True)

    # Behavioral analysis
    st.header("User Behavior Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Scatter plot: Internet Usage vs Time Spent
        fig_scatter = px.scatter(
            filtered_data,
            x='Daily Internet Usage',
            y='Daily Time Spent on Site',
            color='Clicked on Ad',
            title='Internet Usage vs Time Spent on Site',
            color_discrete_map={0: 'lightgray', 1: 'blue'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        # Box plots per target
        feature = st.selectbox(
            "Select feature for box plot:",
            options=['Age', 'Area Income', 'Daily Time Spent on Site', 'Daily Internet Usage']
        )
        
        fig_box = px.box(
            filtered_data,
            x='Clicked on Ad',
            y=feature,
            color='Clicked on Ad',
            title=f'{feature} vs Clicked on Ad',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # KDE plots
    st.header("Feature Distribution by Click Status")
    kde_feature = st.selectbox(
        "Select feature for density plot:",
        options=['Age', 'Daily Time Spent on Site', 'Daily Internet Usage'],
        key='kde_feature'
    )

    clicked_data = filtered_data[filtered_data['Clicked on Ad'] == 1][kde_feature]
    not_clicked_data = filtered_data[filtered_data['Clicked on Ad'] == 0][kde_feature]

    hist_data = [clicked_data, not_clicked_data]
    group_labels = ['Clicked', 'Not Clicked']

    fig_dist = ff.create_distplot(
        hist_data, group_labels, bin_size=10,
        colors=['blue', 'red']
    )
    fig_dist.update_layout(title=f'{kde_feature} Distribution by Click Status')
    st.plotly_chart(fig_dist, use_container_width=True)

    # Temporal analysis
    st.header("Temporal Patterns")

    col1, col2 = st.columns(2)

    with col1:
        # Clicks by hour
        hourly_clicks = filtered_data.groupby(['Hour', 'Clicked on Ad']).size().reset_index(name='Count')
        fig_hour = px.bar(
            hourly_clicks, 
            x='Hour', 
            y='Count', 
            color='Clicked on Ad',
            title='Clicks by Hour of Day',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    with col2:
        # Clicks by day of week
        day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        daily_clicks = filtered_data.copy()
        daily_clicks['DayName'] = daily_clicks['DayOfWeek'].map(day_map)
        daily_clicks = daily_clicks.groupby(['DayName', 'Clicked on Ad']).size().reset_index(name='Count')
        
        # Ensure correct order of days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_clicks['DayName'] = pd.Categorical(daily_clicks['DayName'], categories=day_order, ordered=True)
        daily_clicks = daily_clicks.sort_values('DayName')
        
        fig_day = px.bar(
            daily_clicks, 
            x='DayName', 
            y='Count', 
            color='Clicked on Ad',
            title='Clicks by Day of Week',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_day, use_container_width=True)

    # Geographic analysis
    st.header("Geographic Analysis")

    # Top countries
    top_countries = filtered_data['Country'].value_counts().head(10).index
    top_countries_data = filtered_data[filtered_data['Country'].isin(top_countries)]

    country_clicks = top_countries_data.groupby(['Country', 'Clicked on Ad']).size().reset_index(name='Count')
    fig_country = px.bar(
        country_clicks, 
        x='Country', 
        y='Count', 
        color='Clicked on Ad',
        title='Top 10 Countries by Click Distribution',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig_country.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_country, use_container_width=True)

    # CTR by country and gender
    ctr_by_country_gender = (
        filtered_data.groupby(['Country', 'Male'])['Clicked on Ad']
        .mean()
        .reset_index()
        .sort_values(by='Clicked on Ad', ascending=False)
    )
    ctr_by_country_gender = ctr_by_country_gender[ctr_by_country_gender['Country'].isin(top_countries)]
    ctr_by_country_gender['Gender'] = ctr_by_country_gender['Male'].map({1: 'Male', 0: 'Female'})

    fig_ctr = px.bar(
        ctr_by_country_gender, 
        x='Country', 
        y='Clicked on Ad', 
        color='Gender',
        title='CTR by Gender in Top Countries',
        barmode='group'
    )
    fig_ctr.update_layout(xaxis_tickangle=-45)
    fig_ctr.update_layout(yaxis_tickformat=".0%", yaxis_title="Click-Through Rate")
    st.plotly_chart(fig_ctr, use_container_width=True)

    # Data table
    st.header("Raw Data")
    if st.checkbox("Show raw data"):
        st.dataframe(filtered_data)

# Prediction Page - ONLY THIS SECTION IS UPDATED
elif page == "Click Prediction":
    st.title("Ad Click Prediction")
    st.markdown("""
    Enter user information to predict the probability of clicking on an ad.
    The model uses logistic regression to make predictions based on user behavior and demographics.
    """)
    
    if not model_loaded:
        st.error("Model not loaded. Please make sure you have trained the model first.")
    else:
        # Create input form - ONLY the 5 features your model uses
        with st.form("prediction_form"):
            st.header("User Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                daily_time_spent = st.slider("Daily Time Spent on Site (minutes)", 0, 100, 50)
                age = st.slider("Age", 18, 80, 35)
                area_income = st.slider("Area Income (thousands)", 10, 100, 50)
            
            with col2:
                daily_internet_usage = st.slider("Daily Internet Usage (MB)", 50, 300, 150)
                male = st.radio("Gender", ["Male", "Female"])
            
            submitted = st.form_submit_button("Predict Click Probability")
        
        if submitted:
            # Prepare input data - ONLY the 5 features your model uses
            input_data = pd.DataFrame({
                'Daily Time Spent on Site': [daily_time_spent],
                'Age': [age],
                'Area Income': [area_income * 1000],  # Convert back to original scale
                'Daily Internet Usage': [daily_internet_usage],
                'Male': [1 if male == "Male" else 0]
            })
            
            # Make prediction
            try:
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)
                
                # Display results
                st.header("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Prediction", "Will Click" if prediction[0] == 1 else "Will Not Click")
                
                with col2:
                    st.metric("Confidence", f"{max(prediction_proba[0]):.2%}")
                
                # Show probability breakdown
                st.subheader("Probability Breakdown")
                
                fig = go.Figure(go.Bar(
                    x=[prediction_proba[0][0], prediction_proba[0][1]],
                    y=['Not Click', 'Click'],
                    orientation='h',
                    marker_color=['red', 'green']
                ))
                
                fig.update_layout(
                    title="Probability of Clicking on Ad",
                    xaxis_title="Probability",
                    yaxis_title="Outcome",
                    xaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show feature importance explanation
                st.subheader("Key Factors Influencing This Prediction")
                
                # Get actual feature importances from the model
                if hasattr(model, 'coef_'):
                    coefficients = model.coef_[0]
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': coefficients
                    }).sort_values('Importance', key=abs, ascending=False)
                    
                    # Display top factors
                    for i, row in feature_importance.iterrows():
                        importance_desc = "Increases click probability" if row['Importance'] > 0 else "Decreases click probability"
                        importance_strength = "strongly" if abs(row['Importance']) > 0.5 else "moderately"
                        st.write(f"**{row['Feature']}**: {importance_strength} {importance_desc.lower()}")
                
                # Add some general insights based on typical patterns
                st.subheader("General Insights")
                insights = [
                    "Users who spend more time on site are more likely to click ads",
                    "Middle-aged users (30-50) tend to click more than younger or older users",
                    "Moderate internet users click more often than heavy or light users",
                    "Income level has a moderate impact on click behavior",
                    "Gender shows some correlation with click behavior patterns"
                ]
                
                for insight in insights:
                    st.write(f"• {insight}")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Ad Click Prediction Dashboard | Built with Streamlit")