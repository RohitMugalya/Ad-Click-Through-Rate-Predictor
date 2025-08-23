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

st.set_page_config(
    page_title="Ad Click Through Rate Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        color: white;
        background-color: #0e1117;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp > .main > .block-container {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    filepath = "advertising.csv"
    data = pd.read_csv(filepath)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.drop('City', axis='columns', inplace=True)
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
    return data

@st.cache_resource
def load_model():
    model = joblib.load('ad_click_model.pkl')
    with open('model_features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names


data = load_data()
try:
    model, feature_names = load_model()
    model_loaded = True
except:
    st.warning("Model files not found. Please train the model first.")
    model_loaded = False


st.sidebar.title("Ad Click Through Rate Predictor")
st.sidebar.markdown("""
**Welcome!**  
Navigate between sections to explore:
- **EDA Dashboard**: Analyze user behavior and campaign insights
- **Click Prediction**: Predict ad click probability for new users
""")
page = st.sidebar.radio("Navigation", ["EDA Dashboard", "Click Prediction"])


if page == "EDA Dashboard":
    st.sidebar.header("Filters")
    gender_filter = st.sidebar.selectbox("Gender", options=["All", "Male", "Female"])


    filtered_data = data.copy()
    if gender_filter != "All":
        filtered_data = filtered_data[filtered_data['Male'] == (1 if gender_filter == "Male" else 0)]


if page == "EDA Dashboard":
    st.title("Exploratory Data Analysis")
    st.markdown("""
    Dive deep into user behavior patterns and advertising campaign performance metrics.
    Use the filters in the sidebar to customize your analysis and discover what drives ad clicks.
    """)


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


    st.header("User Demographics Distribution")

    col1, col2 = st.columns(2)

    with col1:

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

        fig_income = go.Figure()
        fig_income.add_trace(go.Histogram(
            x=filtered_data['Area Income'],
            name='Area Income',
            marker_color='red'
        ))
        fig_income.update_layout(title='Area Income Distribution', bargap=0.1)
        st.plotly_chart(fig_income, use_container_width=True)


    st.header("User Behavior Analysis")

    col1, col2 = st.columns(2)

    with col1:

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

        feature = st.selectbox(
            "Select feature for box plot:",
            options=['Age', 'Area Income', 'Daily Time Spent on Site', 'Daily Internet Usage']
        )
        

        box_data = filtered_data.copy()
        box_data['Click Status'] = box_data['Clicked on Ad'].map({0: 'Not Clicked', 1: 'Clicked'})
        
        fig_box = px.box(
            box_data,
            x='Click Status',
            y=feature,
            color='Click Status',
            title=f'{feature} vs Click Status',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_box, use_container_width=True)


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


    st.header("Temporal Patterns")

    col1, col2 = st.columns(2)

    with col1:

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

        day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        daily_clicks = filtered_data.copy()
        daily_clicks['DayName'] = daily_clicks['DayOfWeek'].map(day_map)
        daily_clicks = daily_clicks.groupby(['DayName', 'Clicked on Ad']).size().reset_index(name='Count')
        

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


    st.header("Geographic Analysis")


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


    st.header("Raw Data")
    if st.checkbox("Show raw data"):
        st.dataframe(filtered_data)


elif page == "Click Prediction":
    st.title("Click Prediction Model")
    st.markdown("""
    Enter user characteristics below to predict their likelihood of clicking on an advertisement.
    Our machine learning model analyzes behavioral patterns and demographics to provide accurate predictions.
    """)
    
    if not model_loaded:
        st.error("Model not loaded. Please make sure you have trained the model first.")
    else:

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

            input_data = pd.DataFrame({
                'Daily Time Spent on Site': [daily_time_spent],
                'Age': [age],
                'Area Income': [area_income * 1000],
                'Daily Internet Usage': [daily_internet_usage],
                'Male': [1 if male == "Male" else 0]
            })
            

            try:
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)
                

                st.header("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Prediction", "Will Click" if prediction[0] == 1 else "Will Not Click")
                
                with col2:
                    st.metric("Confidence", f"{max(prediction_proba[0]):.2%}")
                

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
                

                st.subheader("Key Factors Influencing This Prediction")
                

                if hasattr(model, 'coef_'):
                    coefficients = model.coef_[0]
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': coefficients
                    }).sort_values('Importance', key=abs, ascending=False)
                    

                    for i, row in feature_importance.iterrows():
                        importance_desc = "Increases click probability" if row['Importance'] > 0 else "Decreases click probability"
                        importance_strength = "strongly" if abs(row['Importance']) > 0.5 else "moderately"
                        st.write(f"**{row['Feature']}**: {importance_strength} {importance_desc.lower()}")
                

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


st.markdown("---")
st.markdown("**Ad Click Through Rate Predictor** | Built with Streamlit & Machine Learning")