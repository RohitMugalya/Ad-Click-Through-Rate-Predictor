# Ad Click Through Rate Predictor

A comprehensive machine learning web application that analyzes user behavior patterns and predicts the likelihood of users clicking on advertisements. Built with Streamlit, this interactive dashboard provides both exploratory data analysis and real-time prediction capabilities.

## Live Demo

**[Try the live application here](https://ad-click-through-rate-predictor.streamlit.app/)**


## Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Machine Learning**: Scikit-learn (Logistic Regression)

## Project Structure

```
Ad-Click-Through-Rate-Predictor/
├── app.py                           # Main Streamlit application
├── advertising.csv                  # Dataset with user behavior data
├── ad_click_model.pkl              # Trained machine learning model
├── model_features.pkl              # Feature names for model
├── requirements.txt                # Python dependencies
├── model_building.ipynb            # Model development notebook
├── model_training.ipynb            # Model training process
├── exloratory_data_analysis.ipynb  # EDA notebook
└── README.md                       # Project documentation
```

## Dataset

The application uses an advertising dataset with the following features:

- **Daily Time Spent on Site**: Time user spends on the website (minutes)
- **Age**: User's age
- **Area Income**: Average income of user's geographical area
- **Daily Internet Usage**: Average daily internet consumption (MB)
- **Male**: Gender indicator (1 for Male, 0 for Female)
- **Country**: User's country
- **Timestamp**: When the ad was displayed
- **Clicked on Ad**: Target variable (1 if clicked, 0 if not)

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/RohitMugalya/Ad-Click-Through-Rate-Predictor.git
   cd Ad-Click-Through-Rate-Predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## Usage

### Exploratory Data Analysis
1. Navigate to the **EDA Dashboard** from the sidebar
2. Use the **Gender filter** to customize your analysis
3. Explore different sections:
   - View KPIs for overall campaign performance
   - Analyze user demographics and behavior patterns
   - Examine temporal trends in ad clicks
   - Investigate geographic distribution of users

### Making Predictions
1. Switch to the **Click Prediction** page
2. Input user characteristics:
   - Daily time spent on site
   - Age
   - Area income
   - Daily internet usage
   - Gender
3. Click **"Predict Click Probability"**
4. View the prediction results, confidence level, and feature importance

## Model Information

- **Algorithm**: Logistic Regression
- **Features**: 5 key behavioral and demographic features
- **Performance**: Optimized for click-through rate prediction
- **Interpretability**: Provides feature importance and prediction explanations

## Key Insights

The application reveals several interesting patterns:
- Users spending moderate time on site have higher click rates
- Middle-aged users (30-50) show higher engagement
- Geographic and gender-based variations in click behavior
- Temporal patterns in ad engagement throughout the day and week

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- Dataset source: Advertising campaign data
- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/)
- Machine learning with [Scikit-learn](https://scikit-learn.org/)
