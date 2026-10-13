✈️ RayAI — Smart Flight Pricing

<p align="center"> <strong>Predict flight fares. Compare booking scenarios. Understand AI predictions.</strong> </p>

An interactive flight price prediction dashboard built with Python, Streamlit, XGBoost, SHAP, and Plotly. Explore alternative flight configurations, analyze booking-window effects, and understand the factors influencing predicted ticket prices through explainable machine learning.

<p align="center"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"> <img src="https://img.shields.io/badge/XGBoost-189F5B?style=for-the-badge" alt="XGBoost"> <img src="https://img.shields.io/badge/Explainable_AI-SHAP-7B2CBF?style=for-the-badge" alt="SHAP"> <img src="https://img.shields.io/badge/Visualization-Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly"> </p>

Prediction • Scenario Analysis • Explainable AI

Overview • Features • Architecture • Machine Learning • Installation • Usage • Limitations

📌 Overview

RayAI — Smart Flight Pricing is an interactive machine learning application that estimates flight ticket prices using a trained XGBoost regression model.

Users can enter flight details, generate estimated fares, compare alternative flight configurations, and explore how different booking decisions affect model predictions.

Unlike a basic price prediction application, RayAI combines predictive modeling, interactive scenario analysis, and SHAP-based explainability in a single dashboard.

The application also supports downloadable PDF reports containing flight details, predicted fares, and leading feature contributions.

🎯 What It Can Do
✈️ Predict estimated flight fares in Indian Rupees (INR).
🔬 Explore alternative booking scenarios using a what-if simulator.
⚖️ Compare three flight configurations side by side.
🧠 Explain individual predictions using SHAP.
📅 Analyze the relationship between booking lead time and predicted fares.
🏙️ Compare routes, airlines, travel classes, and stop configurations.
📊 Visualize predictions using interactive Plotly charts.
⚙️ Engineer additional features from raw flight information.
📄 Generate downloadable PDF prediction reports.
🤖 Load a saved preprocessing pipeline and trained XGBoost model.
✨ Features
Feature	Description
✈️ Flight Price Prediction	Estimates ticket prices using a trained XGBoost regressor.
🔬 What-If Simulator	Tests alternative flight configurations while keeping other inputs fixed.
⚖️ Flight Comparison	Compares predicted fares for three flight scenarios.
🧠 SHAP Explainability	Identifies features that increase or decrease an individual prediction.
📈 Interactive Charts	Visualizes predicted prices across booking windows, classes, stops, and airlines.
🏙️ Route Analysis	Combines source and destination cities into a route feature.
📅 Booking-Window Analysis	Groups days remaining before departure into meaningful categories.
⚙️ Feature Engineering	Creates route, duration, booking-window, and combined categorical features.
📄 PDF Reports	Exports flight details, estimated prices, and leading SHAP contributions.
🎨 Interactive Dashboard	Provides a dark-themed Streamlit interface with dedicated modules.
📦 Saved Model Artifacts	Reuses a trained model and preprocessing pipeline for predictions.
🖥️ Dashboard Modules
✈️ 1. Flight Price Prediction

Enter flight details to estimate the ticket price.

Inputs include:

Airline
Source and destination cities
Departure and arrival time categories
Number of stops
Travel class
Flight duration
Days remaining before departure

Results include:

Estimated fare in INR
Flight and route summary
Booking-window information
SHAP feature contribution chart
Predicted price comparisons across selected scenarios
Engineered input features
🔬 2. What-If Simulator

Explore how changes in individual flight characteristics influence the model's output.

Experiment with:

Days remaining before departure
Airline
Travel class
Number of stops
Flight duration

The simulator generates alternative predictions while keeping the remaining baseline characteristics fixed.

⚖️ 3. Flight Comparison

Configure three flight scenarios using a shared source city, destination city, and departure-time category.

Compare their predicted fares through an interactive chart and a detailed comparison table.

📄 4. Prediction Report

Generate a downloadable PDF report after producing a prediction.

The report includes:

Estimated flight fare
Selected flight characteristics
Route and booking-window details
Leading SHAP feature contributions
Prediction disclaimer
🤖 5. Model Information

Review the configured XGBoost parameters, reported evaluation metrics, preprocessing pipeline, model artifacts, and SHAP interpretation guidance.

🏗️ Architecture

RayAI follows a modular prediction workflow that connects user input, feature engineering, preprocessing, machine learning inference, explainability, visualization, and reporting.

🔄 Complete Application Flow
                    👤 USER
                       |
                       ▼
               🎨 Streamlit Dashboard
                       |
                       ▼
              ✈️ Flight Information
                       |
                       ▼
               ⚙️ Feature Engineering
                       |
                       ▼
                📋 Input DataFrame
                       |
                       ▼
             🔄 Saved Preprocessor
             preprocessor.joblib
                       |
                       ▼
               🤖 XGBoost Regressor
             flight_price_xgb.json
                       |
                       ▼
                💰 Predicted Fare
                       |
           ┌───────────┼───────────┐
           │           │           │
           ▼           ▼           ▼
      📊 Charts    🔬 What-If   ⚖️ Compare
                       |
                       ▼
                🧠 SHAP Analysis
                       |
                       ▼
                📈 Explanations
                       |
                       ▼
                📄 PDF Report
                       |
                       ▼
                📥 Download Result

🧩 Architecture Components
Component	Responsibility
Presentation Layer	Collects flight details and displays results through Streamlit.
Feature Engineering	Creates derived features from the selected flight information.
Preprocessing Layer	Transforms inputs using the saved preprocessing pipeline.
Prediction Layer	Generates fare estimates using the trained XGBoost model.
Explainability Layer	Calculates SHAP contributions for individual predictions.
Analytics Layer	Supports scenario analysis, comparison, and interactive charts.
Reporting Layer	Generates downloadable PDF reports using ReportLab.
🧠 Machine Learning Pipeline

RayAI uses a trained XGBoost Regressor to estimate flight ticket prices.

🔄 Prediction Workflow
Flight Details
      |
      ▼
Input Validation
      |
      ▼
Feature Engineering
      |
      ▼
Pandas DataFrame
      |
      ▼
Saved Preprocessing Pipeline
      |
      ▼
Transformed Feature Matrix
      |
      ▼
XGBoost Regression Model
      |
      ▼
Predicted Fare
      |
      ▼
Non-Negative Price Constraint
      |
      ▼
SHAP Explanation
      |
      ▼
Dashboard Results

⚙️ Model Artifacts

The application loads two saved artifacts:

Artifact	Purpose
artifacts/preprocessor.joblib	Transforms input features into the representation expected by the trained model.
artifacts/flight_price_xgb.json	Stores the trained XGBoost regression model.

These artifacts must be compatible with the feature definitions and model configuration used during training.

The preprocessing pipeline and model are loaded when the application initializes and reused for subsequent predictions.

🧪 Feature Engineering

RayAI derives additional features from the flight information provided by the user.

Feature	Description
route	Combines source and destination cities.
booking_window	Categorizes the number of days remaining before departure.
duration_category	Groups flight duration into predefined categories.
departure_arrival	Combines departure-time and arrival-time categories.
class_stops	Combines travel class and stop category.
duration	Retains flight duration as a numerical feature.
days_left	Retains days remaining before departure as a numerical feature.
📅 Booking-Window Categories
Days Before Departure	Category
1–3 days	Last_Minute
4–7 days	Very_Near
8–15 days	Near
16–30 days	Medium
31–60 days	Early
More than 60 days	Very_Early
⏱️ Flight Duration Categories
Flight Duration	Category
Up to 2 hours	Short
More than 2, up to 5 hours	Medium
More than 5, up to 8 hours	Long
More than 8, up to 12 hours	Very_Long
More than 12 hours	Ultra_Long

These derived features allow the model to use both numerical and categorical flight characteristics.

Important: The engineered feature names, category boundaries, and preprocessing behavior must match those used when training the saved model.

🔬 What-If Scenario Analysis

The what-if simulator lets users investigate how changing flight characteristics affects the predicted fare.

Example Workflow
             ✈️ Baseline Flight
                     |
          ┌──────────┼──────────┐
          |          |          |
          ▼          ▼          ▼
      Scenario A  Scenario B  Scenario C
          |          |          |
          ▼          ▼          ▼
       Predict     Predict     Predict
          |          |          |
          └──────────┼──────────┘
                     |
                     ▼
             📊 Compare Fares


For example, a user can compare the model's predictions for the same baseline flight with different booking lead times.

Scenario	Days Left	Purpose
A	3	Explore a last-minute booking.
B	15	Explore a nearer-term booking.
C	45	Explore an earlier booking.

The table illustrates example scenario inputs, not actual model predictions.

The simulator helps explore model behavior, but its results do not guarantee that booking earlier or later will produce the same price change in the real market.

⚖️ Flight Comparison

The comparison module evaluates three flight configurations using the same selected route and departure-time category.

Comparison Workflow
Shared Route Configuration
             |
       ┌─────┼─────┐
       ▼     ▼     ▼
    Flight  Flight  Flight
       A      B      C
       |      |      |
       ▼      ▼      ▼
    Model   Model   Model
       |      |      |
       └──────┼──────┘
              ▼
       📊 Price Comparison
              |
              ▼
       📋 Comparison Table


Users can compare predicted prices across supported combinations of airline, class, stops, duration, and booking window.

The resulting values are model estimates rather than live airline offers.

🧠 SHAP Explainability

SHAP stands for SHapley Additive exPlanations. It provides a framework for understanding how individual features contribute to a machine learning prediction.

RayAI uses SHAP to explain the factors influencing a selected flight's predicted fare.

🔄 Explanation Workflow
Flight Input
     |
     ▼
Preprocessing
     |
     ▼
XGBoost Model
     |
     ▼
Predicted Fare
     |
     ▼
SHAP Explainer
     |
     ▼
Feature Contributions
     |
     ▼
📊 Explanation Chart

📊 Interpreting SHAP Values
SHAP Result	Interpretation
Positive contribution	Pushes the prediction higher relative to the explainer's expected output.
Negative contribution	Pushes the prediction lower relative to the explainer's expected output.
Larger absolute contribution	Represents a stronger contribution to that individual prediction.

The dashboard displays the ten features with the largest absolute SHAP contributions.

SHAP helps explain the model's behavior for an individual input. It does not establish causation or guarantee that real-world prices will change in the same way.

📊 Model Performance

The dashboard currently displays the following configured evaluation metrics.

Metric	Displayed Value
Algorithm	XGBoost Regressor
Number of Estimators	500
Maximum Tree Depth	10
Mean Absolute Error (MAE)	₹1,469.82
Root Mean Squared Error (RMSE)	₹2,747.83
R² Score	0.9854
Understanding the Metrics
MAE: Measures the average absolute difference between actual and predicted prices.
RMSE: Measures prediction error while penalizing larger errors more strongly.
R² Score: Measures the proportion of target variance explained by the model relative to a mean-prediction baseline.

The values above are configured in the dashboard source code. They should be verified against the actual evaluation results before being presented as independently validated model performance.

A high R² score alone does not establish how well a model generalizes to unseen flights or future market conditions.

🛠️ Technology Stack
Technology	Role
🐍 Python	Core programming language
🎨 Streamlit	Interactive dashboard
🤖 XGBoost	Flight price regression
🐼 Pandas	Input data preparation
🔢 NumPy	Numerical computations
⚙️ Scikit-learn	Preprocessing pipeline compatibility
🧠 SHAP	Model explainability
📊 Plotly	Interactive visualizations
📦 Joblib	Loading the saved preprocessing artifact
📄 ReportLab	PDF report generation
📁 Project Structure
RayAI-Smart-Flight-Pricing/
│
├── app.py
│
├── artifacts/
│   ├── preprocessor.joblib
│   └── flight_price_xgb.json
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE

📂 File Descriptions
File	Purpose
app.py	Main Streamlit application.
artifacts/preprocessor.joblib	Saved preprocessing pipeline.
artifacts/flight_price_xgb.json	Trained XGBoost model.
requirements.txt	Python dependency list.
README.md	Project documentation.
.gitignore	Excludes unnecessary or sensitive files.
LICENSE	Defines usage and distribution terms, if included.

The trained artifacts are required to run the application. Include them in the repository only if you have permission to distribute them.

💻 Installation
Prerequisites

Ensure that the following are installed:

Python
pip
Git
The trained preprocessing and model artifacts
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY


Replace YOUR_USERNAME and YOUR_REPOSITORY with your actual GitHub username and repository name.

2. Create a Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate


Linux / macOS

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

Create a requirements.txt file containing:

streamlit
pandas
numpy
plotly
shap
xgboost
joblib
scikit-learn
reportlab


Install the dependencies:

pip install -r requirements.txt


For reproducible deployments, pin compatible dependency versions after testing the application with the saved artifacts.

4. Verify Model Artifacts

Ensure that both files exist:

artifacts/preprocessor.joblib
artifacts/flight_price_xgb.json


The preprocessor must accept the engineered input features, and its transformed output must be compatible with the trained XGBoost model.

▶️ Usage
Start the Application

Run the following command from the project root:

streamlit run app.py


Open the local URL displayed in your terminal. The default is usually:

http://localhost:8501

✈️ Generate a Flight Price Prediction
Open the Flight Price Prediction tab.
Select the airline, route, departure and arrival categories, stops, and travel class.
Enter flight duration and days remaining before departure.
Generate the prediction.
Review the estimated fare and SHAP explanation.
🔬 Explore Alternative Scenarios
Open the What-If Simulator.
Choose an alternative booking window, airline, class, stop configuration, or duration.
Compare the resulting predictions with the baseline.
⚖️ Compare Flights
Open Flight Comparison.
Configure three flight scenarios.
Compare their predicted fares in the chart and comparison table.
📄 Download a Prediction Report
Generate a prediction in the Prediction tab.
Open Prediction Report.
Generate and download the PDF report.
🤖 Review Model Information

Open Model Info to review the configured model parameters, evaluation metrics, and explainability guidance.

🔐 Security and Artifact Management

Recommended .gitignore entries:

venv/
.venv/
__pycache__/
*.pyc
.env
.DS_Store


If model artifacts or datasets are private or licensed, exclude them as appropriate:

artifacts/
data/


Only exclude artifacts/ if your deployment process provides the required files separately.

Never commit credentials, private datasets, or model artifacts you are not authorized to distribute.

⚠️ Limitations
The application requires the saved preprocessing and XGBoost model artifacts.
Prediction quality depends on the training data, feature definitions, and model generalization.
Categorical values must be compatible with the saved preprocessing pipeline.
What-if results are model-generated estimates, not verified market prices.
The application does not retrieve live fares or real-time booking availability.
Configured evaluation metrics should be verified against the original evaluation dataset.
Supported airlines and cities are limited to the options defined in the application.
Routes use selected city names rather than a live airport-routing service.
SHAP contributions explain model behavior and do not prove causation.
Predictions are not guaranteed to reflect future ticket prices.
🔮 Future Improvements

Potential enhancements include:

🌐 Integration with live flight-search and fare APIs.
📅 Historical fare analysis and price-trend forecasting.
📍 Airport-level routing and geographic visualization.
📈 Prediction intervals and uncertainty estimates.
🧪 Automated testing for feature engineering and prediction logic.
📊 Model residual analysis and evaluation visualizations.
🔄 Automated model retraining and monitoring.
📦 Dependency version pinning and reproducible environments.
☁️ Deployment to Streamlit Community Cloud or another hosting platform.
✈️ Expanded airline, airport, and route coverage.
🤝 Contributing

Contributions, bug reports, and feature suggestions are welcome.

Fork the repository.
Create a feature branch:
   git checkout -b feature/your-feature

Implement your changes.
Test the application.
Commit your changes:
   git add .
   git commit -m "Add your feature"

Push your branch:
   git push origin feature/your-feature

Open a Pull Request describing your changes.
📜 License

Add a LICENSE file if you intend to distribute this project under an open-source license.

Ensure that you have the appropriate rights to distribute any included datasets, trained models, and other project artifacts.

⚠️ Disclaimer

RayAI is a machine learning demonstration and analytical tool. Predicted fares are model-generated estimates and are not guaranteed future ticket prices, live airline fares, or booking offers.

Always verify actual prices with airlines or authorized travel booking providers before making a purchase.

🙌 Acknowledgements

Built with:

Python
Streamlit
XGBoost
Pandas and NumPy
Scikit-learn
SHAP
Plotly
Joblib
ReportLab

<p align="center"> <strong>✈️ RayAI — Smart Flight Pricing</strong> <br> Predict. Compare. Explain. <br><br> Built with Python · Streamlit · XGBoost · Plotly · SHAP </p>

<p align="center"> <a href="#-rayai--smart-flight-pricing">⬆️ Back to Top</a> </p>