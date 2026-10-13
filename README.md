✈️ RayAI — Smart Flight Pricing
AI-Powered Flight Fare Prediction, Scenario Analysis & Explainable Machine Learning

<p align="center"> An interactive machine learning dashboard that predicts flight ticket prices using XGBoost Regression, explores alternative booking scenarios, and explains predictions using SHAP. </p>

<p align="center"> <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"> <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"> <img src="https://img.shields.io/badge/XGBoost-Regression-16A34A?style=for-the-badge" alt="XGBoost"> <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-8B5CF6?style=for-the-badge" alt="SHAP"> </p>

<p align="center"> <b>Predict · Compare · Explain</b> </p>

📌 Table of Contents
Overview
Key Features
Dashboard Modules
System Architecture
End-to-End ML Flowchart
Prediction Workflow
Feature Engineering
Technology Stack
Model Performance
Project Structure
Installation and Setup
How to Run
SHAP Explainability
Limitations
Future Enhancements
Contributing
License
🌟 Overview

RayAI — Smart Flight Pricing is an interactive machine learning application designed to estimate flight ticket prices based on flight characteristics and booking information.

Built with Python and Streamlit, the application integrates a trained XGBoost regression model with a saved preprocessing pipeline to generate predictions from user-selected flight details.

Rather than providing only a predicted price, RayAI offers an interactive environment where users can explore alternative flight configurations, compare estimated fares, visualize pricing patterns, and understand the factors influencing individual model predictions.

The application also generates downloadable PDF reports containing flight details, predicted prices, and SHAP-based feature contributions.

🎯 Project Objectives
Predict estimated flight fares using machine learning.
Engineer meaningful features from raw flight information.
Analyze how booking lead time affects model predictions.
Compare different airlines, travel classes, and stop configurations.
Explain individual predictions using SHAP.
Visualize model outputs through interactive Plotly charts.
Generate downloadable PDF prediction reports.
🚀 Key Features
Feature	Description
✈️ Flight Price Prediction	Estimates ticket prices using a trained XGBoost regression model.
🔬 What-If Simulator	Explores alternative scenarios by changing one flight parameter at a time.
⚖️ Flight Comparison	Compares the predicted prices of three flight configurations.
🧠 SHAP Explainability	Shows which processed features push a prediction higher or lower.
📈 Interactive Visualizations	Displays price comparisons across booking windows, classes, stops, and airlines.
📄 PDF Reports	Downloads a report containing flight details, the estimated fare, and leading SHAP factors.
⚙️ Feature Engineering	Generates route, booking-window, duration-category, and combined categorical features.
🎨 Modern Dashboard	Provides a dark-themed, animated, tab-based interface using Streamlit and custom CSS.
🖥️ Dashboard Modules
1. ✈️ Prediction

Enter flight details to generate a price estimate.

The dashboard displays:

Estimated fare in Indian Rupees (INR).
Route and flight summary.
Duration and booking-window information.
SHAP feature contribution chart.
Price versus days before departure.
Predicted price by travel class and number of stops.
Engineered input features.
2. 🔬 What-If Simulator

Explore how model predictions change when one parameter is varied.

Supported parameters:

Days before departure.
Travel class.
Number of stops.
Flight duration.
Airline.

The simulator generates alternative predictions while keeping the other baseline flight characteristics fixed.

3. ⚖️ Flight Comparison

Configure three flight scenarios using a shared source city, destination city, and departure-time category.

Compare their predicted prices through an interactive bar chart and a detailed comparison table.

4. 📄 Prediction Report

Download a PDF report after generating a prediction in the Prediction tab.

The report includes:

Estimated flight price.
Selected flight details.
Route and booking-window information.
Top seven SHAP feature contributions.
A disclaimer explaining that the prediction is an estimate.
5. 🤖 Model Information

Review the configured XGBoost parameters, reported evaluation metrics, preprocessing pipeline, model artifacts, and SHAP interpretation guidance.

🏗️ System Architecture

The following diagram illustrates how the dashboard, preprocessing pipeline, prediction model, explanation engine, visualizations, and PDF reporting functionality interact.

flowchart TD
    A(["👤 User"]) --> B["Streamlit Dashboard"]

    B --> C["Flight Details Input"]
    C --> D["Feature Engineering"]

    D --> E["Input DataFrame"]
    E --> F["Saved Preprocessor<br/>preprocessor.joblib"]

    F --> G["XGBoost Regressor<br/>flight_price_xgb.json"]
    G --> H["Predicted Flight Price"]

    H --> I["Interactive Visualizations"]
    H --> J["What-If Simulator"]
    H --> K["Flight Comparison"]

    F --> L["SHAP Explainer"]
    G --> L
    L --> M["Feature Contribution Analysis"]

    H --> N["PDF Report Generator"]
    M --> N

    I --> O["Results Dashboard"]
    J --> O
    K --> O
    M --> O

    N --> P(["📄 Downloadable PDF"])

    classDef user fill:#172554,stroke:#60a5fa,color:#ffffff
    classDef processing fill:#1e3a8a,stroke:#60a5fa,color:#ffffff
    classDef model fill:#14532d,stroke:#4ade80,color:#ffffff
    classDef explain fill:#581c87,stroke:#c084fc,color:#ffffff
    classDef output fill:#334155,stroke:#94a3b8,color:#ffffff

    class A user
    class B,C,D,E,F processing
    class G,H model
    class L,M explain
    class I,J,K,N,O,P output

Architecture Components
Component	Responsibility
Presentation Layer	Collects user inputs and displays prediction results through Streamlit.
Feature Engineering Layer	Creates additional features from the selected flight characteristics.
Preprocessing Layer	Transforms the input using the saved preprocessing pipeline.
Prediction Layer	Generates fare estimates using the trained XGBoost model.
Explainability Layer	Uses SHAP to calculate feature contributions for individual predictions.
Analytics Layer	Generates interactive charts and alternative scenario comparisons.
Reporting Layer	Creates a downloadable PDF using ReportLab.
🔄 End-to-End ML Flowchart

This flowchart describes the prediction process, from user input to the final dashboard results and optional PDF export.

Note: The preprocessing pipeline and XGBoost model are loaded from saved artifacts when the application initializes. They are reused for subsequent predictions.

flowchart TD
    A(["Start Application"]) --> B["Load Saved Artifacts"]

    B --> C["Load Preprocessor"]
    B --> D["Load XGBoost Model"]

    C --> E["Display Dashboard"]
    D --> E

    E --> F["Enter Flight Information"]
    F --> G["Engineer Features"]

    G --> H["Create Input DataFrame"]
    H --> I["Transform Features"]

    I --> J["Generate Price Prediction"]
    J --> K["Apply Non-Negative Price Constraint"]

    K --> L["Calculate SHAP Values"]
    L --> M["Save Results in Session State"]

    M --> N["Display Predicted Fare"]
    N --> O["Render Charts and Explanations"]

    O --> P{"Download PDF?"}

    P -- Yes --> Q["Generate PDF Report"]
    Q --> R(["Download Report"])

    P -- No --> S(["Continue Exploring Dashboard"])
    R --> S

    classDef start fill:#166534,stroke:#4ade80,color:#ffffff
    classDef input fill:#1e3a8a,stroke:#60a5fa,color:#ffffff
    classDef prediction fill:#14532d,stroke:#4ade80,color:#ffffff
    classDef explain fill:#6b21a8,stroke:#c084fc,color:#ffffff
    classDef decision fill:#92400e,stroke:#fbbf24,color:#ffffff

    class A,R,S start
    class E,F,G,H,I input
    class J,K,N prediction
    class L,M,O,Q explain
    class P decision

⚙️ Prediction Workflow

The application accepts the following input features:

Input	Description
Airline	Selected airline category.
Source City	Departure city.
Destination City	Arrival city.
Departure Time	Departure-time category.
Arrival Time	Arrival-time category.
Stops	Number-of-stops category.
Class	Economy or Business.
Duration	Flight duration in hours.
Days Left	Number of days remaining before departure.

The prediction pipeline follows these steps:

Collect the selected flight characteristics.
Derive additional features using the application's feature-engineering functions.
Construct a Pandas DataFrame with the expected input columns.
Transform the DataFrame using the saved preprocessor.
Pass the transformed features to the XGBoost regressor.
Constrain the returned prediction to a non-negative value.
Calculate SHAP values to explain the individual prediction.
Display the results and generate interactive visualizations.
Create a PDF report when requested.
🧪 Feature Engineering

RayAI derives additional features before applying the saved preprocessing pipeline.

Engineered Feature	Logic
route	Combines source and destination cities.
booking_window	Categorizes the number of days remaining before departure.
duration_category	Groups flight duration into predefined categories.
departure_arrival	Combines departure-time and arrival-time categories.
class_stops	Combines travel class and stop category.
duration	Retains flight duration as a numerical feature.
days_left	Retains days remaining before departure as a numerical feature.
Booking Window Categories
Days Before Departure	Category
1–3 days	Last_Minute
4–7 days	Very_Near
8–15 days	Near
16–30 days	Medium
31–60 days	Early
More than 60 days	Very_Early
Duration Categories
Flight Duration	Category
Up to 2 hours	Short
More than 2, up to 5 hours	Medium
More than 5, up to 8 hours	Long
More than 8, up to 12 hours	Very_Long
More than 12 hours	Ultra_Long

These features enable the model to use both numerical and categorical characteristics of a flight.

🛠️ Technology Stack
Technology	Purpose
Python	Core programming language.
Streamlit	Interactive web dashboard.
XGBoost	Flight price regression model.
Pandas	Input data preparation and DataFrame operations.
NumPy	Numerical computations.
Scikit-learn	Preprocessing pipeline compatibility.
SHAP	Model explainability and feature contributions.
Plotly	Interactive data visualizations.
Joblib	Loading the saved preprocessing artifact.
ReportLab	PDF report generation.
📊 Model Performance

The dashboard currently displays the following configured test-set metrics.

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

Important: These values are configured in the dashboard source code. Verify them against the actual evaluation results before presenting them as independently validated model performance.

📂 Project Structure
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

File or Directory	Purpose
app.py	Main Streamlit application.
artifacts/preprocessor.joblib	Saved preprocessing pipeline.
artifacts/flight_price_xgb.json	Trained XGBoost model.
requirements.txt	Python dependency list.
README.md	Project documentation.
.gitignore	Excludes unnecessary or sensitive files.
LICENSE	Defines usage and distribution terms, if included.

The trained artifacts are required to run the application. Include them in the repository only if you have permission to distribute them.

💻 Installation and Setup
Prerequisites
Python installed on your system.
pip package manager.
Git.
The trained preprocessing and model artifacts.
Step 1: Clone the Repository

Replace the example URL with your actual GitHub repository URL.

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

Step 2: Create a Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate


macOS / Linux

python3 -m venv venv
source venv/bin/activate

Step 3: Install Dependencies

Create a requirements.txt file containing the following initial dependency list:

streamlit
pandas
numpy
plotly
shap
xgboost
joblib
scikit-learn
reportlab


Install the packages:

pip install -r requirements.txt


For reproducible deployments, test the application and pin compatible dependency versions, especially the versions used to train and serialize the model artifacts.

Step 4: Verify the Model Artifacts

Ensure that both files exist at the expected locations:

artifacts/preprocessor.joblib
artifacts/flight_price_xgb.json


The preprocessor must accept the engineered input columns, and its transformed output must be compatible with the saved XGBoost model.

▶️ How to Run

Start the Streamlit application from the project root:

streamlit run app.py


Open the local URL shown in the terminal, usually:

http://localhost:8501

Using the Dashboard
Open the Prediction tab and enter flight details.
Click Predict Flight Price.
Explore the predicted fare, route summary, charts, and SHAP explanations.
Open What-If Simulator to compare alternative configurations.
Open Flight Comparison to compare three flight scenarios.
Open Prediction Report to download the PDF.
Open Model Info to review the model configuration and evaluation metrics.
🧠 SHAP Explainability

SHAP stands for SHapley Additive exPlanations. It provides a framework for interpreting how individual model features contribute to a prediction.

RayAI displays the ten features with the largest absolute SHAP contributions for the selected flight.

SHAP Result	Interpretation
Positive value	Pushes the model prediction higher relative to its expected output.
Negative value	Pushes the model prediction lower relative to its expected output.
Larger absolute value	Indicates a stronger contribution to that prediction.

The SHAP chart helps explain the model's behavior for a particular input.

These contributions are not proof of causation and do not guarantee that real-world ticket prices will respond to a feature in the same way.

🔐 Limitations
The application requires the saved preprocessing and XGBoost model artifacts.
Prediction quality depends on the training data, feature definitions, and model generalization.
Categorical input values must be compatible with the preprocessing pipeline.
What-if scenarios are model-generated estimates, not verified market prices.
The application does not fetch live airline fares or real-time booking availability.
Model metrics should be checked against the original evaluation dataset.
The supported airlines and cities are restricted to the options defined in the application.
Flight routes are represented using selected city names rather than a live airport-routing service.
🔮 Future Enhancements

Potential improvements include:

Integration with live flight-search and fare APIs.
Historical fare analysis and price-trend forecasting.
Support for additional airlines, cities, and travel configurations.
Prediction intervals and uncertainty estimates.
Model evaluation charts and residual analysis.
Automated retraining and model monitoring.
Unit tests for feature engineering and prediction logic.
Dependency version pinning and reproducible deployment.
Deployment to Streamlit Community Cloud or another hosting platform.
Improved route visualization using airport codes and geographic data.
🤝 Contributing

Contributions, bug reports, and feature suggestions are welcome.

Fork this repository.
Create a feature branch.
Implement your changes.
Test the application.
Submit a pull request describing your changes.
📜 License

Add a LICENSE file if you intend to distribute this project under an open-source license.

Ensure that you have the appropriate rights to redistribute any included datasets, trained models, and other project artifacts.

⚠️ Disclaimer

RayAI is a machine learning demonstration and analytical tool. All predicted prices are model-generated estimates and are not guaranteed future ticket prices, live airline fares, or booking offers.

Always verify actual prices with airlines or authorized travel booking providers before making a purchase.

<p align="center"> <b>✈️ RayAI — Smart Flight Pricing</b> <br> Predict. Compare. Explain. <br><br> Built with Python · Streamlit · XGBoost · Plotly · SHAP </p>