# ✈️ RayAI — Smart Flight Pricing

<p align="center"> <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"> <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"> <img src="https://img.shields.io/badge/XGBoost-Regression-189F5B?style=for-the-badge" alt="XGBoost"> <img src="https://img.shields.io/badge/SHAP-ExplainableAI-7B2CBF?style=for-the-badge" alt="SHAP"> <img src="https://img.shields.io/badge/Plotly-InteractiveCharts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly"> <img src="https://img.shields.io/badge/Pandas-DataProcessing-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"> <img src="https://img.shields.io/badge/Scikit--learn-Preprocessing-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn"> <img src="https://img.shields.io/badge/ReportLab-PDFReports-B22222?style=for-the-badge" alt="ReportLab"> </p>

<p align="center"> <strong>Predict flight fares. Compare scenarios. Understand AI predictions.</strong> </p>

<p align="center"> An interactive flight price prediction application built with <strong>Python, Streamlit, XGBoost, SHAP, and Plotly.</strong> </p>

<p align="center"> <a href="#-overview">Overview</a> • <a href="#-features">Features</a> • <a href="#-architecture">Architecture</a> • <a href="#-machine-learning-pipeline">Machine Learning</a> • <a href="#-shap-explainability">SHAP</a> • <a href="#-installation">Installation</a> • <a href="#️-usage">Usage</a> </p>

## 📌 Overview

RayAI — Smart Flight Pricing is an interactive machine learning application that predicts estimated flight ticket prices in Indian Rupees (INR).

Powered by a trained XGBoost regression model, the application allows users to explore how flight characteristics such as airline, route, travel class, number of stops, flight duration, and booking lead time influence predicted fares.

RayAI goes beyond basic prediction by combining what-if simulation, three-flight comparison, SHAP explainability, interactive Plotly visualizations, and downloadable PDF reports in a single Streamlit dashboard.

The application loads a saved preprocessing pipeline and trained model to process flight information and generate predictions. SHAP helps explain individual predictions by showing how model features contribute to the estimated fare.

> ⚠️ **Important:** RayAI currently generates machine learning estimates. It does not fetch live airline prices, verify ticket availability, or guarantee actual booking prices.

## 🎯 What It Can Do

RayAI brings flight price prediction and analysis into one interactive dashboard, allowing users to estimate ticket fares, experiment with alternative booking scenarios, and compare three different flight configurations. Users can also explore how booking lead time, airlines, travel classes, routes, and stop configurations relate to predicted prices. The application provides SHAP-based explanations to help interpret individual predictions, interactive Plotly charts for visual analysis, and downloadable PDF reports containing flight details, estimated fares, and important feature contributions. Behind the scenes, a saved preprocessing pipeline and trained XGBoost model handle feature transformation and prediction, while engineered features help represent route information, booking windows, and flight duration.

## ✨ Features

| Feature                    | Description                                                                |
| -------------------------- | -------------------------------------------------------------------------- |
| ✈️ Flight Price Prediction | Estimates ticket prices in INR using a trained XGBoost regressor.          |
| 🔬 What-If Simulator       | Tests alternative flight configurations while keeping other inputs fixed.  |
| ⚖️ Flight Comparison       | Compares predicted fares for three flight scenarios.                       |
| 🧠 SHAP Explainability     | Identifies features contributing positively or negatively to a prediction. |
| 📊 Interactive Charts      | Visualizes predicted fares using Plotly.                                   |
| 🏙️ Route Analysis         | Combines source and destination cities into a route feature.               |
| 📅 Booking-Window Analysis | Categorizes days remaining before departure.                               |
| ⏱️ Duration Categorization | Groups flight duration into predefined categories.                         |
| ⚙️ Feature Engineering     | Creates derived features from raw flight information.                      |
| 📄 PDF Reports             | Exports flight details, predicted fares, and leading SHAP contributions.   |
| 🤖 Model Information       | Displays configured model parameters and reported evaluation metrics.      |
| 📦 Saved Model Artifacts   | Reuses the trained model and preprocessing pipeline.                       |
| 🎨 Interactive Dashboard   | Provides dedicated Streamlit modules for prediction and analysis.          |

## 🏗️ Architecture

RayAI follows a modular prediction workflow that connects user input, feature engineering, preprocessing, model inference, explainability, visualization, and report generation.

### System Architecture

```
flowchart TD
    A["👤 User"] --> B["🎨 Streamlit Dashboard"]
    B --> C["✈️ Flight Details"]
    C --> D["✅ Input Validation"]
    D --> E["⚙️ Feature Engineering"]
    E --> F["📋 Pandas DataFrame"]
    F --> G["🔄 Saved Preprocessor"]
    G --> H["🤖 XGBoost Model"]
    H --> I["💰 Predicted Fare"]
    I --> J["🔬 What-If Simulator"]
    I --> K["⚖️ Flight Comparison"]
    I --> L["📊 Plotly Charts"]
    I --> M["🧠 SHAP Explainer"]
    M --> N["📈 Feature Contributions"]
    I --> O["📄 PDF Generator"]
    N --> O
    O --> P["📥 Download Report"]
    Q["📦 Saved Model Artifacts"] --> G
    Q --> H

```

## 🔄 Complete Application Flow

```
                    👤 USER
                       │
                       ▼
              🎨 Streamlit Dashboard
                       │
                       ▼
                ✈️ Flight Details
                       │
                       ▼
                ✅ Input Validation
                       │
                       ▼
               ⚙️ Feature Engineering
                       │
                       ▼
                📋 Pandas DataFrame
                       │
                       ▼
              🔄 Saved Preprocessor
                       │
                       ▼
               🤖 XGBoost Regressor
                       │
                       ▼
                 💰 Predicted Fare
                       │
            ┌──────────┼──────────┐
            │          │          │
            ▼          ▼          ▼
       🔬 What-If   ⚖️ Compare  📊 Charts
            │          │          │
            └──────────┼──────────┘
                       │
                       ▼
                 🧠 SHAP Analysis
                       │
                       ▼
               📈 Feature Insights
                       │
                       ▼
                📄 PDF Generation
                       │
                       ▼
                📥 Download Report

```

## 🖥️ Dashboard Modules

### ✈️ 1. Flight Price Prediction

The prediction module estimates the price of a flight based on the selected flight characteristics.

**Input parameters**

- Airline
- Source city
- Destination city
- Departure-time category
- Arrival-time category
- Number of stops
- Travel class
- Flight duration
- Days remaining before departure

**Prediction results**

- Estimated fare in INR
- Selected flight and route summary
- Booking-window category
- SHAP feature contribution visualization
- Predicted prices for selected alternative scenarios, where supported

```
Flight Details
      ↓
Feature Engineering
      ↓
Preprocessing Pipeline
      ↓
XGBoost Regressor
      ↓
Estimated Fare
      ↓
SHAP Explanation
      ↓
Dashboard Results

```

### 🔬 2. What-If Simulator

The what-if simulator allows users to experiment with alternative flight configurations and observe how the model's predicted fare changes.

Users can modify supported characteristics such as booking lead time, airline, travel class, number of stops, and flight duration while keeping other inputs fixed.

For example, a user can compare a flight booked 3, 15, or 45 days before departure to investigate how the model responds to different booking windows.

```
✈️ Baseline Flight
        │
        ▼
Select Variable to Change
        │
   ┌────┼────┐
   ▼    ▼    ▼
   A    B    C
   │    │    │
   ▼    ▼    ▼
Predict Each Scenario
        │
        ▼
Compare Estimated Fares

```

The simulator demonstrates model behavior under modified inputs. It does not establish that changing a booking decision will cause the same price change in the real market.

### ⚖️ 3. Flight Comparison

The comparison module evaluates three flight configurations using a shared route and departure-time category.

Users can explore differences in airline, travel class, stop configuration, flight duration, and booking lead time.

The results can be presented through an interactive chart, a comparison table, and relative differences between the predicted fares.

```
Shared Route
     │
     ├──────────┬──────────┐
     ▼          ▼          ▼
  Flight A   Flight B   Flight C
     │          │          │
     ▼          ▼          ▼
   Model A    Model B    Model C
     │          │          │
     └──────────┼──────────┘
                ▼
       📊 Fare Comparison

```

### 📄 4. Prediction Reports

RayAI supports downloadable PDF reports generated with ReportLab.

A report can include the selected flight configuration, route information, estimated fare, booking-window details, leading SHAP feature contributions, and a prediction disclaimer.

```
Flight Details
      ↓
Predicted Fare
      ↓
SHAP Contributions
      ↓
ReportLab
      ↓
PDF Document
      ↓
Download Report

```

PDF reports record model estimates and explanations. They are not airline booking confirmations or guarantees of ticket availability.

### 🤖 5. Model Information

The model information module displays configured model parameters, reported evaluation metrics, saved model artifacts, and guidance for interpreting predictions.

This helps users understand the model configuration and the limitations of its reported performance.

## 🧠 Machine Learning Pipeline

RayAI uses a trained XGBoost Regressor to estimate flight ticket prices.

XGBoost is a gradient-boosted decision tree algorithm that combines multiple trees to learn relationships between input features and a numerical target.

The application uses the saved preprocessing pipeline to transform engineered flight information into the representation expected by the trained model.

### 🔄 Prediction Workflow

```
flowchart TD
    A["✈️ Flight Details"] --> B["✅ Input Validation"]
    B --> C["⚙️ Feature Engineering"]
    C --> D["📋 Input DataFrame"]
    D --> E["🔄 Saved Preprocessor"]
    E --> F["📦 Transformed Features"]
    F --> G["🤖 XGBoost Regressor"]
    G --> H["💰 Predicted Fare"]
    H --> I["🧠 SHAP Explanation"]
    I --> J["📊 Dashboard Results"]

```

### 🧪 Pipeline Stages

1. **Input collection:** The user selects flight characteristics through Streamlit.
2. **Input validation:** The application checks the supplied values against its supported inputs.
3. **Feature engineering:** The application derives features such as route, booking window, and duration category.
4. **Preprocessing:** The saved preprocessing pipeline transforms the engineered input DataFrame.
5. **Prediction:** The transformed features are passed to the trained XGBoost model.
6. **Price handling:** The application applies any configured prediction constraints.
7. **Explainability:** SHAP calculates feature contributions using the configured explanation method.
8. **Visualization:** The dashboard presents the predicted fare and associated analytical results.

The exact validation rules and prediction constraints depend on the implementation in `app.py`.

### 📦 Model Artifacts

| Artifact                          | Purpose                            |
| --------------------------------- | ---------------------------------- |
| `artifacts/preprocessor.joblib`   | Stores the preprocessing pipeline. |
| `artifacts/flight_price_xgb.json` | Stores the trained XGBoost model.  |

Both artifacts must be compatible with the application's feature engineering logic.

```
Application Startup
        │
        ▼
Check Model Artifacts
        │
        ├── preprocessor.joblib
        │
        └── flight_price_xgb.json
        │
        ▼
Load Preprocessor and Model
        │
        ▼
Initialize Prediction Workflow
        │
        ▼
Generate Predictions

```

> ⚠️ The preprocessing pipeline, model, feature names, categorical values, and transformed feature representation must match the training configuration.

## ⚙️ Feature Engineering

Feature engineering transforms raw flight information into additional features that help the model represent relationships between flight characteristics.

| Feature             | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| `route`             | Combines source and destination cities.                         |
| `booking_window`    | Categorizes days remaining before departure.                    |
| `duration_category` | Groups flight duration into predefined categories.              |
| `departure_arrival` | Combines departure and arrival categories.                      |
| `class_stops`       | Combines travel class and stop configuration.                   |
| `duration`          | Retains flight duration as a numerical feature.                 |
| `days_left`         | Retains days remaining before departure as a numerical feature. |

### 📅 Booking-Window Categories

| Days Before Departure | Category      |
| --------------------- | ------------- |
| 1–3 days              | `Last_Minute` |
| 4–7 days              | `Very_Near`   |
| 8–15 days             | `Near`        |
| 16–30 days            | `Medium`      |
| 31–60 days            | `Early`       |
| More than 60 days     | `Very_Early`  |

### ⏱️ Flight Duration Categories

| Flight Duration             | Category     |
| --------------------------- | ------------ |
| Up to 2 hours               | `Short`      |
| More than 2, up to 5 hours  | `Medium`     |
| More than 5, up to 8 hours  | `Long`       |
| More than 8, up to 12 hours | `Very_Long`  |
| More than 12 hours          | `Ultra_Long` |

These category boundaries are the documented configuration and should be verified against the actual feature engineering implementation.

## 🧠 SHAP Explainability

SHAP stands for SHapley Additive exPlanations. It provides a framework for explaining machine learning predictions by estimating the contribution of individual features.

RayAI uses SHAP to help users understand which features contribute to a selected flight's predicted fare.

Rather than displaying only a numerical estimate, the dashboard can show which features push a prediction higher or lower relative to the explainer's expected output.

### 🔄 SHAP Workflow

```
Flight Input
     ↓
Feature Engineering
     ↓
Preprocessing
     ↓
XGBoost Model
     ↓
Predicted Fare
     │
     ├──────→ SHAP Explainer
     │               │
     │               ▼
     │       Feature Contributions
     │               │
     └───────────────┤
                     ▼
              Explanation Chart

```

### 📊 Interpreting SHAP Values

| SHAP Result                  | Interpretation                                                                                     |
| ---------------------------- | -------------------------------------------------------------------------------------------------- |
| Positive contribution        | Pushes the prediction higher relative to the explainer's expected output.                          |
| Negative contribution        | Pushes the prediction lower relative to the explainer's expected output.                           |
| Larger absolute contribution | Indicates a stronger contribution to the individual prediction under the chosen explanation setup. |

The dashboard can display the features with the largest absolute SHAP contributions.

SHAP explains model behavior, not real-world causation. Its interpretation depends on the explainer's baseline, feature representation, and compatibility with the prediction pipeline.

## 📊 Model Performance

The following values are reported in the supplied dashboard configuration.

| Metric                         | Reported Value    |
| ------------------------------ | ----------------- |
| Algorithm                      | XGBoost Regressor |
| Number of Estimators           | 500               |
| Maximum Tree Depth             | 10                |
| Mean Absolute Error (MAE)      | ₹1,469.82         |
| Root Mean Squared Error (RMSE) | ₹2,747.83         |
| R² Score                       | 0.9854            |

### 📐 Understanding the Metrics

- **Mean Absolute Error (MAE)** measures the average absolute difference between actual and predicted fares. Lower values generally indicate smaller average errors.
- **Root Mean Squared Error (RMSE)** measures prediction error while penalizing larger errors more strongly than MAE. Lower values generally indicate better performance on the evaluated dataset.
- **R² Score** measures how well the model explains variation in the target relative to a mean-prediction baseline. A higher score can indicate a better fit, but it does not independently guarantee accurate predictions for unseen flights.

> ⚠️ These are reported configuration values, not independently verified evaluation results. Confirm them against the original held-out evaluation data before making performance claims. Data leakage and unsuitable train/test splits can inflate apparent accuracy.

## 📊 Interactive Visualizations

RayAI uses Plotly to provide interactive visual analysis of predicted fares and alternative flight configurations.

Supported visualization areas include:

- 📈 Predicted-fare comparisons across scenarios
- 📅 Booking-window analysis
- ✈️ Airline comparisons
- 🎟️ Travel-class comparisons
- 🛫 Stop-configuration analysis
- 🔬 What-if prediction differences
- 🧠 SHAP feature contribution charts

```
Flight Configurations
        ↓
Model Predictions
        ↓
Structured Results
        ↓
Plotly Charts
        ↓
Interactive Dashboard

```

These visualizations represent model outputs. They should not be interpreted as historical fare trends unless the application is supplied with appropriate historical pricing data.

## 📄 PDF Report Generation

RayAI uses ReportLab to generate downloadable prediction reports.

### Report Contents

- Selected flight characteristics
- Source and destination information
- Estimated ticket fare
- Booking-window details
- Leading SHAP feature contributions
- Prediction disclaimer

### Generation Workflow

```
Flight Configuration
        ↓
Predicted Fare
        ↓
SHAP Contributions
        ↓
ReportLab
        ↓
PDF Document
        ↓
Download

```

PDF reports record model estimates and explanations. They are not airline booking confirmations or guarantees of ticket availability.

## 🧰 Technology Stack

| Technology      | Role                                  |
| --------------- | ------------------------------------- |
| 🐍 Python       | Core programming language             |
| 🎨 Streamlit    | Interactive dashboard                 |
| 🤖 XGBoost      | Flight fare regression                |
| 🐼 Pandas       | DataFrames and data preparation       |
| 🔢 NumPy        | Numerical computations                |
| ⚙️ Scikit-learn | Preprocessing pipeline compatibility  |
| 🧠 SHAP         | Model explainability                  |
| 📊 Plotly       | Interactive charts                    |
| 📦 Joblib       | Loading saved preprocessing artifacts |
| 📄 ReportLab    | PDF report generation                 |

## 📁 Project Structure

```
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

```

| File                              | Purpose                                            |
| --------------------------------- | -------------------------------------------------- |
| `app.py`                          | Main Streamlit application.                        |
| `artifacts/preprocessor.joblib`   | Saved preprocessing pipeline.                      |
| `artifacts/flight_price_xgb.json` | Trained XGBoost model.                             |
| `requirements.txt`                | Python dependency list.                            |
| `README.md`                       | Project documentation.                             |
| `.gitignore`                      | Excludes unnecessary or sensitive files.           |
| `LICENSE`                         | Defines usage and distribution terms, if included. |

Update this structure if your actual repository contains additional modules or different filenames.

## 🔐 Environment and Model Artifacts

The supplied configuration does not specify any required API keys or external service credentials.

Before launching the application, ensure that both saved model artifacts are present:

```
artifacts/
├── preprocessor.joblib
└── flight_price_xgb.json

```

The model artifacts must be compatible with each other and with the feature engineering logic used by the application.

If the artifacts are excluded from GitHub, document how to obtain them and where to place them.

Avoid loading untrusted Joblib or pickle-based files because deserialization can execute malicious code.

## 🚀 Installation

### 1. Clone the Repository

```
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your actual GitHub username and repository name.

### 2. Create a Virtual Environment

**Windows**

```
python -m venv venv
venv\Scripts\activate

```

**Linux / macOS**

```
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

Ensure that `requirements.txt` contains the packages used by the application.

```
streamlit
pandas
numpy
plotly
shap
xgboost
joblib
scikit-learn
reportlab

```

Install the dependencies:

```
python -m pip install --upgrade pip
pip install -r requirements.txt

```

For reproducible installations, pin compatible package versions after testing the application with the saved model artifacts.

### 4. Verify Model Artifacts

Confirm that the required files exist:

```
artifacts/preprocessor.joblib
artifacts/flight_price_xgb.json

```

### 5. Launch the Application

```
streamlit run app.py

```

The application should display a local URL in the terminal. The default address is usually:

`http://localhost:8501`

## ▶️ Usage

### ✈️ Generate a Flight Price Prediction

1. Launch the Streamlit application.
2. Open the Flight Price Prediction module.
3. Select an airline, source city, and destination city.
4. Configure departure time, arrival time, stops, and travel class.
5. Enter flight duration and days remaining before departure.
6. Generate the prediction.
7. Review the estimated fare and SHAP explanation.

### 🔬 Explore Alternative Scenarios

1. Open the What-If Simulator.
2. Select a baseline flight configuration.
3. Change one or more supported flight characteristics.
4. Generate alternative predictions.
5. Compare the estimated fares against the baseline.

**Example booking-window inputs:**

```
Baseline Flight
      │
      ├── Days Left: 3
      ├── Days Left: 15
      └── Days Left: 45
             │
             ▼
       Model Predictions
             │
             ▼
       Compare Estimated Fares

```

These values illustrate scenario inputs, not actual predicted prices.

### ⚖️ Compare Three Flights

1. Open Flight Comparison.
2. Select the shared source city and destination city.
3. Choose the shared departure-time category.
4. Configure Flight A, Flight B, and Flight C.
5. Generate the comparison.
6. Review the chart and comparison table.

### 📄 Download a Prediction Report

1. Generate a flight price prediction.
2. Open Prediction Report.
3. Generate the PDF.
4. Download the report.
5. Review the flight details, estimated fare, SHAP contributions, and disclaimer.

### 🤖 Review Model Information

Open Model Info to inspect the configured XGBoost parameters, reported evaluation metrics, model artifacts, and explainability guidance.

## 🧪 Testing and Validation

Before deploying RayAI, verify the complete prediction workflow.

Recommended checks:

- Confirm that both model artifacts load successfully.
- Test feature engineering with representative and boundary inputs.
- Verify that the input DataFrame contains the expected feature names.
- Confirm that preprocessing produces the correct model input representation.
- Check that predictions are finite and handled appropriately.
- Verify that SHAP explanations correspond to the prediction being displayed.
- Test all three flight comparison scenarios.
- Confirm that generated PDF reports match the selected flight configuration.
- Compare reported metrics with the original evaluation results.
- Test missing artifacts and unsupported categorical values.

For reliable performance evaluation, use a held-out dataset and investigate possible data leakage before reporting accuracy claims.

## 🔒 Security

Recommended `.gitignore`:

```
# Python
__pycache__/
*.py[cod]

# Virtual environments
venv/
.venv/

# Environment files
.env
.env.*

# IDE files
.vscode/
.idea/

# Operating system files
.DS_Store
Thumbs.db

# Optional private model artifacts
# artifacts/

# Optional private datasets
# data/

```

### Security Recommendations

- Never commit API keys, passwords, or private credentials.
- Exclude private datasets from version control.
- Distribute model artifacts only when you have permission.
- Avoid loading untrusted serialized model or preprocessing files.
- Keep dependencies updated and test compatibility before deployment.

If the model artifacts are required to run the application, ensure users know how to obtain them when they are excluded from the repository.

## ⚠️ Limitations

- The application requires compatible saved preprocessing and XGBoost model artifacts.
- Prediction quality depends on the training data and model generalization.
- The application does not currently retrieve live flight fares or real-time booking availability.
- Supported airlines and cities depend on the options defined in the application.
- What-if comparisons show model responses rather than guaranteed real-world price changes.
- SHAP contributions explain model behavior but do not establish causation.
- Reported evaluation metrics should be verified against the original evaluation data.
- Historical fare trends cannot be established from predictions alone.
- Airline inventory, seasonal demand, competition, and market changes may not be represented in the model.
- A high R² score does not guarantee accurate predictions for unseen flights.

## 🔮 Future Improvements

### ✈️ Flight Data and Pricing

- 🌐 Integration with live flight-search and fare APIs
- 📅 Historical fare analysis and price-trend forecasting
- 📍 Airport-level routing and geographic visualization
- 🔔 Fare alerts and price-change notifications
- 🌍 Expanded airline, airport, and route coverage

### 🧠 Machine Learning

- 📊 Prediction intervals and uncertainty estimates
- 🧪 Automated tests for feature engineering and prediction logic
- 📉 Residual analysis and model diagnostics
- 🔄 Automated model retraining and monitoring
- 📅 Time-aware validation and temporal holdout evaluation
- 🧠 Improved SHAP visualization and explanation reliability

### 🎨 Application and Deployment

- 🎨 Enhanced dashboard design and responsive layouts
- 📄 Customizable PDF report templates
- ☁️ Deployment to Streamlit Community Cloud or another hosting platform
- 🐳 Docker support
- 🚀 GitHub Actions CI/CD
- 📊 Prediction history and saved comparisons
- 📥 CSV export for comparison results
- 🧪 Automated application testing

## 🤝 Contributing

Contributions, bug reports, and feature suggestions are welcome!

### 1. Fork the Repository

Click **Fork** on GitHub.

### 2. Clone the Repository

```
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

```

### 3. Create a Feature Branch

```
git checkout -b feature/your-feature

```

### 4. Make Changes

Improve the dashboard, refine feature engineering, add a visualization, or contribute another enhancement.

### 5. Test and Commit

```
git add .
git commit -m "Add your feature"

```

### 6. Push Your Branch

```
git push origin feature/your-feature

```

Open a Pull Request describing your changes.

## 📜 License

Add a `LICENSE` file if you intend to distribute RayAI under an open-source license.

Ensure that you have the appropriate rights to distribute any included datasets, trained models, dependencies, and other project artifacts.

## 🙌 Acknowledgements

Built with:

- 🐍 Python
- 🎨 Streamlit
- 🤖 XGBoost
- 🐼 Pandas and NumPy
- ⚙️ Scikit-learn
- 🧠 SHAP
- 📊 Plotly
- 📦 Joblib
- 📄 ReportLab

These technologies power RayAI's predictive modeling, data processing, explainability, visualization, and reporting workflows.

## 👨‍💻 Author

**AYUSH GUPTA**

**RayAI — Smart Flight Pricing**

Python • Streamlit • XGBoost • SHAP\
Pandas • NumPy • Scikit-learn\
Plotly • Joblib • ReportLab

Built to explore flight fare prediction through interactive machine learning and explainable AI.

If you find this project useful, consider giving the repository a ⭐ on GitHub!

---

**✈️ RayAI — Smart Flight Pricing**

*Predict. Compare. Explain.*

⬆️ Back to Top
