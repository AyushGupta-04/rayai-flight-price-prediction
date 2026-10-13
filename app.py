import io
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import xgboost

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(circle at 5% 5%, rgba(37,99,235,.13), transparent 28%),
        radial-gradient(circle at 95% 10%, rgba(124,58,237,.12), transparent 28%),
        #080b12;
    color: #f8fafc;
}

.block-container {
    max-width: 1450px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 30px 35px 8px;
    border-radius: 25px;
    background:
        linear-gradient(135deg, rgba(30,64,175,.28), rgba(88,28,135,.20));
    border: 1px solid rgba(96,165,250,.22);
    margin-bottom: 22px;
    box-shadow: 0 15px 50px rgba(0,0,0,.20);
}

.hero-badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 30px;
    background: rgba(37,99,235,.14);
    border: 1px solid rgba(96,165,250,.25);
    color: #93c5fd;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 12px;
}

.hero h1 {
    font-size: 42px;
    line-height: 1.1;
    margin: 0;
    color: #ffffff;
    font-weight: 850;
    letter-spacing: -1px;
}

.hero p {
    margin-top: 12px;
    color: #aab4c3;
    font-size: 16px;
    max-width: 850px;
    line-height: 1.6;
}

.hero::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -100px;
    top: -130px;
    border-radius: 50%;
    background: rgba(59,130,246,.14);
    filter: blur(60px);
    animation: heroGlow 5s ease-in-out infinite alternate;
    pointer-events: none;
}

@keyframes heroGlow {
    0% { transform: scale(.9); opacity: .5; }
    100% { transform: scale(1.2); opacity: 1; }
}

.flight-animation {
    position: relative;
    height: 55px;
    width: 100%;
    overflow: hidden;
    margin-top: 8px;
}

.flight-track {
    position: absolute;
    left: -70px;
    top: 10px;
    font-size: 30px;
    animation: flyAcross 12s linear infinite;
    filter: drop-shadow(0 0 8px rgba(96,165,250,.5));
}

@keyframes flyAcross {
    0% {
        transform: translateX(0) translateY(8px) rotate(-3deg);
        opacity: 0;
    }
    8% { opacity: 1; }
    45% {
        transform: translateX(45vw) translateY(-3px) rotate(2deg);
    }
    75% {
        transform: translateX(75vw) translateY(4px) rotate(-1deg);
    }
    92% { opacity: 1; }
    100% {
        transform: translateX(110vw) translateY(-5px) rotate(2deg);
        opacity: 0;
    }
}

.cloud {
    position: absolute;
    opacity: .08;
    font-size: 28px;
    animation: cloudMove 25s linear infinite;
}

.cloud-1 { top: 3px; left: 15%; }
.cloud-2 { top: 28px; left: 60%; animation-delay: 7s; }

@keyframes cloudMove {
    0% { transform: translateX(-100px); }
    100% { transform: translateX(110vw); }
}

.flow-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 9px;
    margin: 18px 0 28px;
    flex-wrap: wrap;
}

.flow-item {
    padding: 10px 15px;
    border-radius: 12px;
    background: #111827;
    border: 1px solid #263247;
    color: #cbd5e1;
    font-size: 13px;
    font-weight: 650;
    transition: transform .25s ease, border-color .25s ease;
}

.flow-item:hover {
    transform: translateY(-3px);
    border-color: rgba(96,165,250,.4);
}

.flow-arrow {
    color: #64748b;
    font-size: 17px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #0c111b;
    padding: 7px;
    border-radius: 15px;
    border: 1px solid #202b3d;
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    border-radius: 10px;
    color: #94a3b8;
    font-weight: 650;
    padding: 0 18px;
}

.stTabs [aria-selected="true"] {
    background: #172554 !important;
    color: #ffffff !important;
}

.section-title {
    font-size: 22px;
    font-weight: 800;
    color: #ffffff;
    margin-top: 27px;
    margin-bottom: 7px;
}

.section-subtitle {
    color: #8994a5;
    font-size: 14px;
    margin-bottom: 20px;
}

.input-card {
    max-width: 1100px;
    margin: 0 auto 30px;
    padding: 28px 32px 30px;
    background: rgba(15,23,42,.72);
    border: 1px solid #243149;
    border-radius: 22px;
    box-shadow: 0 15px 45px rgba(0,0,0,.22);
}

.price-card {
    background:
        linear-gradient(135deg, rgba(30,64,175,.36), rgba(67,56,202,.23));
    border: 1px solid rgba(96,165,250,.35);
    border-radius: 24px;
    padding: 30px;
    text-align: center;
    margin: 12px 0 25px;
    animation: predictionAppear .7s ease-out;
    box-shadow: 0 15px 40px rgba(0,0,0,.18);
}

@keyframes predictionAppear {
    0% {
        opacity: 0;
        transform: translateY(20px) scale(.98);
    }
    100% {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

.price-label {
    color: #aab4c3;
    font-size: 14px;
    margin-bottom: 8px;
}

.price-value {
    color: #ffffff;
    font-size: 50px;
    font-weight: 850;
}

.price-note {
    color: #93c5fd;
    font-size: 13px;
    margin-top: 7px;
}

.info-card {
    background: rgba(15,23,42,.78);
    border: 1px solid #202b3d;
    border-radius: 18px;
    padding: 20px;
    height: 100%;
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}

.info-card:hover {
    transform: translateY(-4px);
    border-color: rgba(96,165,250,.35);
    box-shadow: 0 10px 30px rgba(0,0,0,.25);
}

.info-label {
    color: #8994a5;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .7px;
    margin-bottom: 7px;
}

.info-value {
    color: #ffffff;
    font-size: 18px;
    font-weight: 750;
}

.route-card {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 25px;
    padding: 25px;
    margin: 15px 0 25px;
    border-radius: 20px;
    background: rgba(15,23,42,.65);
    border: 1px solid #202c40;
}

.airport {
    text-align: center;
    min-width: 90px;
}

.airport-code {
    font-size: 28px;
    font-weight: 850;
    color: #ffffff;
}

.airport-name {
    font-size: 12px;
    color: #8994a5;
    margin-top: 4px;
}

.route-line {
    position: relative;
    width: 55%;
    height: 2px;
    background:
        repeating-linear-gradient(
            to right,
            #334155 0px,
            #334155 7px,
            transparent 7px,
            transparent 14px
        );
}

.route-plane {
    position: absolute;
    top: -17px;
    left: 0;
    font-size: 27px;
    animation: routePlane 4s ease-in-out infinite;
    filter: drop-shadow(0 0 7px rgba(96,165,250,.4));
}

@keyframes routePlane {
    0% { left: 0%; transform: rotate(0deg); }
    50% { left: 48%; transform: rotate(2deg); }
    100% { left: 96%; transform: rotate(0deg); }
}

.insight-card {
    background: #0d1420;
    border: 1px solid #1d2939;
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 12px;
}

.footer {
    text-align: center;
    color: #667085;
    font-size: 12px;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #1d2635;
}

@media (max-width: 900px) {
    .hero h1 { font-size: 34px; }
    .route-card { gap: 12px; }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL PATHS
# ============================================================

PREPROCESSOR_PATH = Path("artifacts/preprocessor.joblib")
MODEL_PATH = Path("artifacts/flight_price_xgb.json")


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not PREPROCESSOR_PATH.exists():
    st.error(
        "❌ Missing artifacts/preprocessor.joblib. "
        "Place your trained preprocessing artifact in the artifacts folder."
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error(
        "❌ Missing artifacts/flight_price_xgb.json. "
        "Place your trained XGBoost JSON model in the artifacts folder."
    )
    st.stop()


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


@st.cache_resource
def load_model():
    loaded_model = xgboost.XGBRegressor()
    loaded_model.load_model(MODEL_PATH)
    return loaded_model


preprocessor = load_preprocessor()
model = load_model()


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource
def load_shap_explainer():
    return shap.TreeExplainer(model)


explainer = load_shap_explainer()


# ============================================================
# DATA OPTIONS
# ============================================================

AIRLINES = [
    "Air_India",
    "GO_FIRST",
    "Indigo",
    "SpiceJet",
    "Vistara",
    "AirAsia",
]

CITIES = [
    "Delhi",
    "Mumbai",
    "Bangalore",
    "Kolkata",
    "Hyderabad",
    "Chennai",
]

TIMES = [
    "Early_Morning",
    "Morning",
    "Afternoon",
    "Evening",
    "Night",
    "Late_Night",
]

STOPS = [
    "zero",
    "one",
    "two_or_more",
]

CLASSES = [
    "Economy",
    "Business",
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def get_booking_window(days_left):
    if days_left <= 3:
        return "Last_Minute"
    if days_left <= 7:
        return "Very_Near"
    if days_left <= 15:
        return "Near"
    if days_left <= 30:
        return "Medium"
    if days_left <= 60:
        return "Early"
    return "Very_Early"


def get_duration_category(duration):
    if duration <= 2:
        return "Short"
    if duration <= 5:
        return "Medium"
    if duration <= 8:
        return "Long"
    if duration <= 12:
        return "Very_Long"
    return "Ultra_Long"


def engineer_features(
    airline,
    source_city,
    departure_time,
    stops,
    arrival_time,
    destination_city,
    flight_class,
    duration,
    days_left,
):
    route = f"{source_city}_to_{destination_city}"
    booking_window = get_booking_window(days_left)
    departure_arrival = f"{departure_time}_to_{arrival_time}"
    duration_category = get_duration_category(duration)
    class_stops = f"{flight_class}_{stops}"

    return pd.DataFrame(
        {
            "airline": [airline],
            "source_city": [source_city],
            "departure_time": [departure_time],
            "stops": [stops],
            "arrival_time": [arrival_time],
            "destination_city": [destination_city],
            "class": [flight_class],
            "route": [route],
            "duration_category": [duration_category],
            "booking_window": [booking_window],
            "departure_arrival": [departure_arrival],
            "class_stops": [class_stops],
            "duration": [float(duration)],
            "days_left": [int(days_left)],
        }
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_price(input_data):
    processed = preprocessor.transform(input_data)
    prediction = model.predict(processed)[0]
    return max(0.0, float(prediction))


# ============================================================
# SHAP
# ============================================================

def get_shap_explanation(input_data):
    processed = preprocessor.transform(input_data)

    shap_values = explainer.shap_values(processed)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 2:
        values = shap_values[0]
    else:
        values = shap_values

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [
            f"Feature_{i}" for i in range(len(values))
        ]

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "impact": values,
            "absolute_impact": np.abs(values),
        }
    )

    return (
        result
        .sort_values("absolute_impact", ascending=False)
        .reset_index(drop=True)
    )


def clean_feature_name(name):
    name = str(name)
    if "__" in name:
        name = name.split("__", 1)[1]
    return name


# ============================================================
# GENERIC SCENARIO PREDICTION
# ============================================================

def scenario_price(
    airline,
    source_city,
    departure_time,
    stops,
    arrival_time,
    destination_city,
    flight_class,
    duration,
    days_left,
):
    data = engineer_features(
        airline=airline,
        source_city=source_city,
        departure_time=departure_time,
        stops=stops,
        arrival_time=arrival_time,
        destination_city=destination_city,
        flight_class=flight_class,
        duration=duration,
        days_left=days_left,
    )
    return predict_price(data)


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(input_data, prediction, shap_df):
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=21,
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "ReportNormal",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
    )

    story = []

    story.append(
        Paragraph(
            "Flight Price Prediction Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Machine Learning Flight Price Prediction Dashboard",
            normal_style,
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            f"<b>Estimated Flight Price:</b> "
            f"INR {prediction:,.0f}",
            heading_style,
        )
    )

    row = input_data.iloc[0]

    table_data = [
        ["Parameter", "Value"],
        ["Airline", str(row["airline"])],
        ["Source City", str(row["source_city"])],
        ["Destination City", str(row["destination_city"])],
        ["Departure Time", str(row["departure_time"])],
        ["Arrival Time", str(row["arrival_time"])],
        ["Stops", str(row["stops"])],
        ["Class", str(row["class"])],
        ["Duration", f"{float(row['duration']):.1f} hours"],
        ["Days Before Departure", str(row["days_left"])],
        ["Booking Window", str(row["booking_window"])],
        ["Duration Category", str(row["duration_category"])],
        ["Route", str(row["route"])],
    ]

    table = Table(
        table_data,
        colWidths=[65 * mm, 100 * mm],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#172554"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(table)
    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Top Model Factors",
            heading_style,
        )
    )

    top_shap = shap_df.head(7).copy()

    shap_table = [
        ["Feature", "SHAP Impact", "Effect"]
    ]

    for _, item in top_shap.iterrows():
        impact = float(item["impact"])
        effect = (
            "Increases prediction"
            if impact > 0
            else "Decreases prediction"
        )

        shap_table.append(
            [
                clean_feature_name(item["feature"]),
                f"{impact:+,.2f}",
                effect,
            ]
        )

    shap_table_obj = Table(
        shap_table,
        colWidths=[75 * mm, 40 * mm, 55 * mm],
    )

    shap_table_obj.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#172554"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(shap_table_obj)
    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Model: XGBoost Regression",
            normal_style,
        )
    )

    story.append(
        Spacer(1, 5)
    )

    story.append(
        Paragraph(
            "This report contains a model-generated estimate. "
            "It should not be interpreted as a guaranteed future market price.",
            normal_style,
        )
    )

    document.build(story)

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "input_data" not in st.session_state:
    st.session_state.input_data = None

if "shap_df" not in st.session_state:
    st.session_state.shap_df = None


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-badge">✈️ RayAI </div>
    <h1>Smart Flight Pricing</h1>
    <p> AI-powered flight fare prediction to help travelers make smarter booking decisions. </p>
    <div class="flight-animation">
        <div class="cloud cloud-1">☁️</div>
        <div class="cloud cloud-3">☁️</div>
        <div class="cloud cloud-2">☁️</div>
        <div class="flight-track">✈️</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PIPELINE
# ============================================================

st.markdown(
    """
<div class="flow-container">
    <div class="flow-item">✈️ Flight Details</div>
    <div class="flow-arrow">→</div>
    <div class="flow-item">⚙️ Feature Engineering</div>
    <div class="flow-arrow">→</div>
    <div class="flow-item">🔄 Preprocessing</div>
    <div class="flow-arrow">→</div>
    <div class="flow-item">🌲 XGBoost</div>
    <div class="flow-arrow">→</div>
    <div class="flow-item">🧠 Prediction</div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tab_prediction, tab_whatif, tab_compare, tab_report, tab_model = st.tabs(
    [
        "✈️ Prediction",
        "🔬 What-If Simulator",
        "⚖️ Flight Comparison",
        "📄 Prediction Report",
        "🤖 Model Info",
    ]
)


# ============================================================
# TAB 1: PREDICTION
# ============================================================

with tab_prediction:

    st.markdown(
        '<div class="section-title">✈️ Enter Flight Details</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Enter your journey information to estimate the flight price."
        "</div>",
        unsafe_allow_html=True,
    )

    # Row 1
    c1, c2, c3 = st.columns(3)

    with c1:
        airline = st.selectbox(
            "Airline",
            AIRLINES,
            key="prediction_airline",
        )

    with c2:
        source_city = st.selectbox(
            "Source City",
            CITIES,
            key="prediction_source",
        )

    with c3:
        destination_city = st.selectbox(
            "Destination City",
            CITIES,
            key="prediction_destination",
        )

    # Row 2
    c1, c2, c3 = st.columns(3)

    with c1:
        departure_time = st.selectbox(
            "Departure Time",
            TIMES,
            key="prediction_departure",
        )

    with c2:
        arrival_time = st.selectbox(
            "Arrival Time",
            TIMES,
            key="prediction_arrival",
        )

    with c3:
        stops = st.selectbox(
            "Stops",
            STOPS,
            key="prediction_stops",
        )

    # Row 3
    c1, c2, c3 = st.columns(3)

    with c1:
        flight_class = st.selectbox(
            "Class",
            CLASSES,
            key="prediction_class",
        )

    with c2:
        duration = st.number_input(
            "Duration (hours)",
            min_value=0.5,
            max_value=50.0,
            value=3.0,
            step=0.5,
            key="prediction_duration",
        )

    with c3:
        days_left = st.number_input(
            "Days Before Departure",
            min_value=1,
            max_value=60,
            value=20,
            step=1,
            key="prediction_days",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    b1, b2 = st.columns(2)

    with b1:
        reset_button = st.button(
            "↻ Reset",
            use_container_width=True,
            key="prediction_reset",
        )

    with b2:
        predict_button = st.button(
            "🚀 Predict Flight Price",
            type="primary",
            use_container_width=True,
            key="prediction_button",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if reset_button:
        st.session_state.prediction = None
        st.session_state.input_data = None
        st.session_state.shap_df = None
        st.rerun()

    if predict_button:

        input_data = engineer_features(
            airline=airline,
            source_city=source_city,
            departure_time=departure_time,
            stops=stops,
            arrival_time=arrival_time,
            destination_city=destination_city,
            flight_class=flight_class,
            duration=duration,
            days_left=days_left,
        )

        try:
            prediction = predict_price(input_data)
            shap_df = get_shap_explanation(input_data)

            st.session_state.prediction = prediction
            st.session_state.input_data = input_data
            st.session_state.shap_df = shap_df

            st.success("Prediction generated successfully.")

        except Exception as exc:
            st.error("Prediction failed.")
            st.exception(exc)

    # ========================================================
    # RESULT
    # ========================================================

    if st.session_state.prediction is not None:

        prediction = st.session_state.prediction
        input_data = st.session_state.input_data
        shap_df = st.session_state.shap_df

        st.markdown(
            '<div class="section-title">💰 Prediction Result</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="price-card">
                <div class="price-label">Estimated Flight Price</div>
                <div class="price-value">₹{prediction:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ====================================================
        # TRIP SUMMARY
        # ====================================================

        st.markdown(
            '<div class="section-title">✈️ Trip Summary</div>',
            unsafe_allow_html=True,
        )

        summary_columns = st.columns(4)

        summary_values = [
            ("ROUTE", f"{source_city} → {destination_city}"),
            ("AIRLINE", airline),
            ("CLASS", flight_class),
            ("STOPS", stops),
        ]

        for col, (label, value) in zip(
            summary_columns,
            summary_values,
        ):
            with col:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="info-label">{label}</div>
                        <div class="info-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ====================================================
        # ROUTE
        # ====================================================

        st.markdown(
            '<div class="section-title">🗺️ Flight Route</div>',
            unsafe_allow_html=True,
        )

        source_code = source_city[:3].upper()
        destination_code = destination_city[:3].upper()

        st.markdown(
            f"""
            <div class="route-card">
                <div class="airport">
                    <div class="airport-code"> {source_code}</div>
                    <div class="airport-name">{source_city}</div>
                </div>
                <div class="route-line"><div class="route-plane">✈️</div>
                </div>
                <div class="airport">
                    <div class="airport-code">{destination_code}</div>
                    <div class="airport-name">{destination_city}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ====================================================
        # METRICS
        # ====================================================

        st.markdown(
            '<div class="section-title">📊 Flight Metrics</div>',
            unsafe_allow_html=True,
        )

        booking_window = get_booking_window(days_left)
        duration_category = get_duration_category(duration)

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                "Duration",
                f"{duration:.1f} hrs",
            )

        with m2:
            st.metric(
                "Days Left",
                str(days_left),
            )

        with m3:
            st.metric(
                "Booking Window",
                booking_window.replace("_", " "),
            )

        with m4:
            st.metric(
                "Duration Type",
                duration_category.replace("_", " "),
            )

        # ====================================================
        # SHAP
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            "🧠 Why did the model predict this price?"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">'
            "SHAP shows how individual processed features contributed "
            "to this specific prediction."
            "</div>",
            unsafe_allow_html=True,
        )

        top_shap = shap_df.head(10).copy()
        top_shap["feature"] = top_shap["feature"].apply(
            clean_feature_name
        )

        shap_fig = go.Figure()

        shap_fig.add_trace(
            go.Bar(
                x=top_shap["impact"],
                y=top_shap["feature"],
                orientation="h",
                text=[
                    f"{x:+,.0f}"
                    for x in top_shap["impact"]
                ],
                textposition="outside",
                hovertemplate=(
                    "%{y}<br>"
                    "SHAP Impact: %{x:+,.2f}"
                    "<extra></extra>"
                ),
            )
        )

        shap_fig.update_layout(
            height=440,
            margin=dict(l=20, r=70, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            xaxis=dict(
                title="SHAP Impact",
                gridcolor="#202938",
                zeroline=True,
            ),
            yaxis=dict(
                autorange="reversed",
            ),
        )

        st.plotly_chart(
            shap_fig,
            use_container_width=True,
            key="prediction_shap_chart",
        )

        st.caption(
            "Positive SHAP values push this prediction higher; "
            "negative SHAP values push it lower."
        )

        # ====================================================
        # PRICE VS DAYS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            "📈 Price vs Days Before Departure"
            "</div>",
            unsafe_allow_html=True,
        )

        days_range = [1, 3, 7, 15, 30, 45]

        day_prices = []

        for day in days_range:
            day_prices.append(
                scenario_price(
                    airline,
                    source_city,
                    departure_time,
                    stops,
                    arrival_time,
                    destination_city,
                    flight_class,
                    duration,
                    day,
                )
            )

        fig_days = go.Figure()

        fig_days.add_trace(
            go.Scatter(
                x=days_range,
                y=day_prices,
                mode="lines+markers",
                line=dict(width=3),
                marker=dict(size=9),
                hovertemplate=(
                    "Days Left: %{x}<br>"
                    "Predicted Price: ₹%{y:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

        fig_days.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            xaxis=dict(
                title="Days Before Departure",
                gridcolor="#202938",
            ),
            yaxis=dict(
                title="Predicted Price (₹)",
                gridcolor="#202938",
            ),
        )

        st.plotly_chart(
            fig_days,
            use_container_width=True,
            key="price_days_chart",
        )

        # ====================================================
        # CLASS + STOPS
        # ====================================================

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            class_prices = []

            for selected_class in CLASSES:
                class_prices.append(
                    scenario_price(
                        airline,
                        source_city,
                        departure_time,
                        stops,
                        arrival_time,
                        destination_city,
                        selected_class,
                        duration,
                        days_left,
                    )
                )

            fig_class = go.Figure()

            fig_class.add_trace(
                go.Bar(
                    x=CLASSES,
                    y=class_prices,
                    text=[
                        f"₹{x:,.0f}"
                        for x in class_prices
                    ],
                    textposition="outside",
                    hovertemplate=(
                        "%{x}<br>"
                        "Predicted Price: ₹%{y:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )

            fig_class.update_layout(
                title="💺 Price by Class",
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                yaxis=dict(
                    title="Predicted Price (₹)",
                    gridcolor="#202938",
                ),
            )

            st.plotly_chart(
                fig_class,
                use_container_width=True,
                key="class_chart",
            )

        with chart_col2:

            stop_labels = [
                "Non-stop",
                "1 Stop",
                "2+ Stops",
            ]

            stop_prices = []

            for selected_stops in STOPS:
                stop_prices.append(
                    scenario_price(
                        airline,
                        source_city,
                        departure_time,
                        selected_stops,
                        arrival_time,
                        destination_city,
                        flight_class,
                        duration,
                        days_left,
                    )
                )

            fig_stops = go.Figure()

            fig_stops.add_trace(
                go.Bar(
                    x=stop_labels,
                    y=stop_prices,
                    text=[
                        f"₹{x:,.0f}"
                        for x in stop_prices
                    ],
                    textposition="outside",
                    hovertemplate=(
                        "%{x}<br>"
                        "Predicted Price: ₹%{y:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )

            fig_stops.update_layout(
                title="🛫 Price by Stops",
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                yaxis=dict(
                    title="Predicted Price (₹)",
                    gridcolor="#202938",
                ),
            )

            st.plotly_chart(
                fig_stops,
                use_container_width=True,
                key="stops_chart",
            )

        # ====================================================
        # ENGINEERED FEATURES
        # ====================================================

        with st.expander("⚙️ View Engineered Features"):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# TAB 2: WHAT-IF SIMULATOR
# ============================================================

with tab_whatif:

    st.markdown(
        '<div class="section-title">🔬 What-If Flight Simulator</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Change one flight parameter and let the trained model "
        "generate alternative scenarios."
        "</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.input_data is not None:

        base = st.session_state.input_data.iloc[0]

        base_airline = base["airline"]
        base_source = base["source_city"]
        base_departure = base["departure_time"]
        base_stops = base["stops"]
        base_arrival = base["arrival_time"]
        base_destination = base["destination_city"]
        base_class = base["class"]
        base_duration = float(base["duration"])
        base_days = int(base["days_left"])

    else:

        base_airline = "Indigo"
        base_source = "Delhi"
        base_departure = "Evening"
        base_stops = "one"
        base_arrival = "Morning"
        base_destination = "Mumbai"
        base_class = "Economy"
        base_duration = 3.0
        base_days = 20

    base_price = scenario_price(
        base_airline,
        base_source,
        base_departure,
        base_stops,
        base_arrival,
        base_destination,
        base_class,
        base_duration,
        base_days,
    )

    st.markdown("### Current Flight")

    b1, b2, b3, b4 = st.columns(4)

    with b1:
        st.metric(
            "Route",
            f"{base_source} → {base_destination}",
        )

    with b2:
        st.metric("Airline", base_airline)

    with b3:
        st.metric("Class", base_class)

    with b4:
        st.metric(
            "Current Price",
            f"₹{base_price:,.0f}",
        )

    st.divider()

    parameter = st.selectbox(
        "What do you want to change?",
        [
            "Days Before Departure",
            "Travel Class",
            "Number of Stops",
            "Flight Duration",
            "Airline",
        ],
        key="whatif_parameter",
    )

    scenario_values = []
    scenario_prices = []

    if parameter == "Days Before Departure":

        scenario_values = [1, 3, 7, 15, 20, 30, 45]

        for value in scenario_values:
            scenario_prices.append(
                scenario_price(
                    base_airline,
                    base_source,
                    base_departure,
                    base_stops,
                    base_arrival,
                    base_destination,
                    base_class,
                    base_duration,
                    value,
                )
            )

        x_title = "Days Before Departure"

    elif parameter == "Travel Class":

        scenario_values = CLASSES

        for value in scenario_values:
            scenario_prices.append(
                scenario_price(
                    base_airline,
                    base_source,
                    base_departure,
                    base_stops,
                    base_arrival,
                    base_destination,
                    value,
                    base_duration,
                    base_days,
                )
            )

        x_title = "Class"

    elif parameter == "Number of Stops":

        raw_values = STOPS

        scenario_values = [
            "Non-stop",
            "1 Stop",
            "2+ Stops",
        ]

        for value in raw_values:
            scenario_prices.append(
                scenario_price(
                    base_airline,
                    base_source,
                    base_departure,
                    value,
                    base_arrival,
                    base_destination,
                    base_class,
                    base_duration,
                    base_days,
                )
            )

        x_title = "Stops"

    elif parameter == "Flight Duration":

        scenario_values = [1, 2, 3, 5, 8, 12]

        for value in scenario_values:
            scenario_prices.append(
                scenario_price(
                    base_airline,
                    base_source,
                    base_departure,
                    base_stops,
                    base_arrival,
                    base_destination,
                    base_class,
                    value,
                    base_days,
                )
            )

        x_title = "Duration (Hours)"

    else:

        scenario_values = AIRLINES

        for value in scenario_values:
            scenario_prices.append(
                scenario_price(
                    value,
                    base_source,
                    base_departure,
                    base_stops,
                    base_arrival,
                    base_destination,
                    base_class,
                    base_duration,
                    base_days,
                )
            )

        x_title = "Airline"

    fig_whatif = go.Figure()

    fig_whatif.add_trace(
        go.Scatter(
            x=scenario_values,
            y=scenario_prices,
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=10),
            hovertemplate=(
                "%{x}<br>"
                "Predicted Price: ₹%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig_whatif.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        xaxis=dict(
            title=x_title,
            gridcolor="#202938",
        ),
        yaxis=dict(
            title="Predicted Price (₹)",
            gridcolor="#202938",
        ),
    )

    st.plotly_chart(
        fig_whatif,
        use_container_width=True,
        key="whatif_chart",
    )

    scenario_df = pd.DataFrame(
        {
            "Scenario": scenario_values,
            "Predicted Price": [
                f"₹{x:,.0f}"
                for x in scenario_prices
            ],
        }
    )

    st.dataframe(
        scenario_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 3: FLIGHT COMPARISON
# ============================================================

with tab_compare:

    st.markdown(
        '<div class="section-title">⚖️ Flight Comparison</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Create three flight scenarios and compare their "
        "model-predicted prices."
        "</div>",
        unsafe_allow_html=True,
    )

    rc1, rc2, rc3 = st.columns(3)

    with rc1:
        compare_source = st.selectbox(
            "Source City",
            CITIES,
            index=CITIES.index("Delhi"),
            key="compare_source",
        )

    with rc2:
        compare_destination = st.selectbox(
            "Destination City",
            CITIES,
            index=CITIES.index("Mumbai"),
            key="compare_destination",
        )

    with rc3:
        compare_departure = st.selectbox(
            "Departure Time",
            TIMES,
            index=TIMES.index("Evening"),
            key="compare_departure",
        )

    scenario_columns = st.columns(3)
    comparison_results = []

    for index, column in enumerate(scenario_columns):

        with column:

            scenario_number = index + 1

            st.markdown(
                f"### ✈️ Flight {scenario_number}"
            )

            selected_airline = st.selectbox(
                "Airline",
                AIRLINES,
                index=2,
                key=f"compare_airline_{index}",
            )

            selected_class = st.selectbox(
                "Class",
                CLASSES,
                key=f"compare_class_{index}",
            )

            selected_stops = st.selectbox(
                "Stops",
                STOPS,
                index=1,
                key=f"compare_stops_{index}",
            )

            selected_arrival = st.selectbox(
                "Arrival Time",
                TIMES,
                index=1,
                key=f"compare_arrival_{index}",
            )

            selected_duration = st.number_input(
                "Duration (hours)",
                min_value=0.5,
                max_value=50.0,
                value=3.0,
                step=0.5,
                key=f"compare_duration_{index}",
            )

            selected_days = st.number_input(
                "Days Before Departure",
                min_value=1,
                max_value=60,
                value=20,
                step=1,
                key=f"compare_days_{index}",
            )

            price = scenario_price(
                selected_airline,
                compare_source,
                compare_departure,
                selected_stops,
                selected_arrival,
                compare_destination,
                selected_class,
                selected_duration,
                selected_days,
            )

            comparison_results.append(
                {
                    "Flight": f"Flight {scenario_number}",
                    "Airline": selected_airline,
                    "Class": selected_class,
                    "Stops": selected_stops,
                    "Duration": selected_duration,
                    "Days Left": selected_days,
                    "Predicted Price": price,
                }
            )

    comparison_df = pd.DataFrame(comparison_results)

    st.markdown(
        '<div class="section-title">📊 Price Comparison</div>',
        unsafe_allow_html=True,
    )

    fig_compare = go.Figure()

    fig_compare.add_trace(
        go.Bar(
            x=comparison_df["Flight"],
            y=comparison_df["Predicted Price"],
            text=[
                f"₹{x:,.0f}"
                for x in comparison_df["Predicted Price"]
            ],
            textposition="outside",
            hovertemplate=(
                "%{x}<br>"
                "Predicted Price: ₹%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig_compare.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        yaxis=dict(
            title="Predicted Price (₹)",
            gridcolor="#202938",
        ),
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True,
        key="comparison_chart",
    )

    display_comparison = comparison_df.copy()

    display_comparison["Predicted Price"] = (
        display_comparison["Predicted Price"]
        .map(lambda value: f"₹{value:,.0f}")
    )

    st.dataframe(
        display_comparison,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 4: PREDICTION REPORT
# ============================================================

with tab_report:

    st.markdown(
        '<div class="section-title">📄 Prediction Report</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Generate a downloadable PDF containing the flight "
        "details, prediction and SHAP explanation."
        "</div>",
        unsafe_allow_html=True,
    )

    if (
        st.session_state.prediction is None
        or st.session_state.input_data is None
        or st.session_state.shap_df is None
    ):

        st.info(
            "Make a prediction in the Prediction tab first."
        )

    else:

        prediction = st.session_state.prediction
        input_data = st.session_state.input_data
        shap_df = st.session_state.shap_df

        st.markdown(
            f"""
            <div class="price-card">
                <div class="price-label">Report Prediction</div>
                <div class="price-value">₹{prediction:,.0f}</div>
                <div class="price-note">XGBoost Regression</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            report_pdf = create_pdf_report(
                input_data=input_data,
                prediction=prediction,
                shap_df=shap_df,
            )

            st.download_button(
                label="📥 Download Prediction Report",
                data=report_pdf,
                file_name="flight_price_prediction_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_pdf",
            )

        except Exception as exc:

            st.error(
                "Could not generate the PDF report."
            )
            st.exception(exc)

        st.markdown("### Report Contains")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(
                """
                <div class="info-card">
                    <div class="info-label">FLIGHT</div>
                    <div class="info-value">
                        Route & Flight Details
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r2:
            st.markdown(
                """
                <div class="info-card">
                    <div class="info-label">PREDICTION</div>
                    <div class="info-value">
                        Estimated Price
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r3:
            st.markdown(
                """
                <div class="info-card">
                    <div class="info-label">EXPLAINABILITY</div>
                    <div class="info-value">
                        Top SHAP Factors
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# TAB 5: MODEL INFO
# ============================================================

with tab_model:

    st.markdown(
        '<div class="section-title">🤖 Model Information</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Technical overview of the trained machine learning pipeline."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Algorithm", "XGBoost")

    with c2:
        st.metric("Estimators", "500")

    with c3:
        st.metric("Max Depth", "10")

    with c4:
        st.metric("R²", "0.9854")
    st.markdown("### 🔄 ML Pipeline")
    st.code(
        """
Raw Flight Information
        ↓
Feature Engineering
        ↓
Route
Booking Window
Duration Category
Departure → Arrival
Class + Stops
        ↓
ColumnTransformer
        ↓
OneHotEncoder
        ↓
XGBoost Regressor
        ↓
Predicted Flight Price
        ↓
SHAP Explainability
""",
        language="text",
    )

    st.markdown("### 📊 Test-Set Performance")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "MAE",
            "₹1,469.82",
        )

    with c2:
        st.metric(
            "RMSE",
            "₹2,747.83",
        )

    with c3:
        st.metric(
            "R²",
            "0.9854",
        )

    st.markdown("### 🧠 SHAP Explainability")

    st.markdown(
        """
SHAP explains the contribution of processed model features
for an individual prediction.

- **Positive SHAP value** → pushes the prediction higher.
- **Negative SHAP value** → pushes the prediction lower.
- **Larger absolute SHAP value** → stronger contribution
  to that prediction.

The chart therefore explains **why the model produced a
particular prediction**, rather than claiming that a feature
is causally responsible for the real-world ticket price.
"""
    )

    st.markdown("### 📦 Model Artifacts")

    st.code(
        """
artifacts/
├── preprocessor.joblib
└── flight_price_xgb.json
""",
        language="text",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
    Flight Price Prediction & Explainability Dashboard
    <br><br>
</div>
""",
    unsafe_allow_html=True,
)
