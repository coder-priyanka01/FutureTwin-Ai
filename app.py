import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import altair as alt
from datetime import date

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FutureTwin AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 8% 8%,
            rgba(124, 92, 255, 0.22),
            transparent 30%
        ),
        radial-gradient(
            circle at 92% 12%,
            rgba(31, 211, 198, 0.16),
            transparent 28%
        ),
        radial-gradient(
            circle at 55% 100%,
            rgba(225, 82, 170, 0.10),
            transparent 32%
        ),
        #0b1020;
    color: #f6f7ff;
}

/* Main container */

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hero */

.hero {
    padding: 55px 45px;
    border-radius: 28px;
    margin-bottom: 30px;

    background:
        linear-gradient(
            135deg,
            rgba(108, 78, 220, 0.22),
            rgba(23, 190, 181, 0.13),
            rgba(214, 76, 165, 0.09)
        );

    border: 1px solid rgba(180, 170, 255, 0.20);

    box-shadow:
        0 25px 70px rgba(0,0,0,0.30),
        inset 0 1px 0 rgba(255,255,255,0.04);
}

.hero-title {
    font-family: 'Poppins', sans-serif;
    font-size: 58px;
    font-weight: 800;
    margin: 0;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #b9a7ff,
            #61e6d2,
            #ff9bd8
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #aeb8d2;
    font-size: 18px;
    line-height: 1.7;
    max-width: 850px;
    margin-top: 12px;
}

/* Section */

.section-title {
    font-family: 'Poppins', sans-serif;
    font-size: 30px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 15px;
}

.section-subtitle {
    color: #8994b0;
    font-size: 15px;
    margin-bottom: 22px;
}

/* Cards */

.card {
    background:
        linear-gradient(
            145deg,
            rgba(24, 29, 53, 0.94),
            rgba(15, 23, 43, 0.94)
        );

    border: 1px solid rgba(145, 157, 210, 0.13);
    border-radius: 20px;
    padding: 25px;

    box-shadow:
        0 15px 45px rgba(0,0,0,0.20);
}

.metric-card {
    background:
        linear-gradient(
            145deg,
            rgba(29, 34, 62, 0.98),
            rgba(16, 25, 48, 0.98)
        );

    border: 1px solid rgba(145,157,210,0.14);
    border-radius: 18px;
    padding: 24px;
}

.metric-label {
    color: #8f9ab5;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    color: #ffffff;
    font-size: 34px;
    font-weight: 700;
    margin: 8px 0;
}

/* Prediction */

.prediction-box {
    background:
        linear-gradient(
            135deg,
            rgba(111,82,217,0.18),
            rgba(34,199,195,0.10)
        );

    border: 1px solid rgba(154,137,255,0.30);
    border-radius: 20px;
    padding: 26px;
}

.prediction-title {
    font-family: 'Poppins', sans-serif;
    font-size: 21px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 12px;
}

.prediction-text {
    color: #aeb8d2;
    line-height: 1.9;
}

/* Buttons */

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(170,160,255,0.25);

    background:
        linear-gradient(
            90deg,
            #6f52d9,
            #19b9aa,
            #d35cae
        );

    color: white;
    font-weight: 700;
    min-height: 48px;

    transition: 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow:
        0 10px 30px rgba(111,82,217,0.30);
}

/* Inputs */

div[data-baseweb="input"] {
    background-color: #121a30;
}

div[data-baseweb="select"] > div {
    background-color: #121a30;
}

input {
    color: white !important;
}

/* Tabs */

button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    color: #8994b0;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff;
    background: rgba(111,82,217,0.16);
    border-radius: 10px;
}

/* Dataframe */

div[data-testid="stDataFrame"] {
    border: 1px solid rgba(154,137,255,0.14);
    border-radius: 14px;
    overflow: hidden;
}

/* Divider */

.soft-divider {
    height: 1px;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(145,157,210,0.20),
        transparent
    );
    margin: 25px 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    revenue_model = joblib.load(
        "futuretwin_revenue_model.pkl"
    )

    demand_model = joblib.load(
        "futuretwin_demand_model.pkl"
    )

    nlp_model = joblib.load(
        "futuretwin_nlp_model.pkl"
    )

    tfidf = joblib.load(
        "futuretwin_tfidf.pkl"
    )

    return (
        revenue_model,
        demand_model,
        nlp_model,
        tfidf
    )


try:

    (
        revenue_model,
        demand_model,
        nlp_model,
        tfidf
    ) = load_models()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(
        "Model files could not be loaded. "
        "Please make sure all .pkl files are in the same folder as app.py."
    )

    st.code(str(e))


# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "input_data" not in st.session_state:
    st.session_state.input_data = None

# History is kept in session state and mirrored to a local JSON file.
# This preserves records across ordinary Streamlit reruns on the same filesystem.
HISTORY_FILE = "futuretwin_history.json"


def load_saved_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as history_file:
                saved_data = json.load(history_file)
                if isinstance(saved_data, list):
                    return saved_data
    except (OSError, json.JSONDecodeError):
        pass
    return []


def persist_saved_history():
    try:
        temporary_file = HISTORY_FILE + ".tmp"
        with open(temporary_file, "w", encoding="utf-8") as history_file:
            json.dump(
                st.session_state.history,
                history_file,
                ensure_ascii=False,
                indent=2
            )
        os.replace(temporary_file, HISTORY_FILE)
    except OSError as error:
        st.warning(f"Could not persist scenario history: {error}")


if "history" not in st.session_state:
    st.session_state.history = load_saved_history()

if "auto_saved_scenarios" not in st.session_state:
    st.session_state.auto_saved_scenarios = set()


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🔮 FutureTwin AI
</div>

<div class="hero-subtitle">

Simulate possible business futures using machine learning,
scenario modelling and interactive decision support.

</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "Overview",
    "Company Setup",
    "Digital Twin",
    "Future Simulator",
    "AI Insights",
    "Compare Futures",
    "History"
])


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_features(
    quantity,
    price,
    selected_date
):

    previous_revenue = quantity * price
    previous_quantity = quantity

    rolling7_revenue = previous_revenue
    rolling7_quantity = previous_quantity

    average_price = price
    transactions = 1
    returns = 0

    return pd.DataFrame([{

        "PreviousRevenue": previous_revenue,

        "PreviousQuantity": previous_quantity,

        "Rolling7Revenue": rolling7_revenue,

        "Rolling7Quantity": rolling7_quantity,

        "AveragePrice": average_price,

        "Transactions": transactions,

        "Returns": returns,

        "Month": selected_date.month,

        "DayOfWeek": selected_date.weekday()

    }])


def scenario_adjusted_revenue(
    baseline_revenue,
    demand_change,
    price_change
):

    demand_factor = (
        1 + float(demand_change) / 100
    )

    price_factor = (
        1 + float(price_change) / 100
    )

    simulated_revenue = (
        float(baseline_revenue)
        * demand_factor
        * price_factor
    )

    return max(
        simulated_revenue,
        0.0
    )


def predict_revenue(
    quantity,
    price,
    selected_date
):

    features = create_features(
        quantity,
        price,
        selected_date
    )

    prediction = revenue_model.predict(
        features
    )[0]

    return max(
        float(prediction),
        0
    )


def predict_demand(
    quantity,
    price,
    selected_date
):

    features = create_features(
        quantity,
        price,
        selected_date
    )

    prediction = demand_model.predict(
        features
    )[0]

    probability = None

    if hasattr(
        demand_model,
        "predict_proba"
    ):

        probabilities = (
            demand_model.predict_proba(
                features
            )[0]
        )

        if len(probabilities) > 1:

            probability = (
                float(probabilities[1])
                * 100
            )

    status = (
        "High Demand"
        if int(prediction) == 1
        else "Normal Demand"
    )

    return (
        status,
        probability
    )


def analyze_text(
    description
):

    if not description:

        return (
            "No Description",
            None
        )

    try:

        transformed = tfidf.transform(
            [description]
        )

        prediction = nlp_model.predict(
            transformed
        )[0]

        probability = None

        if hasattr(
            nlp_model,
            "predict_proba"
        ):

            probability = (
                float(
                    nlp_model.predict_proba(
                        transformed
                    )[0][1]
                )
                * 100
            )

        status = str(
            prediction
        )

        return (
            status,
            probability
        )

    except Exception:

        return (
            "Unable to analyse",
            None
        )


def analyze_scenario(
    quantity,
    price,
    selected_date,
    description
):

    revenue = predict_revenue(
        quantity,
        price,
        selected_date
    )

    demand_status, demand_probability = (
        predict_demand(
            quantity,
            price,
            selected_date
        )
    )

    text_status, text_probability = (
        analyze_text(
            description
        )
    )

    return {

        "revenue": revenue,

        "demand_status": demand_status,

        "demand_probability": demand_probability,

        "text_status": text_status,

        "text_probability": text_probability

    }


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">What is FutureTwin AI?</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'An interactive machine-learning based business simulation environment.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="card">

        <h3>📈 Predict</h3>

        <p>
        Estimate future revenue using a trained
        Random Forest regression model.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="card">

        <h3>🔮 Simulate</h3>

        <p>
        Change demand and price assumptions
        to explore possible future scenarios.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="card">

        <h3>⚖️ Compare</h3>

        <p>
        Compare multiple possible futures
        before making a business decision.
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="prediction-box">

    <div class="prediction-title">
    How FutureTwin works
    </div>

    <div class="prediction-text">

    Historical Retail Data
    →
    Feature Engineering
    →
    Machine Learning
    →
    Digital Twin
    →
    What-if Scenario
    →
    Future Comparison

    </div>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.info(
        "FutureTwin AI provides model-based estimates and scenario simulations. "
        "It does not guarantee future business outcomes."
    )


# ============================================================
# TAB 2 — COMPANY SETUP
# ============================================================

with tabs[1]:

    st.markdown(
        '<div class="section-title">Company Setup</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Create the business state that will be used by your Digital Twin.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        company_name = st.text_input(
            "Company Name",
            value="FutureMart"
        )

        product_name = st.text_input(
            "Product Name",
            value="Wireless Headphones"
        )

        product_description = st.text_area(
            "Product Description",
            value="Premium wireless headphones"
        )

    with c2:

        quantity = st.number_input(
            "Expected Quantity",
            min_value=1,
            value=500,
            step=1
        )

        price = st.number_input(
            "Unit Price (₹)",
            min_value=0.01,
            value=1500.0,
            step=50.0
        )

        selected_date = st.date_input(
            "Scenario Date",
            value=date.today()
        )

    st.write("")

    if st.button(
        "Create Digital Twin",
        use_container_width=True
    ):

        result = analyze_scenario(
            quantity,
            price,
            selected_date,
            product_description
        )

        st.session_state.input_data = {

            "company": company_name,

            "product": product_name,

            "description": product_description,

            "quantity": quantity,

            "price": price,

            "date": selected_date

        }

        st.session_state.prediction = result

        st.success(
            "Digital Twin created successfully."
        )


# ============================================================
# TAB 3 — DIGITAL TWIN
# ============================================================

with tabs[2]:

    st.markdown(
        '<div class="section-title">Digital Twin</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Current model-based representation of your business scenario.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.prediction is None:

        st.warning(
            "Create your Digital Twin from Company Setup first."
        )

    else:

        data = st.session_state.input_data
        prediction = st.session_state.prediction

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown("""
            <div class="metric-card">
            <div class="metric-label">COMPANY</div>
            <div class="metric-value"
            style="font-size:22px;">
            """ + str(data["company"]) + """
            </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:

            st.markdown("""
            <div class="metric-card">
            <div class="metric-label">PRODUCT</div>
            <div class="metric-value"
            style="font-size:22px;">
            """ + str(data["product"]) + """
            </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                QUANTITY
                </div>

                <div class="metric-value">
                {data["quantity"]:,.0f}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                UNIT PRICE
                </div>

                <div class="metric-value">
                ₹{data["price"]:,.0f}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                f"""
                <div class="prediction-box">

                <div class="prediction-title">
                📈 Revenue Prediction
                </div>

                <div class="metric-value">
                ₹{prediction["revenue"]:,.0f}
                </div>

                <div class="prediction-text">
                Estimated future revenue based on the
                trained Random Forest model.
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="prediction-box">

                <div class="prediction-title">
                📊 Demand Signal
                </div>

                <div class="metric-value"
                style="font-size:28px;">
                {prediction["demand_status"]}
                </div>

                <div class="prediction-text">
                Model-based demand classification.
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# TAB 4 — FUTURE SIMULATOR
# ============================================================

with tabs[3]:

    st.markdown(
        '<div class="section-title">Future Simulator</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Change business assumptions and explore a possible future.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.prediction is None:

        st.warning(
            "Create your Digital Twin first."
        )

    else:

        input_data = st.session_state.input_data

        base_quantity = float(
            input_data["quantity"]
        )

        base_price = float(
            input_data["price"]
        )

        selected_date = input_data["date"]

        description = input_data["description"]

        c1, c2 = st.columns(2)

        with c1:

            demand_change = st.slider(
                "Demand Change (%)",
                min_value=-50,
                max_value=100,
                value=0,
                step=5
            )

        with c2:

            price_change = st.slider(
                "Price Change (%)",
                min_value=-30,
                max_value=50,
                value=0,
                step=5
            )

        future_quantity = max(
            base_quantity *
            (1 + demand_change / 100),
            1
        )

        future_price = max(
            base_price *
            (1 + price_change / 100),
            0.01
        )

        # IMPORTANT:
        # This is the trained ML baseline.
        base_result = analyze_scenario(
            base_quantity,
            base_price,
            selected_date,
            description
        )

        # Keep the demand classifier working.
        future_demand_status, future_demand_probability = (
            predict_demand(
                future_quantity,
                future_price,
                selected_date
            )
        )

        # Working scenario logic:
        future_revenue = scenario_adjusted_revenue(
            base_result["revenue"],
            demand_change,
            price_change
        )

        # Save each distinct simulated combination once, without changing the UI.
        scenario_signature = (
            str(description),
            str(selected_date),
            round(float(base_quantity), 4),
            round(float(base_price), 4),
            int(demand_change),
            int(price_change)
        )
        if scenario_signature not in st.session_state.auto_saved_scenarios:
            st.session_state.history.append({
                "Scenario": (
                    f"Demand {demand_change:+d}% / "
                    f"Price {price_change:+d}%"
                ),
                "Baseline Revenue": round(float(base_result["revenue"]), 2),
                "Simulated Revenue": round(float(future_revenue), 2),
                "Revenue Change": round(float(future_revenue - base_result["revenue"]), 2),
                "Demand": future_demand_status
            })
            st.session_state.auto_saved_scenarios.add(scenario_signature)
            persist_saved_history()

        revenue_change = (
            future_revenue -
            base_result["revenue"]
        )

        revenue_change_percent = (
            revenue_change /
            base_result["revenue"] * 100
            if base_result["revenue"] != 0
            else 0
        )

        st.write("")

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                CURRENT / BASELINE
                </div>

                <div class="metric-value">
                ₹{base_result["revenue"]:,.0f}
                </div>

                <div class="metric-label">
                Trained ML estimate
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                SIMULATED FUTURE
                </div>

                <div class="metric-value">
                ₹{future_revenue:,.0f}
                </div>

                <div class="metric-label">
                Scenario-adjusted estimate
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        if revenue_change > 0:

            st.success(
                f"Simulated revenue increases by "
                f"₹{revenue_change:,.0f} "
                f"({revenue_change_percent:.1f}%)."
            )

        elif revenue_change < 0:

            st.error(
                f"Simulated revenue decreases by "
                f"₹{abs(revenue_change):,.0f} "
                f"({abs(revenue_change_percent):.1f}%)."
            )

        else:

            st.info(
                "No revenue change because the scenario "
                "assumptions are unchanged."
            )

        st.write("")

        # ----------------------------------------------------
        # COLOURED CURRENT VS FUTURE GRAPH
        # ----------------------------------------------------

        chart_display = pd.DataFrame({

            "State": [
                "Current / Baseline",
                "Simulated Future"
            ],

            "Revenue": [
                base_result["revenue"],
                future_revenue
            ]

        })

        sim_chart = alt.Chart(
            chart_display
        ).mark_bar(
            cornerRadiusTopLeft=8,
            cornerRadiusTopRight=8
        ).encode(

            x=alt.X(
                "State:N",
                title=None,
                sort=[
                    "Current / Baseline",
                    "Simulated Future"
                ]
            ),

            y=alt.Y(
                "Revenue:Q",
                title="Revenue (₹)"
            ),

            color=alt.Color(
                "State:N",

                scale=alt.Scale(

                    domain=[
                        "Current / Baseline",
                        "Simulated Future"
                    ],

                    range=[
                        "#8B5CF6",
                        "#22C7C3"
                    ]

                ),

                legend=None
            ),

            tooltip=[

                alt.Tooltip(
                    "State:N",
                    title="Scenario"
                ),

                alt.Tooltip(
                    "Revenue:Q",
                    title="Revenue",
                    format=",.0f"
                )

            ]

        ).properties(
            height=380
        )

        st.altair_chart(
            sim_chart,
            use_container_width=True
        )

        st.caption(
            "Baseline = trained ML revenue estimate • "
            "Future = scenario-adjusted estimate"
        )

        st.write("")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Demand Change",
                f"{demand_change:+d}%"
            )

        with c2:

            st.metric(
                "Price Change",
                f"{price_change:+d}%"
            )

        with c3:

            st.metric(
                "Future Quantity",
                f"{future_quantity:,.0f}"
            )

        with c4:

            st.metric(
                "Future Price",
                f"₹{future_price:,.2f}"
            )

        st.write("")

        st.markdown(
            f"""
            <div class="prediction-box">

            <div class="prediction-title">
            🔮 Future Scenario Result
            </div>

            <div class="prediction-text">

            Demand change:
            <b>{demand_change:+d}%</b>

            <br>

            Price change:
            <b>{price_change:+d}%</b>

            <br><br>

            Simulated quantity:
            <b>{future_quantity:,.0f}</b>

            <br>

            Simulated price:
            <b>₹{future_price:,.2f}</b>

            <br><br>

            Future demand signal:
            <b>{future_demand_status}</b>

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "💾 Save This Future Scenario",
            use_container_width=True
        ):

            st.session_state.history.append({

                "Scenario":
                    f"Demand {demand_change:+d}% / "
                    f"Price {price_change:+d}%",

                "Baseline Revenue":
                    round(
                        base_result["revenue"],
                        2
                    ),

                "Simulated Revenue":
                    round(
                        future_revenue,
                        2
                    ),

                "Revenue Change":
                    round(
                        revenue_change,
                        2
                    ),

                "Demand":
                    future_demand_status

            })
            persist_saved_history()

            st.success(
                "Future scenario saved."
            )


# ============================================================
# TAB 5 — AI INSIGHTS
# ============================================================

with tabs[4]:

    st.markdown(
        '<div class="section-title">FutureTwin AI Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Understand what each trained model is signalling.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.prediction is None:

        st.warning(
            "Create your Digital Twin first."
        )

    else:

        prediction = st.session_state.prediction

        c1, c2 = st.columns(2)

        with c1:

            st.markdown("""
            <div class="card">

            <h3>📈 Revenue Prediction</h3>

            <p>
            The Random Forest Regressor estimates
            future revenue using engineered
            business features.
            </p>

            </div>
            """, unsafe_allow_html=True)

            st.write("")

            st.metric(
                "Predicted Revenue",
                f"₹{prediction['revenue']:,.0f}"
            )

        with c2:

            st.markdown("""
            <div class="card">

            <h3>📊 Future Demand</h3>

            <p>
            The Random Forest Classifier estimates
            whether the scenario corresponds to
            a high-demand or normal-demand condition.
            </p>

            </div>
            """, unsafe_allow_html=True)

            st.write("")

            if prediction["demand_probability"] is not None:

                st.metric(
                    "High Demand Probability",
                    f"{prediction['demand_probability']:.1f}%"
                )

            else:

                st.metric(
                    "Demand Signal",
                    prediction["demand_status"]
                )

        st.write("")

        st.markdown("""
        <div class="card">

        <h3>📝 Product Text Analysis</h3>

        <p>
        Product descriptions are transformed into
        TF-IDF features and analysed using the
        trained Logistic Regression model.
        </p>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.metric(
            "Product Text Signal",
            prediction["text_status"]
        )

        if prediction["text_probability"] is not None:

            st.write(
                f"Model probability: "
                f"**{prediction['text_probability']:.1f}%**"
            )

        st.info(
            "Model outputs are learned from historical "
            "Online Retail II data and should be interpreted "
            "as predictive signals, not guaranteed outcomes."
        )


# ============================================================
# TAB 6 — COMPARE FUTURES
# ============================================================

with tabs[5]:

    st.markdown(
        '<div class="section-title">Compare Possible Futures</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore multiple what-if scenarios side by side.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.prediction is None:

        st.warning(
            "Create your Digital Twin first."
        )

    else:

        input_data = st.session_state.input_data

        base_quantity = float(
            input_data["quantity"]
        )

        base_price = float(
            input_data["price"]
        )

        selected_date = input_data["date"]

        description = input_data["description"]

        scenarios = {

            "Current": (0, 0),

            "Demand +10%": (10, 0),

            "Demand +20%": (20, 0),

            "Price +10%": (0, 10),

            "Demand +20% + Price +10%": (20, 10)

        }

        # Get one trained ML baseline.
        baseline = predict_revenue(
            base_quantity,
            base_price,
            selected_date
        )

        rows = []

        for name, changes in scenarios.items():

            demand_change = changes[0]

            price_change = changes[1]

            scenario_quantity = (
                base_quantity *
                (1 + demand_change / 100)
            )

            scenario_price = (
                base_price *
                (1 + price_change / 100)
            )

            # Keep the trained demand classifier.
            demand_status, probability = (
                predict_demand(
                    scenario_quantity,
                    scenario_price,
                    selected_date
                )
            )

            # IMPORTANT:
            # Use the working scenario adjustment
            # instead of asking the Random Forest to
            # produce identical tree-leaf predictions.
            simulated_revenue = (
                scenario_adjusted_revenue(
                    baseline,
                    demand_change,
                    price_change
                )
            )

            rows.append({

                "Future Scenario": name,

                "Revenue": round(
                    simulated_revenue,
                    2
                ),

                "Demand Status":
                    demand_status

            })

        comparison = pd.DataFrame(
            rows
        )

        st.dataframe(
            comparison.rename(
                columns={
                    "Revenue":
                    "Simulated Revenue"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        st.write("")

        # ----------------------------------------------------
        # COLOURED COMPARE FUTURES GRAPH
        # ----------------------------------------------------

        colour_domain = [
            "Current",
            "Demand +10%",
            "Demand +20%",
            "Price +10%",
            "Demand +20% + Price +10%"
        ]

        colour_range = [
            "#8B5CF6",
            "#38BDF8",
            "#22C7C3",
            "#F59E0B",
            "#EC6AAE"
        ]

        compare_chart = alt.Chart(
            comparison
        ).mark_bar(
            cornerRadiusTopLeft=8,
            cornerRadiusTopRight=8
        ).encode(

            x=alt.X(
                "Future Scenario:N",
                title=None,
                sort=colour_domain
            ),

            y=alt.Y(
                "Revenue:Q",
                title="Simulated Revenue (₹)"
            ),

            color=alt.Color(

                "Future Scenario:N",

                scale=alt.Scale(
                    domain=colour_domain,
                    range=colour_range
                ),

                legend=None
            ),

            tooltip=[

                alt.Tooltip(
                    "Future Scenario:N",
                    title="Scenario"
                ),

                alt.Tooltip(
                    "Revenue:Q",
                    title="Revenue",
                    format=",.0f"
                ),

                alt.Tooltip(
                    "Demand Status:N",
                    title="Demand"
                )

            ]

        ).properties(
            height=400
        )

        st.altair_chart(
            compare_chart,
            use_container_width=True
        )

        st.caption(
            "Each bar represents a different scenario assumption. "
            "Revenue is calculated from the trained ML baseline "
            "and the transparent scenario-adjustment layer."
        )


# ============================================================
# TAB 7 — HISTORY
# ============================================================

with tabs[6]:

    st.markdown(
        '<div class="section-title">Scenario History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Your simulated business futures from this session.'
        '</div>',
        unsafe_allow_html=True
    )

    if len(
        st.session_state.history
    ) == 0:

        st.info(
            "No scenarios saved yet. "
            "Create your Digital Twin and run a simulation."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.write("")

st.markdown("""
<div class="soft-divider"></div>

<div style="
text-align:center;
color:#78839f;
font-size:13px;
padding:15px;
">

<b style="color:#aeb8d2;">
FutureTwin AI
</b>

<br><br>

AI-powered future scenario simulation

<br><br>

Model-based simulation only.
Results are not guaranteed future outcomes
and should be validated before real-world implementation.

</div>
""", unsafe_allow_html=True)
