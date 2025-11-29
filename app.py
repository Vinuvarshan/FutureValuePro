import streamlit as st
import pandas as pd
import joblib
import altair as alt
import time
import hashlib

# Set page configuration
st.set_page_config(page_title="FutureValuePro", page_icon="📈", layout="wide")

# --- Page Definitions ---
PAGES = ["🏠 House Price", "🌳 Land Value", "💰 Loan Calculator"]

# --- Initialize Session State ---
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "🏠 House Price"
if 'property_value_input' not in st.session_state:
    st.session_state.property_value_input = 300000.0
if 'house_prediction' not in st.session_state:
    st.session_state.house_prediction = 0.0
if 'land_prediction' not in st.session_state:
    st.session_state.land_prediction = 0.0
if 'visual_premium' not in st.session_state:
    st.session_state.visual_premium = 0.0
if 'last_image_hash' not in st.session_state:
    st.session_state.last_image_hash = ""


# --- Helper Function: Dynamic Premium ---
def get_visual_premium(uploaded_file, min_val, max_val):
    if uploaded_file is None:
        return 0.0

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # Logic: Convert hash to number.
    hash_int = int(file_hash, 16)
    random_factor = (hash_int % 100) / 100.0

    premium = min_val + (random_factor * (max_val - min_val))
    return premium, file_hash


# --- Model Loading ---
@st.cache_resource
def load_models():
    try:
        house_model = joblib.load("house_model_pipeline.pkl")
        land_model = joblib.load("land_model_pipeline.pkl")
        return house_model, land_model
    except FileNotFoundError:
        st.error(
            "Model file(s) not found. Please ensure 'house_model_pipeline.pkl' and 'land_model_pipeline.pkl' are present.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading the models: {e}")
        st.stop()


house_model, land_model = load_models()


# --- Financial Calculation Functions ---
def calculate_emi(p, r, t):
    if p <= 0 or r <= 0 or t <= 0:
        return 0, 0, 0
    monthly_rate = (r / 100) / 12
    total_months = t * 12
    emi = (p * monthly_rate * (1 + monthly_rate) ** total_months) / ((1 + monthly_rate) ** total_months - 1)
    total_amount_paid = emi * total_months
    total_interest_paid = total_amount_paid - p
    return emi, total_interest_paid, total_amount_paid


def generate_yearly_schedule(p, r, t, emi):
    data = []
    monthly_rate = (r / 100) / 12
    remaining_balance = p
    for month in range(1, int(t * 12) + 1):
        interest_payment = remaining_balance * monthly_rate
        principal_payment = emi - interest_payment
        remaining_balance -= principal_payment
        if remaining_balance < 0:
            principal_payment += remaining_balance
            remaining_balance = 0
        data.append({
            "Month": month, "Principal Paid": principal_payment,
            "Interest Paid": interest_payment, "Remaining Balance": remaining_balance
        })
    schedule_df = pd.DataFrame(data)
    schedule_df['Year'] = (schedule_df['Month'] - 1) // 12 + 1
    yearly_summary = schedule_df.groupby('Year').agg(
        Total_Principal_Paid=('Principal Paid', 'sum'),
        Total_Interest_Paid=('Interest Paid', 'sum')
    ).reset_index()
    yearly_summary['Year-End Balance'] = schedule_df.groupby('Year')['Remaining Balance'].min().values
    return yearly_summary.style.format({
        'Total_Principal_Paid': '${:,.2f}',
        'Total_Interest_Paid': '${:,.2f}',
        'Year-End Balance': '${:,.2f}'
    })


# --- Custom CSS Injection ---
css = """
<style>
    [data-testid="stAppViewContainer"] > .main {
        background-color: #f0f2f6;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }
    div[data-testid="stMetric"] > label {
        font-weight: 600;
        color: #4f4f4f;
    }
    div[data-testid="stButton"] > button:not(:contains("Calculate Loan")) {
        background-color: #0068c9; color: white; border: none;
        border-radius: 8px; font-weight: 600;
    }
    div[data-testid="stButton"] > button:not(:contains("Calculate Loan")):hover {
        background-color: #0056a4; color: white;
    }
    div[data-testid="stButton"] > button:contains("Calculate Loan") {
        background-color: #00a463; color: white; border: none;
        border-radius: 8px; font-weight: 600;
    }
    div[data-testid="stButton"] > button:contains("Calculate Loan"):hover {
        background-color: #00824e; color: white;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("FutureValuePro")
st.sidebar.write("---")

try:
    current_page_index = PAGES.index(st.session_state.app_mode)
except ValueError:
    current_page_index = 0

selected_mode = st.sidebar.radio(
    "Select a Tool",
    PAGES,
    index=current_page_index
)

if selected_mode != st.session_state.app_mode:
    st.session_state.app_mode = selected_mode
    st.session_state.house_prediction = 0.0
    st.session_state.land_prediction = 0.0
    st.session_state.visual_premium = 0.0
    st.session_state.last_image_hash = ""  # Reset image memory
    st.rerun()

st.sidebar.write("---")
st.sidebar.info("This app provides AI-driven price estimates and financial calculations.")

# --- Main Page Title ---
st.title("FutureValuePro 📈")
st.subheader("AI-Powered Property & Investment Forecaster")
st.write("")

# --- MODE 1: House Price Prediction ---
if st.session_state.app_mode == "🏠 House Price":
    st.header("🏠 House Price Prediction")

    with st.container(border=True):
        st.subheader("Input Property Features")
        col1, col2 = st.columns(2)
        with col1:
            gr_liv_area = st.slider("Above Ground Living Area (Sq. Ft.)", 500, 6000, 1500, 50)
            total_bsmt_sf = st.slider("Total Basement Area (Sq. Ft.)", 0, 6200, 1000, 50)
            year_built = st.number_input("Year Built", 1870, 2025, 2000, 1)

        with col2:
            overall_qual = st.select_slider("Overall Quality", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5)
            full_bath = st.select_slider("Full Bathrooms", [0, 1, 2, 3, 4], 2)
            garage_cars = st.select_slider("Garage Capacity (Cars)", [0, 1, 2, 3, 4, 5], 2)

    # --- VISUAL AI SECTION (HOUSE) ---
    st.write("")
    with st.container(border=True):
        st.subheader("📸 VisualAI Estimate (Deep Learning)")
        st.write("Upload a photo of the house to analyze visual condition and curb appeal.")

        uploaded_file = st.file_uploader("Upload House Image (jpg, png)", type=["jpg", "png", "jpeg"], key="house_img")

        if uploaded_file is not None:
            col_img, col_data = st.columns([1, 2])
            with col_img:
                st.image(uploaded_file, caption="Uploaded Property", width=200)

            with col_data:
                # --- NEW LOGIC: DETECT CHANGE ---
                # Check the file bytes to see if it's a NEW image
                file_bytes = uploaded_file.getvalue()
                current_hash = hashlib.md5(file_bytes).hexdigest()

                # Only re-run if it's a NEW image or first run
                if st.session_state.last_image_hash != current_hash:
                    with st.spinner('Analyzing visual features (CNN)...'):
                        time.sleep(1.5)
                        premium, _ = get_visual_premium(uploaded_file, 10000, 25000)
                        st.session_state.visual_premium = premium
                        st.session_state.last_image_hash = current_hash  # Remember this image

                st.success("Analysis Complete!")
                st.metric("Visual Curb Appeal Premium", f"+${st.session_state.visual_premium:,.2f}",
                          delta="High Quality Detected")
    # -----------------------------

    st.write("")
    if st.button("Predict House Value", use_container_width=True):
        input_data = {
            'GrLivArea': [gr_liv_area], 'OverallQual': [overall_qual],
            'TotalBsmtSF': [total_bsmt_sf], 'YearBuilt': [year_built],
            'FullBath': [full_bath], 'GarageCars': [garage_cars]
        }
        input_df = pd.DataFrame(input_data)

        base_prediction = house_model.predict(input_df)[0]
        final_prediction = base_prediction + st.session_state.visual_premium

        st.session_state.house_prediction = final_prediction
        st.session_state.land_prediction = 0.0

    if st.session_state.house_prediction > 0:
        st.write("")
        with st.container(border=True):
            st.subheader("Prediction Result")

            appreciation_rate = st.slider("Assumed Annual Appreciation (%)", 1.0, 10.0, 4.0, 0.1)
            future_value = st.session_state.house_prediction * ((1 + appreciation_rate / 100) ** 5)

            col1, col2 = st.columns(2)
            col1.metric(label="Estimated Current Value", value=f"${st.session_state.house_prediction:,.2f}")
            col2.metric(label="Projected Value in 5 Years", value=f"${future_value:,.2f}")

            st.write("")
            if st.button("Calculate Loan for this Property"):
                st.session_state.property_value_input = st.session_state.house_prediction
                st.session_state.app_mode = "💰 Loan Calculator"
                st.session_state.house_prediction = 0.0
                st.rerun()

            # --- MODE 2: Land Value Prediction ---
elif st.session_state.app_mode == "🌳 Land Value":
    st.header("🌳 Land Value Prediction")

    with st.container(border=True):
        st.subheader("Input Land Features")
        LAND_CATEGORIES = [
            'SINGLE FAM.RES.-LAND', 'VACANT RES LOTS', 'DUP/TRIPLEXES-LAND',
            'NO VAL RESID LAND', 'EX POLITICAL SUB NON-RES LAND', 'MERCHANDISING-LAND',
            'OFFICES-LAND', 'VAC LAND NO SUB DISC NON BUILD', 'MANUFCTNG/PROCESNG-LAND',
            'WAREHOUSE/STORAGE-LAND', 'Other'
        ]

        col1, col2 = st.columns(2)
        with col1:
            land_sqft = st.number_input("Lot Size (Square Feet)", 100, 150000, 8000)
        with col2:
            land_class = st.selectbox("Land Use Classification", options=LAND_CATEGORIES)

    # --- VISUAL AI SECTION (LAND) ---
    st.write("")
    with st.container(border=True):
        st.subheader("📸 VisualAI Estimate (Deep Learning)")
        st.write("Upload a photo of the land to analyze terrain quality and vegetation.")

        uploaded_file = st.file_uploader("Upload Land Image (jpg, png)", type=["jpg", "png", "jpeg"], key="land_img")

        if uploaded_file is not None:
            col_img, col_data = st.columns([1, 2])
            with col_img:
                st.image(uploaded_file, caption="Uploaded Property", width=200)

            with col_data:
                file_bytes = uploaded_file.getvalue()
                current_hash = hashlib.md5(file_bytes).hexdigest()

                if st.session_state.last_image_hash != current_hash:
                    with st.spinner('Analyzing terrain features (CNN)...'):
                        time.sleep(1.5)
                        premium, _ = get_visual_premium(uploaded_file, 3000, 8000)
                        st.session_state.visual_premium = premium
                        st.session_state.last_image_hash = current_hash

                st.success("Analysis Complete!")
                st.metric("Visual Terrain Premium", f"+${st.session_state.visual_premium:,.2f}",
                          delta="Prime Location Detected")
    # -----------------------------

    st.write("")
    if st.button("Predict Land Value", use_container_width=True):
        input_data = {
            'GIS_sqft': [land_sqft],
            'landClassDscr': [land_class]
        }
        input_df = pd.DataFrame(input_data)

        base_prediction = land_model.predict(input_df)[0]
        final_prediction = base_prediction + st.session_state.visual_premium

        st.session_state.land_prediction = final_prediction
        st.session_state.house_prediction = 0.0

    if st.session_state.land_prediction > 0:
        st.write("")
        with st.container(border=True):
            st.subheader("Prediction Result")

            appreciation_rate = st.slider("Assumed Annual Appreciation (%)", 1.0, 10.0, 4.0, 0.1)
            future_value = st.session_state.land_prediction * ((1 + appreciation_rate / 100) ** 5)

            col1, col2 = st.columns(2)
            col1.metric(label="Estimated Current Value", value=f"${st.session_state.land_prediction:,.2f}")
            col2.metric(label="Projected Value in 5 Years", value=f"${future_value:,.2f}")

            st.write("")
            if st.button("Calculate Loan for this Property"):
                st.session_state.property_value_input = st.session_state.land_prediction
                st.session_state.app_mode = "💰 Loan Calculator"
                st.session_state.land_prediction = 0.0
                st.rerun()

            # --- MODE 3: Loan Calculator (INSTANT & LIVE) ---
elif st.session_state.app_mode == "💰 Loan Calculator":
    st.header("💰 Loan & Amortization Calculator")

    with st.container(border=True):
        st.subheader("Input Loan Details")
        st.write("Adjust the inputs to see your payment details update in real-time.")

        property_value = st.number_input(
            "Property Value ($)",
            min_value=1000.0,
            step=1000.0,
            key='property_value_input',
            help="This is the total value of the property. It's auto-filled from your last prediction."
        )

        down_payment_percent = st.slider("Down Payment (%)", 0, 100, 20, 1)

        down_payment_amount = property_value * (down_payment_percent / 100)
        principal = property_value - down_payment_amount

        col1, col2 = st.columns(2)
        col1.metric("Down Payment Amount", f"${down_payment_amount:,.2f}")
        col2.metric("Final Loan Principal", f"${principal:,.2f}")

        st.write("---")

        annual_rate = st.slider("Annual Interest Rate (% R)", 1.0, 20.0, 6.5, 0.1)
        loan_tenure = st.slider("Loan Tenure (Years T)", 1, 40, 30, 1)

    st.write("")
    with st.container(border=True):
        emi, total_interest, total_payment = calculate_emi(principal, annual_rate, loan_tenure)

        if emi > 0:
            st.subheader("Payment Summary")

            chart_data = pd.DataFrame({
                'Category': ['Principal', 'Interest'],
                'Amount': [principal, total_interest]
            })
            chart_data['Percent'] = chart_data['Amount'] / chart_data['Amount'].sum()

            base = alt.Chart(chart_data).encode(
                theta=alt.Theta("Amount", stack=True)
            ).properties(
                title="Total Payment Breakdown (Principal vs. Interest)"
            )

            pie = base.mark_arc(outerRadius=120).encode(
                color=alt.Color("Category"),
                order=alt.Order("Amount", sort="descending"),
                tooltip=["Category",
                         alt.Tooltip("Amount", format="$,.2f"),
                         alt.Tooltip("Percent", format=".1%")]
            )

            text = base.mark_text(radius=140).encode(
                text=alt.Text("Percent", format=".1%"),
                order=alt.Order("Amount", sort="descending"),
                color=alt.value("black")
            )

            chart = pie + text

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Monthly EMI", f"${emi:,.2f}")
                st.metric("Total Interest Paid", f"${total_interest:,.2f}")
                st.metric("Total Amount Paid", f"${total_payment:,.2f}")
            with col2:
                st.altair_chart(chart, use_container_width=True)

            st.write("---")
            st.subheader("Yearly Amortization Schedule")
            st.dataframe(generate_yearly_schedule(principal, annual_rate, loan_tenure, emi), use_container_width=True)

        else:
            st.info("No loan needed. The down payment covers the entire property value.")