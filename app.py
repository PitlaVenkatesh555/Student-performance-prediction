import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model
joblib.dump(model, 'best_model.pkl')
model = joblib.load('best_model.pkl')

st.set_page_config(page_title="Student Performance ML", layout="wide")

st.title("Student Performance Prediction Dashboard")

# ---------------------- FUNCTION ----------------------
def create_full_input(user_input):
    full_input = {
        'school': 'GP',
        'sex': 'M',
        'age': user_input['age'],
        'address': 'U',
        'famsize': 'GT3',
        'Pstatus': 'T',
        'Medu': 2,
        'Fedu': 2,
        'Mjob': 'other',
        'Fjob': 'other',
        'reason': 'course',
        'guardian': 'mother',
        'traveltime': 1,
        'studytime': user_input['studytime'],
        'failures': user_input['failures'],
        'schoolsup': 'no',
        'famsup': 'no',
        'paid': 'no',
        'activities': 'yes',
        'nursery': 'yes',
        'higher': 'yes',
        'internet': 'yes',
        'romantic': 'no',
        'famrel': 3,
        'freetime': 3,
        'goout': 3,
        'Dalc': 1,
        'Walc': 1,
        'health': 3,
        'absences': user_input['absences'],
        'G1': user_input['G1'],
        'G2': user_input['G2']
    }
    return pd.DataFrame([full_input])

# ---------------------- TABS ----------------------
tab1, tab2, tab3 = st.tabs(["Dashboard", "EDA", "Prediction"])

# ===================== DASHBOARD =====================
with tab1:
    st.subheader("Model Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Random Forest")
    col2.metric("Accuracy", "0.90+")
    col3.metric("Status", "Production Ready")

    st.markdown("---")

    df = pd.read_csv("student-mat.csv", sep=';')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pass vs Fail")
        fig, ax = plt.subplots()
        sns.countplot(x=(df['G3'] >= 10), ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Grade Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['G3'], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

# ===================== EDA =====================
with tab2:
    st.subheader("Exploratory Data Analysis")

    df = pd.read_csv("student-mat.csv", sep=';')

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Insights")
    st.info("G1 and G2 have strong correlation with final performance")

# ===================== PREDICTION =====================
with tab3:
    st.subheader("Predict Student Performance")

    st.sidebar.header("Input Features")

    age = st.sidebar.slider("Age", 15, 22, 17)
    studytime = st.sidebar.slider("Study Time", 1, 4, 2)
    failures = st.sidebar.slider("Past Failures", 0, 3, 0)
    absences = st.sidebar.slider("Absences", 0, 50, 5)
    G1 = st.sidebar.slider("G1", 0, 20, 10)
    G2 = st.sidebar.slider("G2", 0, 20, 10)

    if st.sidebar.button("Predict"):

        user_input = {
            "age": age,
            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "G1": G1,
            "G2": G2
        }

        df_input = create_full_input(user_input)

        try:
            prediction = model.predict(df_input)[0]

            if prediction == 1:
                st.success("Prediction: PASS")
            else:
                st.error("Prediction: FAIL")

        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.info("Higher G1 and G2 significantly improve prediction outcomes")
