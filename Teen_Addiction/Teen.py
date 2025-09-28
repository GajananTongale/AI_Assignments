# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

# ------------------------
# MODEL DICTIONARY
# ------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Naive Bayes": GaussianNB(),
    "Neural Net": MLPClassifier(max_iter=1000),
    "KMeans": KMeans(n_clusters=3, random_state=42)
}

# ------------------------
# STREAMLIT UI
# ------------------------
st.set_page_config(page_title="Addiction Prediction App", layout="wide")

st.title("📱 Student Phone Addiction Prediction")
st.markdown("Predict addiction levels or cluster students based on their daily habits using multiple ML models.")

# Sidebar for model selection
st.sidebar.header("Model Selection & Settings")
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

# Upload dataset
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

# Feature input sliders
st.sidebar.header("Input Features")


def user_input_features():
    data = {}
    features = ["Daily_Usage_Hours", "Sleep_Hours", "Academic_Performance", "Social_Interactions", "Exercise_Hours",
                "Anxiety_Level", "Depression_Level", "Self_Esteem", "Parental_Control", "Screen_Time_Before_Bed",
                "Phone_Checks_Per_Day", "Apps_Used_Daily", "Time_on_Social_Media", "Time_on_Gaming",
                "Time_on_Education", "Family_Communication", "Weekend_Usage_Hours"]

    for feature in features:
        data[feature] = st.sidebar.slider(feature, 0.0, 24.0, 1.0)

    return pd.DataFrame([data])


input_df = user_input_features()

# Display uploaded dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset")
    st.dataframe(df.head())
else:
    st.info("Upload a CSV to see the dataset preview and train models.")

# Scale input features
scaler = StandardScaler()
if uploaded_file is not None:
    X = df.drop(columns=["Addicted_Label"], errors='ignore')
    y = df["Addicted_Label"]
    X_scaled = scaler.fit_transform(X)
    input_scaled = scaler.transform(input_df)
else:
    X_scaled = None
    input_scaled = scaler.fit_transform(input_df)  # just to scale input

# Train and predict
if st.sidebar.button("Predict"):
    model = models[model_name]

    if model_name != "KMeans":
        if uploaded_file is None:
            st.warning("Upload a dataset to train the model.")
        else:
            model.fit(X_scaled, y)
            prediction = model.predict(input_scaled)

            # Map numeric prediction to label
            label_map = {0: "Not Addicted", 1: "Addicted"}
            predicted_label = label_map[prediction[0]]

            st.subheader("Prediction")
            st.write(f"Predicted Addiction Status: {predicted_label}")

            # Show probability for classifiers
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_scaled)
                st.subheader("Prediction Probability")
                st.write(proba)

            # Feature importance for tree-based models
            if hasattr(model, "feature_importances_"):
                st.subheader("Feature Importances")
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=False)
                st.bar_chart(feature_importance_df.set_index("Feature"))
    else:
        # KMeans clustering
        if uploaded_file is None:
            st.warning("Upload a dataset to train KMeans.")
        else:
            model.fit(X_scaled)
            cluster = model.predict(input_scaled)
            st.subheader("KMeans Cluster")
            st.write(f"The input belongs to cluster: {cluster[0]}")

# Extra visualization
if uploaded_file is not None:
    st.subheader("Data Distribution")
    st.bar_chart(df.describe().T["mean"])

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and scikit-learn")
