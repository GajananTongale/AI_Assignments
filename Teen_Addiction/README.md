# 📱 Student Phone Addiction Prediction App

A **Streamlit web application** that predicts **student phone addiction** levels based on daily habits, mental health indicators, and usage patterns using multiple machine learning models. The app also supports **clustering students** using KMeans.

You can test the live app here: [Phone Addiction Prediction App](https://phone-addiction-app.streamlit.app/)

---

## Features

- **Multiple ML Models**:  
  - Logistic Regression  
  - K-Nearest Neighbors (KNN)  
  - Support Vector Machine (SVM)  
  - Decision Tree  
  - Random Forest  
  - Gradient Boosting  
  - Naive Bayes  
  - Neural Network (MLP)  
  - KMeans Clustering  

- **Custom Feature Input**: Input 17 student-related features using **interactive sliders**.  

- **Data Upload**: Upload your own CSV dataset to train models dynamically.  

- **Feature Scaling**: All features are scaled automatically using `StandardScaler`.  

- **Prediction Output**:
  - Classification models output **Addicted / Not Addicted**.
  - Prediction probability for classifiers that support it.  
  - Feature importance visualization for tree-based models.  
  - KMeans clustering for unsupervised grouping.  

- **Data Visualization**:  
  - Dataset preview  
  - Summary statistics  
  - Feature importance bar charts  

- **Clean & Interactive UI** using Streamlit.

---

## Input Features

| Feature | Description |
|---------|-------------|
| Daily_Usage_Hours | Hours spent on phone daily |
| Sleep_Hours | Hours of sleep per day |
| Academic_Performance | Academic performance score |
| Social_Interactions | Social interaction rating |
| Exercise_Hours | Daily exercise hours |
| Anxiety_Level | Anxiety rating |
| Depression_Level | Depression rating |
| Self_Esteem | Self-esteem score |
| Parental_Control | Level of parental control |
| Screen_Time_Before_Bed | Hours spent on screen before bed |
| Phone_Checks_Per_Day | Number of times phone checked per day |
| Apps_Used_Daily | Number of apps used daily |
| Time_on_Social_Media | Daily hours on social media |
| Time_on_Gaming | Daily hours on gaming |
| Time_on_Education | Daily hours on educational apps |
| Family_Communication | Family communication rating |
| Weekend_Usage_Hours | Hours spent on phone on weekends |
| Addiction_Level | Target label (1 = Addicted, 0 = Not Addicted) |

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/addiction-prediction-app.git
cd addiction-prediction-app
