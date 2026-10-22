# 🔮 FutureTwin AI — Business Future Simulation Platform

🚀 **Live Demo:** [Try FutureTwin AI](https://futuretwin-ai.streamlit.app//)

💻 **GitHub Repository:** [View Source Code](https://github.com/coder-priyanka01/FutureTwin-Ai)

**Simulate possible business futures using machine learning, scenario modelling, and interactive decision support.**


FutureTwin AI is a machine-learning-based business simulation platform designed to help users explore possible business outcomes through revenue prediction, demand analysis, customer segmentation, product text analysis, and what-if scenario simulation.

The project combines trained machine learning models with interactive scenario modelling to support data-driven business decisions.

## 🚀 Key Features

- **Revenue Prediction:** Estimate revenue using a trained Random Forest Regressor.
- **Future Demand Prediction:** Classify demand using a Random Forest Classifier.
- **Customer Segmentation:** Group customers into segments using K-Means clustering.
- **Product Text Analysis:** Analyse product text using TF-IDF and Logistic Regression.
- **What-If Scenario Simulation:** Explore how changes in demand and price can affect simulated revenue.
- **Interactive Dashboard:** Explore business scenarios through a Streamlit interface.
- **Scenario Comparison:** Compare different simulated business outcomes.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Text Analysis:** TF-IDF, Logistic Regression
- **Clustering:** K-Means
- **Visualization:** Altair, Matplotlib
- **Model Persistence:** Joblib

## 🤖 Machine Learning Models

| Component | Technique |
|---|---|
| Revenue Prediction | Random Forest Regressor |
| Demand Prediction | Random Forest Classifier |
| Customer Segmentation | K-Means Clustering |
| Product Text Analysis | TF-IDF + Logistic Regression |
| Scenario Simulation | ML baseline + what-if adjustments |

## 📂 Project Structure

```text
FutureTwin-Ai/
├── .devcontainer/
├── .gitignore
├── Untitled.ipynb
├── app.py
├── requirements.txt
├── futuretwin_cluster_features.pkl
├── futuretwin_config.pkl
├── futuretwin_customer_scaler.pkl
├── futuretwin_demand_features.pkl
├── futuretwin_demand_model.pkl
├── futuretwin_kmeans.pkl
├── futuretwin_nlp_model.pkl
├── futuretwin_revenue_features.pkl
├── futuretwin_revenue_model.pkl
└── futuretwin_tfidf.pkl
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/coder-priyanka01/FutureTwin-Ai.git
```

### 2. Open the project folder

```bash
cd FutureTwin-Ai
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 How It Works

1. The application loads trained machine learning models and their associated feature configurations.
2. Users enter business and product information through the interactive interface.
3. The platform generates model-based predictions and demand insights.
4. Users explore hypothetical changes in price and demand.
5. Scenario results can be compared to understand how different assumptions affect simulated outcomes.

**Note:** Scenario simulations are estimates based on model predictions and user-defined assumptions. They do not guarantee actual future business performance.

## 🎯 Project Objective

The objective of FutureTwin AI is to demonstrate how machine learning and scenario modelling can be combined to explore business possibilities, understand potential risks, and support informed decision-making.

## 👩‍💻 Author

**Priyanka Kumari**

Aspiring Data Scientist | Machine Learning | Python | Data Analytics

- **GitHub:** [@coder-priyanka01](https://github.com/coder-priyanka01)
- **Project Repository:** [FutureTwin-Ai](https://github.com/coder-priyanka01/FutureTwin-Ai)

---

*Built as a machine learning and business analytics project to explore data-driven decision support.*
