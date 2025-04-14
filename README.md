# Churn Predictor App

A customizable churn analytics solution for data-driven teams.
Ingest your customer data, generate predictions, and explore risk breakdowns visually and interactively.

## Try It Out

Try the churn prediction tool in your browser:  
[Click here to open the Streamlit App](https://churn-predictor-app-er7gei7fod3qck79vpfv2j.streamlit.app/)

[![Streamlit App](https://img.shields.io/badge/Live%20App-Click%20to%20Open-red?logo=streamlit)][Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## Features

- **Upload CSV datasets** from any domain
- **Auto-train Voting Classifier** (RandomForest + XGBoost)
- **Select target column** dynamically
- **Fuzzy search & dropdown** to lookup customers
- **KPI Cards**: Total customers, churn rate, avg. risk score
- **Charts**: Churn segments, confusion matrix, risk by category
- **Download predictions** (Top 50 risky + full output)
- Works on Telco, Bank, Retail, SaaS, Insurance, etc.
- Designed to be **modular**, **scalable**, and **client-ready**

---

## Live Demo

Launch here → [churn-predictor-app-er7gei7fod3qck79vpfv2j.streamlit.app](https://churn-predictor-app-er7gei7fod3qck79vpfv2j.streamlit.app/)

---

## Tech Stack

- Python 3.11
- Streamlit
- scikit-learn
- XGBoost
- imbalanced-learn
- RapidFuzz (fuzzy matching)
- matplotlib, pandas

---

## Setup Instructions

```bash
git clone https://github.com/ManojM9894/Churn-Predictor-App.git
cd Churn-Predictor-App
pip install -r requirements.txt
streamlit run app.py
```

---

## Folder Structure

```bash
multi-industry-churn-predictor/
├── app.py               # Main Streamlit App
├── train_dynamic.py     # Model training & prediction logic
├── requirements.txt     # Dependencies
├── data/                # Sample datasets (optional)
└── README.md            # This file
```

---

## Use Cases

| Industry       | Application Example                        |
|----------------|---------------------------------------------|
| Telecom        | Predict churn for prepaid/postpaid users   |
| Banking        | Predict likelihood of customer attrition   |
| SaaS / B2B     | Flag clients at risk of cancellation       |
| Insurance      | Detect policyholders likely to switch      |
| E-commerce     | Identify repeat buyers vs one-time users   |

---

## Sample Datasets

To test the app, this repo includes 2 sample datasets inside the `/data` folder:

| Dataset             | Description                          |
|---------------------|--------------------------------------|
| `telco_churn.csv`   | Telecom customer churn data          |
| `bank_churn.csv`    | Bank customer attrition prediction   |

Once cloned, simply run the app and upload either dataset manually via the Streamlit file uploader.

---

## Downloads

- Top 50 Risky Customers → `top_50_risky_customers.csv`
- Full Predictions        → `churn_predictions.csv`

---

## License

MIT © [Manoj Mandava](https://github.com/ManojM9894)

---

## Connect with Me

- GitHub: [@ManojM9894](https://github.com/ManojM9894)
- LinkedIn:[LinkedIn](https://www.linkedin.com/in/manojmandava9894) 

> If you found this helpful, give it a ⭐ on GitHub to support the project!
