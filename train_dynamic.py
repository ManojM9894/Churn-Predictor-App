import matplotlib.pyplot as plt
import shap
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

def train_model_from_upload(df, target_column):
    df = df.copy()

    # Drop ID-like columns
    for col in df.columns:
        if 'id' in col.lower():
            df.drop(columns=[col], inplace=True)

    # Encode categorical features
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != target_column:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Encode target column if needed
    if df[target_column].dtype == 'object':
        df[target_column] = LabelEncoder().fit_transform(df[target_column].astype(str))

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle class imbalance
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

    # Build and train VotingClassifier
    rf = RandomForestClassifier(random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    model = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb)], voting="soft")
    model.fit(X_res, y_res)

    return model, encoders, X.columns.tolist()

def predict_with_model(model, df):
    return model.predict(df), model.predict_proba(df)[:, 1]

def apply_encoders(df, encoders):
    df = df.copy()
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))
    return df

def generate_shap_explanation(model, single_row):
    try:
        rf_model = model.named_estimators_["rf"]
        explainer = shap.Explainer(rf_model, single_row)
        shap_values = explainer(single_row)

        fig = plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values, show=False)
        plt.title("Top Feature Contributions")
        plt.tight_layout()
    except Exception as e:
        fig = plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, f"SHAP Error:\n{e}", ha="center", va="center", fontsize=12)
    return fig

def generate_shap_bar_plot(model, input_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        rf_model = model.named_estimators_["rf"]
        numeric_df = input_df.select_dtypes(include=[float, int]).copy()
        if "Churn" in numeric_df.columns:
            numeric_df.drop(columns=["Churn"], inplace=True)

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(numeric_df)

        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap.summary_plot(shap_values[1], numeric_df, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, numeric_df, plot_type="bar", show=False)
    except Exception as e:
        ax.text(0.5, 0.5, f"SHAP Error:\n{e}", ha="center", va="center", fontsize=12)
    return fig
