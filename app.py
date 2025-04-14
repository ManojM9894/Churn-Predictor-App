import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from train_dynamic import (
    train_model_from_upload,
    apply_encoders,
    predict_with_model
)

st.set_page_config(page_title="Churn Predictor App")
st.title("\U0001F4C8 Churn Predictor App")

uploaded_file = st.file_uploader("Upload your CSV or Excel dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    if st.checkbox("Preview uploaded data"):
        st.dataframe(df.head())

    excluded = ["customerID", "id"]
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and col not in excluded]

    target_column = st.selectbox("Select the target", binary_cols)

    if "model" not in st.session_state:
        st.session_state.model = None
        st.session_state.encoders = None
        st.session_state.feature_cols = None
        st.session_state.df_results = None
        st.session_state.top_50 = None

    if st.button("Train Model"):
        model, encoders, feature_cols = train_model_from_upload(df, target_column)
        df_encoded = apply_encoders(df.copy(), encoders)
        df_encoded = df_encoded[feature_cols]
        preds, probs = predict_with_model(model, df_encoded)
        df_results = df.copy()
        df_results["Prediction"] = preds
        df_results["Probability"] = probs

        # Save to session
        st.session_state.model = model
        st.session_state.encoders = encoders
        st.session_state.feature_cols = feature_cols
        st.session_state.df_results = df_results
        st.session_state.top_50 = df_results.sort_values(by="Probability", ascending=False).head(50)

    if st.session_state.df_results is not None:
        model = st.session_state.model
        encoders = st.session_state.encoders
        feature_cols = st.session_state.feature_cols
        df_results = st.session_state.df_results
        top_50 = st.session_state.top_50

        st.subheader("\U0001F4B8 Top 50 Customers Likely to Churn")
        st.dataframe(
            top_50[["customerID", "Prediction", "Probability"]] if "customerID" in top_50.columns else top_50[["Prediction", "Probability"]]
        )

        top_csv = top_50.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Top 50 Risky Customers", top_csv, "top_50_risky_customers.csv", "text/csv")

        # --- Determine identifier column ---
        preferred_ids = ["customerID", "Customer Name", "Name", "CustomerId", "RowNumber", "ID"]
        customer_id_col = next((col for col in preferred_ids if col in df.columns and df[col].is_unique), None)
        if customer_id_col is None:
            df = df.reset_index()  # Ensure index is usable
            customer_id_col = "index"

        st.subheader("👤 Predict Churn for a Customer")
        st.markdown("Select a customer from all customers, sorted by churn risk.")
        st.caption(f"Using `{customer_id_col}` as the customer identifier column.")

        if customer_id_col in df_results.columns:
            sorted_ids = df_results.sort_values(by="Probability", ascending=False)[customer_id_col].astype(str).tolist()
        else:
            sorted_ids = df_results.index.astype(str).tolist()

        entry_mode = st.radio("Choose input method:", ["Dropdown", "Type to search by ID or Name"], horizontal=True)

        if entry_mode == "Dropdown":
            manual_id = st.selectbox("Select Customer ID (sorted by churn risk):", options=sorted_ids, index=0, key="manual_id_dropdown")
        else:
            manual_id = st.text_input("Search by customer ID or name:", key="manual_id_text")

        active_id = manual_id

        if active_id:
            try:
                if customer_id_col in df.columns:
                    selected_row = df[df[customer_id_col].astype(str).str.contains(active_id, case=False, na=False)]
                else:
                    selected_row = df[df.index.astype(str) == str(active_id)]

                if selected_row.empty:
                    raise ValueError("No matching customer found.")

                encoded_row = apply_encoders(selected_row.copy(), encoders)
                encoded_row = encoded_row[feature_cols]
                pred_prob = model.predict_proba(encoded_row)[0][1]
                churn_risk = "🚨 High Risk" if pred_prob > 0.7 else "⚠️ Medium Risk" if pred_prob > 0.4 else "✅ Low Risk"
                st.markdown(f"### Prediction for `{active_id}`")
                st.metric("Churn Probability", f"{pred_prob * 100:.2f}%", help=churn_risk)
                st.dataframe(selected_row)
            except Exception as e:
                st.error(f"Customer not found or invalid input: {e}")

        # Global Feature Importance
        st.subheader("📊 Top Churn Drivers (Global Feature Importance)")
        rf_model = model.named_estimators_["rf"]
        importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        importances.plot(kind="bar", ax=ax)
        ax.set_title("Top Churn Indicators")
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)

        st.subheader("📝 Download All Predictions")
        csv_data = df_results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download All Predictions", csv_data, "churn_predictions.csv", "text/csv")
else:
    st.info("Please upload a CSV file to get started.")
