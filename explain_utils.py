# explain_utils.py
def display_shap_bar_chart(model, input_df, class_names=None):
    import shap
    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st

    st.subheader("💡 SHAP Explanation (Bar Chart)")

    try:
        # 🔑 Extract individual base model from VotingClassifier
        base_model = model.named_estimators_["xgb"]  # You can also try "rf"

        # 🧹 Ensure numeric input only
        numeric_df = input_df.select_dtypes(include=[np.number]).astype(np.float64)

        # 🚫 Drop target column if present (super important)
        if "Churn" in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=["Churn"])

        # ✅ SHAP Explanation
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(numeric_df)

        plt.clf()

        # 💡 Pick the right class (usually 1 = churn)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap.summary_plot(shap_values[1], numeric_df, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, numeric_df, plot_type="bar", show=False)

        st.pyplot(plt.gcf())

    except Exception as e:
        st.error("SHAP Error: Could not generate bar chart.")
        st.text(f"Details: {e}")
