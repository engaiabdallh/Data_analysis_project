import streamlit as st
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_dataset
from utils.visualizations_before import visualize_dataset, plot_boxplots, plot_correlation_heatmap, visualize_categorical
from utils.preprocessing import preprocess_titanic
from utils.visualizations_after import visualize_dataset_af, plot_boxplots_af, plot_correlation_heatmap_af, visualize_categorical_af
from utils.modeling import modeling_classifier

st.set_page_config(page_title="DA App", layout="wide")
st.title("🤖 Data Analysis GUI Project")

if "df" not in st.session_state:
    st.session_state["df"] = None
if "df_processed" not in st.session_state:
    st.session_state["df_processed"] = None

uploaded_file = st.file_uploader("📂 Upload your dataset (.csv)", type="csv")

if uploaded_file:
    st.session_state["df"] = load_dataset(uploaded_file)
    df = st.session_state["df"]

    tab_preview, tab_before, tab_preprocess, tab_after, tab_after_prev, tab_model = st.tabs([
        "📊 Preview (Before)", "📈 Visualize (Before)", "🧹 Preprocess", "📈 Visualize (After)", "Preview (After)", "🤖 Models"
    ])

    with tab_preview:
        st.subheader("Data Preview")
        st.write(df)
        col1, col2 = st.columns(2)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        
        st.write("Data Types:")
        st.write(df.dtypes)

        null_percent = (df.isnull().sum() / df.shape[0] * 100).round(2)
        st.write("🧪 Null Percentage (%) per Column:")
        st.dataframe(null_percent.to_frame(name="Null %").style.background_gradient(cmap='Reds'))

        st.markdown("## 🧩 Missing Value Visualization")
        fig, ax = plt.subplots(figsize=(30, 15))
        msno.matrix(df, ax=ax)
        st.pyplot(fig)

        num_desc = df.describe(include="number").T
        cat_desc = df.describe(include="object").T

        if not num_desc.empty:
            st.write("## Describe Numeric Data", num_desc)
        else:
            st.info("No numeric data available.")

        if not cat_desc.empty:
            st.write("## Describe Categorical Data", cat_desc)
        else:
            st.info("No categorical data available.")

        duplicates = df[df.duplicated()]
        st.write(f"Total duplicated rows: {duplicates.shape[0]}")
        if not duplicates.empty:
            st.dataframe(duplicates.reset_index(drop=True))

    with tab_before:
        st.markdown("## 📊 Full Data Visualization")

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()

        st.markdown("## Numeric Column Visualizations")
        visualize_dataset(df, numeric_cols=numeric_cols)

        if len(numeric_cols) > 1:
            st.markdown("### Correlation Heatmap")
            plot_correlation_heatmap(df[numeric_cols])
            plot_boxplots(df, numeric_cols=numeric_cols)

        if categorical_cols:
            st.markdown("## Categorical Column Visualizations")
            visualize_categorical(df, categorical_cols)

    with tab_preprocess:
        st.subheader("🧹 Preprocessing")
        
        if st.button("🚀 Run Preprocessing"):
            if df is not None:
                try:
                    df_processed, logs = preprocess_titanic(df, display_log=True)
                    st.success("Preprocessing Done ✅")
                    st.session_state["df_processed"] = df_processed

                    st.markdown("### 🔍 Processed Preview")
                    st.dataframe(df_processed.head())

                    st.markdown("### 📝 Processing Log")

                    st.markdown("#### Columns dropped due to null values:")
                    st.write(logs["dropped_columns"])

                    st.markdown("#### Applied transformations:")

                    st.markdown("**Dropped Columns:**")
                    st.write(logs["dropped_columns"])

                    st.markdown("**Imputed Numeric Columns:**")
                    st.write(logs["imputed_numeric"])

                    st.markdown("**Imputed Categorical Columns:**")
                    st.write(logs["imputed_categorical"])

                    st.markdown("**Encoded Columns:**")
                    st.write(logs["encoded"])

                    st.markdown("**Scaled Columns:**")
                    st.write(logs["scaled"])

                    st.info("You can now go to the '📈 Visualize After' tab to explore the cleaned data.")
                except Exception as e:
                    st.error(f"An error occurred during preprocessing: {str(e)}")
            else:
                st.error("Please upload a dataset first!")

    with tab_after:
        st.markdown("## 📊 Full Data Visualization After Preprocessing")

        if st.session_state.get("df_processed") is not None:
            df_processed = st.session_state["df_processed"]

            numeric_cols = df_processed.select_dtypes(include='number').columns.tolist()
            categorical_cols = df_processed.select_dtypes(include='category').columns.tolist()

            st.markdown("## Numeric Column Visualizations")
            visualize_dataset_af(df_processed, numeric_cols, categorical_cols)

            if len(numeric_cols):
                st.markdown("### Correlation Heatmap")
                plot_correlation_heatmap_af(df_processed[numeric_cols])
                plot_boxplots_af(df_processed, numeric_cols=numeric_cols)

            if categorical_cols:
                st.markdown("## Categorical Column Visualizations")
                visualize_categorical_af(df_processed, categorical_cols)

    with tab_after_prev:
        if st.session_state.get("df_processed") is not None:
            df_processed = st.session_state["df_processed"]
            st.subheader("Data Preview")
            st.write(df_processed)
            col1, col2 = st.columns(2)
            col1.metric("Rows", df_processed.shape[0])
            col2.metric("Columns", df_processed.shape[1])
            
            st.write("Data Types:")
            st.write(df_processed.dtypes)

            null_percent = (df_processed.isnull().sum() / df_processed.shape[0] * 100).round(2)
            st.write("🧪 Null Percentage (%) per Column:")
            st.dataframe(null_percent.to_frame(name="Null %").style.background_gradient(cmap='Reds'))

            st.markdown("## 🧩 Missing Value Visualization")
            fig, ax = plt.subplots(figsize=(30, 15))
            msno.matrix(df_processed, ax=ax)
            st.pyplot(fig)

            num_desc = df_processed.describe(include="number").T

            if not num_desc.empty:
                st.write("## Describe Numeric Data", num_desc)
            else:
                st.info("No numeric data available.")
        else:
            st.error("Please upload a dataset first!")

    with tab_model:
        if st.session_state.get("df_processed") is not None:
            df_processed = st.session_state["df_processed"]
            st.subheader("Modeling Classifiers")
            if st.button("Train Model"):
                with st.spinner("Training the model..."):
                    result = modeling_classifier(df_processed, drop_cols=['alive'])
                st.success("Model trained successfully!")

                st.subheader("🔍 Model Evaluation Metrics")
                tab1, tab2 = st.tabs(["🎯 Voting Classifier", "🧠 Logistic Regression (K-Fold)"])

                with tab1:
                    st.markdown("### ✅ Voting Classifier Metrics")
                    tts_result = result["train_test_split"]
                    col1s, cols2 = st.columns(2)

                    with col1s:
                        row1_col1, row1_col2 = st.columns(2)
                        row1_col1.metric("🎯 Accuracy", f"{tts_result['accuracy']*100:.2f}%")
                        row1_col2.metric("🧠 Precision", f"{tts_result['precision']*100:.2f}%")

                        row2_col1, row2_col2 = st.columns(2)
                        row2_col1.metric("❤️ Recall", f"{tts_result['recall']*100:.2f}%")
                        row2_col2.metric("🔥 F1 Score", f"{tts_result['f1_score']*100:.2f}%")

                    with cols2:
                        st.markdown("#### Confusion Matrix")
                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.heatmap(
                            tts_result['confusion_matrix'],
                            annot=True,
                            fmt='d',
                            cmap='Reds',
                            cbar=False,
                            xticklabels=["Pred: No", "Pred: Yes"],
                            yticklabels=["Actual: No", "Actual: Yes"],
                            linewidths=0.5,
                            linecolor='black',
                            ax=ax,
                            annot_kws={"size": 12}
                        )
                        ax.set_title("Confusion Matrix", fontsize=12)
                        ax.set_xlabel("Predicted Label", fontsize=10)
                        ax.set_ylabel("True Label", fontsize=10)
                        plt.tight_layout()
                        st.pyplot(fig)

                with tab2:
                    st.markdown("### 🔁 Logistic Regression (Fold CV)")
                    kfs_result = result["kfold_split"]
                    col1s_lr, cols2_lr = st.columns(2)

                    with col1s_lr:
                        row1_col1, row1_col2 = st.columns(2)
                        row1_col1.metric("🎯 Accuracy", f"{kfs_result['accuracy']*100:.2f}%")
                        row1_col2.metric("🧠 Precision", f"{kfs_result['precision']*100:.2f}%")

                        row2_col1, row2_col2 = st.columns(2)
                        row2_col1.metric("❤️ Recall", f"{kfs_result['recall']*100:.2f}%")
                        row2_col2.metric("🔥 F1 Score", f"{kfs_result['f1_score']*100:.2f}%")

                    with cols2_lr:
                        st.markdown("#### Confusion Matrix (Summed over 5 folds)")
                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.heatmap(
                            kfs_result['confusion_matrix'],
                            annot=True,
                            fmt='d',
                            cmap='Reds',
                            cbar=False,
                            xticklabels=["Pred: No", "Pred: Yes"],
                            yticklabels=["Actual: No", "Actual: Yes"],
                            linewidths=0.5,
                            linecolor='black',
                            ax=ax,
                            annot_kws={"size": 12}
                        )
                        ax.set_title("Confusion Matrix", fontsize=12)
                        ax.set_xlabel("Predicted Label", fontsize=10)
                        ax.set_ylabel("True Label", fontsize=10)
                        plt.tight_layout()
                        st.pyplot(fig)
        else:
            st.error("Please upload a dataset first!")
