import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_dataset(df, numeric_cols=None):
    if numeric_cols:
        st.markdown("### Histograms Overview")
        cols = st.columns(4)
        for i, col in enumerate(numeric_cols):
            with cols[i % 4]:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=False, ax=ax, color='skyblue')
                ax.set_title(col)
                ax.set_xlabel("")
                ax.set_ylabel("")
                fig.tight_layout()
                st.pyplot(fig)

def plot_boxplots(df, numeric_cols):
    st.markdown("### Boxplots Overview")
    cols = st.columns(4)
    for i, col in enumerate(numeric_cols):
        with cols[i % 4]:
            fig, ax = plt.subplots()
            sns.boxplot(y=df[col], ax=ax, color='lightgreen')
            ax.set_title(col)
            fig.tight_layout()
            st.pyplot(fig)

def plot_correlation_heatmap(df_numeric):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    st.pyplot(fig)

def visualize_categorical(df, categorical_cols):
    st.markdown("### Count Plots Overview")
    cols = st.columns(4)
    for i, col in enumerate(categorical_cols):
        with cols[i % 4]:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind='bar', ax=ax, color='orchid')
            ax.set_title(col)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            fig.tight_layout()
            st.pyplot(fig)

    st.markdown("### Pie Charts Overview")
    cols = st.columns(4)
    for i, col in enumerate(categorical_cols):
        with cols[i % 4]:
            fig, ax = plt.subplots()
            top_vals = df[col].value_counts().nlargest(10)
            top_vals.plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_ylabel("")
            ax.set_title(col)
            fig.tight_layout()
            st.pyplot(fig)
