import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="ABC Revenue Predictor", layout="wide")

# Title
st.title("📦 ABC Company Product Revenue Analysis")

# Upload file
uploaded_file = st.file_uploader("Upload your Amazon CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head())

    # Preprocessing
    st.subheader("🔧 Data Preprocessing")

    df['discounted_price'] = df['discounted_price'].str.replace('₹','').str.replace(',','').astype(float)
    df['actual_price'] = df['actual_price'].str.replace('₹','').str.replace(',','').astype(float)
    df['discount_percentage'] = df['discount_percentage'].str.replace('%','').astype(float)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = df['rating_count'].str.replace(',','')
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

    df_clean = df.dropna(subset=['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count'])
    df_clean['revenue'] = df_clean['discounted_price'] * df_clean['rating_count']

    st.success("✅ Data cleaned and revenue column added!")

    # Sidebar chart selection
    st.sidebar.title("📊 Visualization Selector")
    show_chart1 = st.sidebar.checkbox("1. Product Count by Category", value=True)
    show_chart2 = st.sidebar.checkbox("2. Revenue Distribution", value=True)
    show_chart3 = st.sidebar.checkbox("3. Discount % vs Revenue", value=True)
    show_chart4 = st.sidebar.checkbox("4. Rating vs Revenue (Boxplot)", value=True)
    show_chart5 = st.sidebar.checkbox("5. Correlation Heatmap", value=True)

    # Visualizations
    st.subheader("📈 Visualizations")

    if show_chart1:
        st.markdown("### 1. Product Count by Category")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        sns.countplot(data=df_clean, x='category', order=df_clean['category'].value_counts().index, ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig1)

    if show_chart2:
        st.markdown("### 2. Revenue Distribution")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.histplot(df_clean['revenue'], bins=50, kde=True, ax=ax2)
        st.pyplot(fig2)

    if show_chart3:
        st.markdown("### 3. Discount % vs Revenue")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df_clean, x='discount_percentage', y='revenue', ax=ax3)
        st.pyplot(fig3)

    if show_chart4:
        st.markdown("### 4. Rating Range vs Revenue (Boxplot)")
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.boxplot(x=pd.cut(df_clean['rating'], bins=[0,2,3,4,5]), y='revenue', data=df_clean, ax=ax4)
        st.pyplot(fig4)

    if show_chart5:
        st.markdown("### 5. Correlation Heatmap")
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        sns.heatmap(df_clean[['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count', 'revenue']].corr(), annot=True, cmap='coolwarm', ax=ax5)
        st.pyplot(fig5)

    # Modeling
    st.subheader("🤖 Forecasting Revenue with Linear Regression")

    features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']
    X = df_clean[features]
    y = df_clean['revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown("### 📉 Model Evaluation Metrics")
    st.write(f"**Mean Absolute Error (MAE)**: `{mae:,.2f}`")
    st.write(f"**Mean Squared Error (MSE)**: `{mse:,.2f}`")
    st.write(f"**R-squared (R²)**: `{r2:.3f}`")

    # Optional: Show coefficients
    coef_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
    st.markdown("### 📌 Model Feature Importance")
    st.dataframe(coef_df)

else:
    st.warning("👆 Please upload a CSV file to begin.")



