import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

uploaded_file = st.file_uploader("Upload the amazon.csv dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
  # 1. Count by category
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df_clean, x='category', order=df_clean['category'].value_counts().index, ax=ax1)
    ax1.set_title("Product Count by Category")
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    # 2. Revenue distribution
    fig2, ax2 = plt.subplots()
    sns.histplot(df_clean['revenue'], bins=50, kde=True, ax=ax2)
    ax2.set_title("Revenue Distribution")
    st.pyplot(fig2)

    # 3. Discount vs Revenue
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df_clean, x='discount_percentage', y='revenue', ax=ax3)
    ax3.set_title("Discount % vs Revenue")
    st.pyplot(fig3)

    # 4. Rating range vs Revenue
    fig4, ax4 = plt.subplots()
    sns.boxplot(x=pd.cut(df_clean['rating'], bins=[0,2,3,4,5]), y='revenue', data=df_clean, ax=ax4)
    ax4.set_title("Rating Range vs Revenue")
    st.pyplot(fig4)

    # 5. Correlation heatmap
    fig5, ax5 = plt.subplots()
    sns.heatmap(df_clean[['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count', 'revenue']].corr(), annot=True, cmap='coolwarm', ax=ax5)
    ax5.set_title("Correlation Heatmap")
    st.pyplot(fig5)

    st.subheader("🤖 Revenue Prediction Model (Linear Regression)")

    features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']
    X = df_clean[features]
    y = df_clean['revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("### Model Evaluation")
    st.write(f"- Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"- Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):,.2f}")
    st.write(f"- R-squared (R² Score): {r2_score(y_test, y_pred):.2f}")



