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
df.head()

df.columns

df.info()

df.isnull().sum()

df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹','').str.replace(',','').astype(float)
df['actual_price'] = df['actual_price'].astype(str).str.replace('₹','').str.replace(',','').astype(float)
df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%','').astype(float)
df['rating_count'] = df['rating_count'].astype(str).str.replace(',','').astype(float)

df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

df_clean = df.dropna(subset=['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count'])

df = df.dropna(subset=['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count'])

df = df.drop_duplicates()

scaler = MinMaxScaler()
df[['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']] = scaler.fit_transform(
    df[['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']]
)

df['revenue'] = df['discounted_price'] * df['rating_count']

sns.countplot(data=df_clean, x='category', order=df_clean['category'].value_counts().index)

sns.histplot(df_clean['revenue'], bins=50, kde=True)

sns.scatterplot(data=df_clean, x='discount_percentage', y='revenue')

sns.boxplot(x=pd.cut(df_clean['rating'], bins=[0,2,3,4,5]), y='revenue', data=df_clean)

sns.heatmap(df_clean[['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count', 'revenue']].corr(), annot=True)

features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']
X = df_clean[features]
y = df_clean['revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


