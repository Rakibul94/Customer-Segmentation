import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.title("Exploratory Data Analysis on Mall Customers dataset")
st.sidebar.markdown("EDA")

df = pd.read_csv('Mall_Customers.csv')



le = LabelEncoder()

df['Genre'] = le.fit_transform(df['Genre'])

st.subheader("📊 Statstical Summary")

st.table(df.describe())


# Create the plot




st.subheader("Correlation Matrix")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)


# Show it in Streamlit
st.pyplot(fig)

st.subheader("📊 Feature Distributions")

# Loop through columns
for col in df.columns:
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f'Distribution of {col}')
    st.pyplot(fig)
