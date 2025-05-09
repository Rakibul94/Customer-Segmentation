
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

df = pd.read_csv('Mall_Customers.csv')

df1 = df[['Annual Income (k$)','Spending Score (1-100)']]
df2 = df[['Age','Spending Score (1-100)']]

st.title("Customer Segmentation using K-Means Clustering")


#EDA on Mall Customers dataset

st.sidebar.markdown("K-Means Clustering")

km = KMeans(n_clusters=5)
y_predicted = km.fit_predict(df1) #Annual Income (k$) and Spending Score (1-100)

df1['Cluster'] = y_predicted
centroids = km.cluster_centers_

# Create scatterplot for Annual Income and Spending Score
fig, ax = plt.subplots()
sns.scatterplot(data=df1, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='tab10', ax=ax)
ax.set_title("Customer Segments based on Annual Income & Spending Score")


ax.scatter(centroids[:, 0], centroids[:, 1], s=100, c='black', marker='*', label='Centroids')
ax.legend()

# Display in Streamlit
st.pyplot(fig)


st.markdown("**It can be observed that having moderate annual income leads to " \
"moderate spending.Some high annual income spend more and some spend less same"
"goes for low annual income**")


km2 = KMeans(n_clusters=4)
y_predicted2 = km2.fit_predict(df2)

df2['cluster'] = y_predicted2
centroids2 = km2.cluster_centers_

# Create scatterplot for Age and Spending Score
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df2, x='Age', y='Spending Score (1-100)',
                hue='cluster', palette='tab10')
ax2.set_title("Customer Segments based on Age & Spending Score")
ax2.scatter(centroids2[:, 0], centroids2[:, 1], s=100, c='black', marker='*', label='Centroids')
ax2.legend()

# Display in Streamlit
st.pyplot(fig2)

st.markdown("**It can be observed that ages between 20 to 40 spend the most.**")