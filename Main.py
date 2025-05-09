
import streamlit as st

# Define the pages
main_page = st.Page("Customer_Segmentation_app.py", title="K-Means Clustering")
page_2 = st.Page("Mall_Customers_EDA.py", title="EDA on Mall Customers dataset")


# Set up navigation
pg = st.navigation([main_page, page_2])

# Run the selected page
pg.run()
