# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 10:02:26 2025

@author: Sorin
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
transactions = pd.read_excel("Orders2.0.xlsx")

# 1. Group by Customer and SKU to count purchases
reseller_counts = transactions.groupby(['Customer_ID', 'SKU']).size().reset_index(name='Purchase Count')

# 2. Filter for customers who bought the same SKU multiple times
potential_resellers = reseller_counts[reseller_counts['Purchase Count'] > 10] # Adjust the threshold (10) if needed

# 3. Consider time window if needed (e.g., repeat purchases within a year)
# If you have an 'Order Date', you can filter based on a time window.
transactions['Date'] = pd.to_datetime(transactions['Date'])  # Convert to datetime if necessary

# Create the Timedelta
td = pd.Timedelta(days=30)

reseller_data_timed = transactions.merge(potential_resellers[['Customer_ID', 'SKU']], on=['Customer_ID', 'SKU'], how='inner')

resellers_timed = []

for customer in reseller_data_timed['Customer_ID'].unique():
    customer_orders = reseller_data_timed[reseller_data_timed['Customer_ID'] == customer].sort_values(by='Date')
    
    for i in range(1, len(customer_orders)):
        if customer_orders['Date'].iloc[i] - customer_orders['Date'].iloc[i-1] <= td:
            resellers_timed.append(customer)
            break # break from inner loop if condition met


resellers_timed = list(set(resellers_timed))

# 4. Identify Resellers
# Customers meeting the criteria are considered resellers
resellers = potential_resellers['Customer_ID'].unique()

print("Potential Resellers:\n", resellers)

# Rank resellers by purchase count or total sales value
reseller_ranking = potential_resellers.groupby('Customer_ID')['Purchase Count'].sum().sort_values(ascending=False)



# Calculate total sales value for each reseller
reseller_sales = transactions[transactions['Customer_ID'].isin(resellers)].groupby('Customer_ID')['Sales_Amount'].sum().sort_values(ascending=False)  # Replace 'Price' with the actual sales value column if different


# Combine ranking and sales
reseller_summary = pd.DataFrame({'Purchase Count': reseller_ranking, 'Total Sales': reseller_sales})

print("\nReseller Summary:\n", reseller_summary)

# Calculate the most popular product for each reseller

# Filter sales data for only reseller purchases
reseller_sales = transactions[transactions['Customer_ID'].isin(resellers)]

# Calculate the most popular product for each reseller
most_popular_product = reseller_sales.groupby('Customer_ID')['SKU'].agg(lambda x: x.value_counts().index[0]).reset_index()
most_popular_product.rename(columns={'SKU': 'Most Popular Product'}, inplace=True)

# Calculate the Number of times most popular product is purchased by reseller
most_popular_product_count = reseller_sales.groupby(['Customer_ID', 'SKU'])['SKU'].count().reset_index(name='Product Count')
most_popular_product_count = most_popular_product_count.groupby('Customer_ID').max().reset_index()


most_popular_product = pd.merge(most_popular_product, most_popular_product_count, on='Customer_ID', how="left")



print("Most Popular Product by Reseller:\n", most_popular_product)
