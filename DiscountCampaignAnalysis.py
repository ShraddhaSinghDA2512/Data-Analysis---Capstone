import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
transactions = pd.read_excel("Orders2.0.xlsx")
customers = pd.read_excel("Customers.xlsx")

# Calculate the top price for each SKU
top_prices = transactions.groupby('SKU')['Price'].max().reset_index()
top_prices.rename(columns={'Price': 'Top Price'}, inplace=True)

# Merge top prices back into the sales DataFrame
transactions = pd.merge(transactions, top_prices, on='SKU', how='left')

# Identify discount campaigns
transactions['Discount Campaign'] = transactions['Price'] < transactions['Top Price']


# Calculate the dates of discount campaigns for each product type
discount_campaign_dates = transactions[transactions['Discount Campaign']].groupby('SKU')['Date'].unique().reset_index()
discount_campaign_dates.to_excel('discount_campaign_dates.xlsx', index=False)


# 1. Campaign Frequency:

# Calculate the number of discount campaigns for each product type
campaign_frequency = discount_campaign_dates.groupby('SKU')['Date'].count().reset_index()
campaign_frequency.rename(columns={'Date': 'Number of Campaigns'}, inplace=True)

campaign_frequency.to_excel('campaign_frequency.xlsx', index=False)

# 2. Extract Overall Campaign Information


# A. Campaign Periods by Product Type:
campaigns_by_product = transactions[transactions['Discount Campaign']].groupby('SKU')['Date'].agg(['min', 'max']).reset_index()
campaigns_by_product.columns = ['SKU', 'Campaign Start', 'Campaign End']


print("Campaign Periods by Product Type:\n", campaigns_by_product)




# B. All Campaign Periods (regardless of product):
overall_campaign_periods = transactions[transactions['Discount Campaign']]['Date'].agg(['min', 'max']).reset_index()

overall_campaign_periods.columns = ['min/max', 'Date']
print("\nOverall Campaign Periods:\n", overall_campaign_periods)





# C.  Unique Campaign Dates:
unique_campaign_dates = sorted(transactions[transactions['Discount Campaign']]['Date'].unique())
print("\nUnique Campaign Dates:\n", unique_campaign_dates)




# D.  SKUs on discount during campaigns:
campaign_skus = transactions[transactions['Discount Campaign']].groupby('Date')['SKU'].unique().reset_index()

print("\nSKUs on Discount During Each Campaign Date:\n", campaign_skus)




# 3. Further Analysis (Optional):

# Calculate the duration of overall campaigns
overall_campaign_duration = (overall_campaign_periods.loc[1, "Date"]- overall_campaign_periods.loc[0, "Date"]).days

print(f"\nOverall Campaign Duration: {overall_campaign_duration} days")
