import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
transactions = pd.read_excel("Orders2.0.xlsx")
customers = pd.read_excel("Customers.xlsx")

# Merge the datasets on 'Customer_ID'
transactions = pd.merge(transactions, customers, on='Customer_ID', how='left')

# Convert 'Date' column to datetime objects if needed
transactions['Date'] = pd.to_datetime(transactions['Date'])


transactions['Season'] = transactions['Date'].dt.quarter.map({
    1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall' })


# Group by season and country to calculate total sales
seasonal_sales_by_country = transactions.groupby(['Season', 'GEOGRAPHY'])['Sales_Amount'].sum().reset_index()

# Plot sales per season for each country
for country in seasonal_sales_by_country['GEOGRAPHY'].unique():
    country_data = seasonal_sales_by_country[seasonal_sales_by_country['GEOGRAPHY'] == country]
    plt.plot(country_data['Season'], country_data['Sales_Amount'], label=country)


plt.xlabel("Season")
plt.ylabel("Sales")
plt.title("Seasonal Sales Trends by Country")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Plots/seasonal_sales_by_country.jpeg", dpi=300)  # Save as JPEG
plt.show()
