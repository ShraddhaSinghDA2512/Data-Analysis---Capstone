import pandas as pd
#from sqlalchemy import create_engine

# Load the datasets
orders = pd.read_excel("Orders1.0.xlsx", "scanner_data")
orders = orders.reset_index()  # make sure indexes pair with number of rows
customers = pd.read_excel("Customers.xlsx")
customers = customers.reset_index()  # make sure indexes pair with number of rows
products=pd.read_excel("Orders1.0.xlsx", "product description ")
products = products.reset_index()  # make sure indexes pair with number of rows

#1. Removing Orders with Quantities Less Than 1:

orders = orders.drop(orders[orders.Quantity < 1].index)
        
#2. Removing Orders with Products Having >80% Price Drop:

#Step 1: Calculate Min/Max Price per SKU:
sku_price_range = {}  # Dictionary to store min/max prices
for index, order in orders.iterrows():
    sku = order['SKU']
    price = order['Sales_Amount']
    if sku not in sku_price_range:
        sku_price_range[sku] = {"min": price, "max": price}
    else:
        sku_price_range[sku]["min"] = min(sku_price_range[sku]["min"], price)
        sku_price_range[sku]["max"] = max(sku_price_range[sku]["max"], price)        
            

#Step 2: Identify and Remove Orders with Large Price Drops:
orders_to_remove = set() # Use a set for efficient checking
for index, order in orders.iterrows():
    sku = order['SKU']
    price = order['Sales_Amount']
    if sku in sku_price_range:
        max_price = sku_price_range[sku]["max"]
        if price < max_price * 0.2:  # 80% drop check
            orders_to_remove.add(order['id']) #  Add order ID to removal set

for index, order in orders.iterrows():
    if order['id'] in orders_to_remove:
        orders = orders.drop(orders[orders.id == order['id']].index)

#3. Updating Price Based on Sales Amount and Quantity:
orders['Price'] = orders['Sales_Amount'] / orders['Quantity']

orders = orders.drop(orders[orders.Price < 1].index)

orders.to_excel('Orders2.0.xlsx', index=False)
    

# Create a SQLAlchemy engine
#engine = create_engine('mysql+mysqlconnector://oraclefashion:OraFashion_2024!@localhost/oraclefashion')

#customers.to_sql('customers', con=engine, if_exists='replace', index=False)  # 'replace' creates the table or replaces it if it exists
#orders.to_sql('orders', con=engine, if_exists='replace', index=False)  # 'replace' creates the table or replaces it if it exists
#products.to_sql('products', con=engine, if_exists='replace', index=False)  # 'replace' creates the table or replaces it if it exists

