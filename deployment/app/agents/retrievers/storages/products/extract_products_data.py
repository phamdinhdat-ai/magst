import numpy as np
from typing import List
import pandas as pd
import os
from pathlib import Path


products_data = pd.read_csv('app/agents/retrievers/storages/products/ProductConfig2024.csv')
products_data = products_data.fillna('')

# columns_to_check: Index(['ID', 'Name', 'Type', 'Price', 'Subject', 'Working_time', 'Technology',
#        'Summary', 'Feature', 'Feature_en', 'Product_Index'],
#       dtype='object')
 
def extract_products_data(products_data):
    """
    Extracts product data from the CSV file and returns it as a list of dictionaries.
    
    Returns:
        List[dict]: A list of dictionaries containing product information.
    """
    products_list = []
    content = "Thong tin sản phẩm:\n"
    for _, row in products_data.iterrows():
        
        
        product_info = {
            "Tên sản phẩm": row['Name'],
            "Loại": row['Type'],
            "Giá": row['Price'],
            "Đối tượng": row['Subject'],
            "Thời gian làm việc (ngày)": row['Working_time'],
            "Công nghệ": row['Technology'],
            "Tóm tắt": row['Summary'],
            "Tính năng": row['Feature'],
        }
        info_str = "\n".join([f"{key}: {value}" for key, value in product_info.items() if value])
        content += f"\n{info_str}\n"
        
    return content
        


with open('app/agents/retrievers/storages/products/genstory_products.txt', 'w', encoding='utf-8') as f:
    f.write(extract_products_data(products_data))
    print("Product data extracted and saved to genstory_products.txt")
    print("Product data extraction complete.")