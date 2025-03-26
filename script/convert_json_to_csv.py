import json
import csv
import os

def convert_json_to_csv(json_file_path, csv_file_path):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    # Check if data is not empty
    if not data:
        print(f"No data found in {json_file_path}")
        return
    
    # Extract fieldnames from the first row
    fieldnames = data[0].keys()
    
    # Write to CSV file
    with open(csv_file_path, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data rows
        writer.writerows(data)
    
    print(f"Successfully converted {json_file_path} to {csv_file_path}")

if __name__ == "__main__":
    # Define file paths
    json_file_path = "data/faqs.json"
    csv_file_path = "data/faqs.csv"
    
    # Call the function
    convert_json_to_csv(json_file_path, csv_file_path)
