import csv
import json
import os

def convert_csv_to_json(csv_file_path, json_file_path):
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    
    # Read the CSV file
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        # Create a CSV reader
        csv_reader = csv.DictReader(csv_file)
        
        # Convert CSV data to list of dictionaries
        data = [row for row in csv_reader]
    
    # Write to JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"Successfully converted {csv_file_path} to {json_file_path}")

if __name__ == "__main__":
    csv_file_path = "data/faqs.csv"
    json_file_path = "data/faqs.json"
    
    convert_csv_to_json(csv_file_path, json_file_path)
