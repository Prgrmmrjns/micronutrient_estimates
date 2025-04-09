import json
import os
import sys

def preprocess_usda_database(input_file="foundationDownload.json", output_file="simplified_nutrients.json"):
    """
    Preprocess the USDA food database to extract only the essential information for micronutrient analysis.
    
    Args:
        input_file: Path to the original USDA JSON file
        output_file: Path to save the simplified JSON file
    """
    print(f"Processing USDA database from '{input_file}'...")
    
    try:
        # Load the original database
        with open(input_file, 'r') as f:
            raw_data = json.load(f)
        
        # Extract the foods array
        foods = raw_data.get("FoundationFoods", [])
        print(f"Found {len(foods)} food items in the original database")
        
        # Create simplified database
        simplified_db = {}
        
        # Set of excluded nutrient categories (macronutrients, etc.)
        excluded_nutrient_keywords = [
            "energy", "protein", "carbohydrate", "lipid", "water", "ash", "nitrogen", 
            "fatty acid", "sfa", "pufa", "mufa", "tfa", "starch", "sugar"
        ]
        
        # Process each food item
        for food_item in foods:
            if not isinstance(food_item, dict) or "description" not in food_item:
                continue
                
            # Get food name (lowercase for better matching)
            food_name = food_item["description"].lower()
            
            # Skip if no nutrient data
            if "foodNutrients" not in food_item:
                continue
                
            # Process micronutrients
            micronutrients = {}
            for nutrient_entry in food_item["foodNutrients"]:
                if "nutrient" not in nutrient_entry or "amount" not in nutrient_entry:
                    continue
                    
                nutrient = nutrient_entry["nutrient"]
                if "name" not in nutrient or "unitName" not in nutrient:
                    continue
                    
                # Get nutrient name and check if it's a micronutrient
                nutrient_name = nutrient["name"].lower()
                
                # Skip excluded nutrients
                if any(keyword in nutrient_name.lower() for keyword in excluded_nutrient_keywords):
                    continue
                
                # Get amount and unit
                amount = nutrient_entry.get("amount", 0)
                unit = nutrient["unitName"].lower()
                
                # Store only if amount > 0
                if amount > 0:
                    micronutrients[nutrient_name] = {
                        "amount": amount,
                        "unit": unit
                    }
            
            # Only add foods with at least one micronutrient
            if micronutrients:
                simplified_db[food_name] = {
                    "micronutrients": micronutrients
                }
        
        # Save the simplified database
        with open(output_file, 'w') as f:
            json.dump(simplified_db, f, indent=2)
        
        print(f"Processed database successfully!")
        print(f"Original database: {len(foods)} food items")
        print(f"Simplified database: {len(simplified_db)} food items")
        print(f"Saved to: {output_file}")
        
        # Calculate file size reduction
        original_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
        new_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        reduction = (1 - (new_size / original_size)) * 100
        
        print(f"File size reduced from {original_size:.2f} MB to {new_size:.2f} MB ({reduction:.2f}% reduction)")
        
        return True
    
    except Exception as e:
        print(f"Error processing database: {str(e)}")
        return False

if __name__ == "__main__":
    # Use command line arguments if provided
    input_file = sys.argv[1] if len(sys.argv) > 1 else "foundationDownload.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "simplified_nutrients.json"
    
    preprocess_usda_database(input_file, output_file) 