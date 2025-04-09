import pandas as pd
import re
import os
from typing import Dict, List, Tuple

def parse_micronutrients(micros_str: str) -> Dict[str, Tuple[float, str]]:
    """Parse micronutrient string into a dictionary of {name: (amount, unit)}."""
    result = {}
    # Extract key-value pairs like "calcium, ca: 439.80 mg"
    pattern = r'([^,]+):\s*([\d\.]+)\s*([^\s,]+)'
    for match in re.finditer(pattern, micros_str):
        name = match.group(1).strip()
        amount = float(match.group(2))
        unit = match.group(3).strip()
        result[name] = (amount, unit)
    return result

def parse_components(components_str: str) -> List[Tuple[str, int]]:
    """Parse component string into a list of (name, weight) tuples."""
    components = []
    # Example: "bread (200g), ham (150g), green grapes (50g), cherry tomatoes (30g)"
    pattern = r'([^,()]+)\s*\((\d+)g\)'
    for match in re.finditer(pattern, components_str):
        name = match.group(1).strip()
        weight = int(match.group(2))
        components.append((name, weight))
    return components

def create_excel_friendly_csv(input_csv: str = "meal_analysis_results.csv", 
                             output_excel: str = "meal_analysis.xlsx"):
    """Convert the complex CSV into Excel-friendly format with multiple sheets."""
    # Read the original CSV
    df = pd.read_csv(input_csv)
    print(f"Read {len(df)} rows from {input_csv}")
    
    # Create wide-format DataFrame for combined nutrients
    wide_nutrients_data = []
    
    # Process each meal
    for idx, row in df.iterrows():
        image_name = row['image_path']
        
        # Parse components
        components = parse_components(row['meal_components'])
        component_names = [name for name, _ in components]
        component_weights = [weight for _, weight in components]
        
        # Create base row for this meal
        meal_row = {
            'image_name': image_name,
            'total_components': len(components),
            'components_list': ', '.join(component_names),
            'components_weights': ', '.join([f"{w}g" for w in component_weights])
        }
        
        # Parse combined micronutrients and add to the row
        nutrients = parse_micronutrients(row['combined_micronutrients'])
        for nutrient_name, (amount, unit) in nutrients.items():
            column_name = f"{nutrient_name} ({unit})"
            meal_row[column_name] = amount
        
        wide_nutrients_data.append(meal_row)
    
    # Create the wide-format DataFrame
    wide_nutrients_df = pd.DataFrame(wide_nutrients_data)
    
    # Create component-level DataFrames
    component_detail_data = []
    
    for idx, row in df.iterrows():
        image_name = row['image_path']
        
        # Process component nutrients
        components_str = row['micronutrients_by_component']
        component_sections = components_str.split(' | ')
        
        for section in component_sections:
            # Extract component name and its matched database item
            match = re.match(r'([^(]+)\s*\(([^)]+)\):\s*\[(.*)\]', section)
            if match:
                component_name = match.group(1).strip()
                matched_item = match.group(2).strip()
                micros_str = match.group(3).strip()
                
                if micros_str != "No data":
                    # Create base row for this component
                    component_row = {
                        'image_name': image_name,
                        'component_name': component_name,
                        'matched_database_item': matched_item
                    }
                    
                    # Parse the component's micronutrients
                    comp_nutrients = parse_micronutrients(micros_str)
                    for nutrient_name, (amount, unit) in comp_nutrients.items():
                        column_name = f"{nutrient_name} ({unit})"
                        component_row[column_name] = amount
                    
                    component_detail_data.append(component_row)
                else:
                    # Add a row with just the basic info for components with no data
                    component_detail_data.append({
                        'image_name': image_name,
                        'component_name': component_name,
                        'matched_database_item': matched_item,
                        'note': 'No nutrient data available'
                    })
    
    # Create the component details DataFrame
    component_detail_df = pd.DataFrame(component_detail_data)
    
    # Create a simplified meal summary DataFrame
    meal_summary = []
    for idx, row in df.iterrows():
        image_name = row['image_path']
        components = parse_components(row['meal_components'])
        
        # Count the total weight
        total_weight = sum(weight for _, weight in components)
        
        # Get the top 5 nutrients by amount
        nutrients = parse_micronutrients(row['combined_micronutrients'])
        
        # Group nutrients by unit
        nutrients_by_unit = {}
        for name, (amount, unit) in nutrients.items():
            if unit not in nutrients_by_unit:
                nutrients_by_unit[unit] = []
            nutrients_by_unit[unit].append((name, amount))
        
        # Get top 5 nutrients for each unit
        top_nutrients = []
        for unit, nutrient_list in nutrients_by_unit.items():
            sorted_nutrients = sorted(nutrient_list, key=lambda x: x[1], reverse=True)
            for name, amount in sorted_nutrients[:5]:
                top_nutrients.append(f"{name}: {amount:.2f} {unit}")
        
        meal_summary.append({
            'image_name': image_name,
            'total_components': len(components),
            'components_list': ', '.join([f"{name} ({weight}g)" for name, weight in components]),
            'total_weight': f"{total_weight}g",
            'top_nutrients': '; '.join(top_nutrients[:10])  # Limit to top 10 across all units
        })
    
    # Create the meal summary DataFrame
    meal_summary_df = pd.DataFrame(meal_summary)
    
    # 1. Create a pivot table for nutrients across all meals
    # Extract all nutrients with their units
    all_nutrients = set()
    for idx, row in df.iterrows():
        nutrients = parse_micronutrients(row['combined_micronutrients'])
        for name, (_, unit) in nutrients.items():
            all_nutrients.add((name, unit))
    
    # Create the pivot data
    pivot_data = []
    for nutrient_name, unit in sorted(all_nutrients):
        nutrient_row = {'nutrient': nutrient_name, 'unit': unit}
        
        for idx, row in df.iterrows():
            image_name = os.path.splitext(row['image_path'])[0]  # Remove file extension
            nutrients = parse_micronutrients(row['combined_micronutrients'])
            
            if nutrient_name in nutrients:
                amount, _ = nutrients[nutrient_name]
                nutrient_row[image_name] = amount
            else:
                nutrient_row[image_name] = 0
        
        pivot_data.append(nutrient_row)
    
    # Create the pivot DataFrame
    pivot_df = pd.DataFrame(pivot_data)
    
    # Create Excel writer
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        meal_summary_df.to_excel(writer, sheet_name='Meal Summary', index=False)
        wide_nutrients_df.to_excel(writer, sheet_name='Meal Details', index=False)
        pivot_df.to_excel(writer, sheet_name='Nutrient Comparison', index=False)
        component_detail_df.to_excel(writer, sheet_name='Component Details', index=False)
    
    print(f"Excel file created: {output_excel}")
    print(f"Contains {len(meal_summary_df)} meals summary rows")
    print(f"Nutrient comparison table has {len(pivot_df)} rows")
    print(f"Component details has {len(component_detail_df)} rows")
    
    # Also create a CSV version of the meal summary for maximum compatibility
    meal_summary_df.to_csv("meal_summary.csv", index=False)
    print("Simple meal summary CSV created: meal_summary.csv")

if __name__ == "__main__":
    create_excel_friendly_csv() 