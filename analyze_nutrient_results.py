import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from typing import Dict, List, Tuple
import os

def load_meal_data(csv_path: str = "meal_analysis_results.csv") -> pd.DataFrame:
    """Load the meal analysis results from CSV file."""
    return pd.read_csv(csv_path)

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

def categorize_by_unit(nutrients: Dict[str, Tuple[float, str]]) -> Dict[str, Dict[str, float]]:
    """Group nutrients by their unit for easier comparison."""
    result = {}
    for name, (amount, unit) in nutrients.items():
        if unit not in result:
            result[unit] = {}
        result[unit][name] = amount
    return result

def get_top_nutrients(nutrients: Dict[str, Tuple[float, str]], n: int = 10) -> Dict[str, Tuple[float, str]]:
    """Get the top n nutrients by amount for each unit type."""
    categorized = categorize_by_unit(nutrients)
    top_nutrients = {}
    
    for unit, unit_nutrients in categorized.items():
        # Sort nutrients by amount
        sorted_nutrients = {k: v for k, v in sorted(unit_nutrients.items(), 
                                                    key=lambda item: item[1], 
                                                    reverse=True)}
        # Take top n for this unit
        for name, amount in list(sorted_nutrients.items())[:n]:
            top_nutrients[name] = (amount, unit)
            
    return top_nutrients

def visualize_top_nutrients(nutrients: Dict[str, Tuple[float, str]], title: str):
    """Create bar charts for top nutrients by unit type."""
    categorized = categorize_by_unit(nutrients)
    
    # Create a separate plot for each unit type
    for unit, unit_nutrients in categorized.items():
        # Sort nutrients by amount
        sorted_nutrients = {k: v for k, v in sorted(unit_nutrients.items(), 
                                                    key=lambda item: item[1], 
                                                    reverse=True)}
        
        # Take top 10 or all if less than 10
        top_n = dict(list(sorted_nutrients.items())[:10])
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot horizontal bar chart
        bars = plt.barh(list(top_n.keys()), list(top_n.values()), color='skyblue')
        
        # Add amount labels to the bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', va='center')
        
        plt.xlabel(f'Amount ({unit})')
        plt.title(f'Top Nutrients ({unit}) - {title}')
        plt.tight_layout()
        
        # Save the figure
        safe_title = re.sub(r'[^\w\s-]', '', title.replace(' ', '_'))
        plt.savefig(f'top_nutrients_{safe_title}_{unit}.png')
        plt.close()

def visualize_nutrient_comparison(df: pd.DataFrame, nutrient_name: str):
    """Compare a specific nutrient across all meals."""
    plt.figure(figsize=(12, 6))
    
    # Extract nutrient values for each image
    values = []
    image_names = []
    units = set()
    
    for idx, row in df.iterrows():
        nutrients = parse_micronutrients(row['combined_micronutrients'])
        if nutrient_name in nutrients:
            amount, unit = nutrients[nutrient_name]
            values.append(amount)
            image_names.append(os.path.splitext(row['image_path'])[0])  # Remove file extension
            units.add(unit)
    
    if not values:
        print(f"Nutrient {nutrient_name} not found in any meal")
        return
    
    # Create the bar chart
    unit = next(iter(units)) if units else ""
    bars = plt.bar(image_names, values, color='lightgreen')
    
    # Add amount labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height * 1.01,
                f'{height:.2f}', ha='center')
    
    plt.ylabel(f'Amount ({unit})')
    plt.title(f'{nutrient_name} Content Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    safe_name = re.sub(r'[^\w\s-]', '', nutrient_name.replace(' ', '_'))
    plt.savefig(f'nutrient_comparison_{safe_name}.png')
    plt.close()

def analyze_component_contribution(df: pd.DataFrame, nutrient_name: str):
    """Analyze which meal components contribute most to a specific nutrient."""
    contributions = []
    
    for idx, row in df.iterrows():
        image_name = row['image_path']
        
        # Parse the component micronutrients
        components_str = row['micronutrients_by_component']
        
        # Split by component
        component_sections = components_str.split(' | ')
        
        for section in component_sections:
            # Extract component name and its matched database item
            match = re.match(r'([^(]+)\s*\(([^)]+)\):\s*\[(.*)\]', section)
            if match:
                component_name = match.group(1).strip()
                matched_item = match.group(2).strip()
                micros_str = match.group(3).strip()
                
                # Parse the micronutrients
                if micros_str != "No data":
                    micros = parse_micronutrients(micros_str)
                    if nutrient_name in micros:
                        amount, unit = micros[nutrient_name]
                        contributions.append({
                            'image': image_name,
                            'component': component_name,
                            'matched_item': matched_item,
                            'amount': amount,
                            'unit': unit
                        })
    
    # Create a DataFrame from the contributions
    if contributions:
        contrib_df = pd.DataFrame(contributions)
        print(f"\nTop Contributors to {nutrient_name}:")
        print(contrib_df.sort_values('amount', ascending=False).head(10))
        
        # Visualize the contributions
        plt.figure(figsize=(12, 6))
        
        # Group by component and sum amounts
        grouped = contrib_df.groupby('component')['amount'].sum().sort_values(ascending=False).head(10)
        
        # Plot
        bars = plt.bar(grouped.index, grouped.values, color='salmon')
        
        # Add amount labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height * 1.01,
                    f'{height:.2f}', ha='center')
        
        plt.ylabel(f'Amount ({contrib_df.iloc[0]["unit"] if len(contrib_df) > 0 else ""})')
        plt.title(f'Top Contributors to {nutrient_name}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the figure
        safe_name = re.sub(r'[^\w\s-]', '', nutrient_name.replace(' ', '_'))
        plt.savefig(f'nutrient_contributors_{safe_name}.png')
        plt.close()
    else:
        print(f"No data found for {nutrient_name} contributions")

def visualize_nutrient_coverage(df: pd.DataFrame):
    """Visualize which meals provide the best coverage of essential micronutrients."""
    # List of essential micronutrients to check for
    essential_nutrients = [
        'calcium, ca', 'iron, fe', 'magnesium, mg', 'phosphorus, p', 
        'potassium, k', 'sodium, na', 'zinc, zn', 'copper, cu',
        'manganese, mn', 'selenium, se', 'vitamin c, total ascorbic acid',
        'vitamin b-6', 'vitamin b-12', 'folate, total', 'vitamin a, rae',
        'vitamin e (alpha-tocopherol)', 'vitamin d (d2 + d3)', 'vitamin k (phylloquinone)'
    ]
    
    # Count presence of essential nutrients in each meal
    coverage_data = []
    
    for idx, row in df.iterrows():
        image_name = os.path.splitext(row['image_path'])[0]  # Remove file extension
        nutrients = parse_micronutrients(row['combined_micronutrients'])
        
        # Count how many essential nutrients are present
        count = sum(1 for nutrient in essential_nutrients if nutrient in nutrients)
        percentage = (count / len(essential_nutrients)) * 100
        
        coverage_data.append({
            'image': image_name,
            'count': count,
            'percentage': percentage
        })
    
    # Create visualization
    coverage_df = pd.DataFrame(coverage_data)
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(coverage_df['image'], coverage_df['percentage'], color='lightblue')
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height * 1.01,
                f'{height:.1f}%', ha='center')
    
    plt.ylabel('Coverage (%)')
    plt.title('Essential Micronutrient Coverage by Meal')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.7)  # Add reference line at 50%
    plt.axhline(y=75, color='g', linestyle='--', alpha=0.7)  # Add reference line at 75%
    plt.tight_layout()
    
    plt.savefig('nutrient_coverage.png')
    plt.close()
    
    return coverage_df

def combine_all_nutrients(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Combine micronutrients from all meals by unit type."""
    combined_nutrients = {}  # unit -> {nutrient: total_amount}
    
    for _, row in df.iterrows():
        # Parse the combined micronutrients from this meal
        nutrients = parse_micronutrients(row['combined_micronutrients'])
        
        # Add to the combined totals
        for name, (amount, unit) in nutrients.items():
            if unit not in combined_nutrients:
                combined_nutrients[unit] = {}
            
            if name in combined_nutrients[unit]:
                combined_nutrients[unit][name] += amount
            else:
                combined_nutrients[unit][name] = amount
    
    return combined_nutrients

def visualize_combined_nutrients(combined_nutrients: Dict[str, Dict[str, float]], output_file="combined_micronutrients.png"):
    """Create a single visualization showing the combined micronutrients from all meals."""
    # Create a multi-panel figure - one subplot for each unit type
    units = list(combined_nutrients.keys())
    n_units = len(units)
    
    # Determine the grid layout
    if n_units <= 3:
        n_rows, n_cols = 1, n_units
    else:
        n_rows = (n_units + 1) // 2  # Ceiling division
        n_cols = 2
    
    fig = plt.figure(figsize=(15, n_rows * 5))
    fig.suptitle('Combined Micronutrients Across All Meals', fontsize=16, y=0.98)
    
    # Plot each unit type in a separate subplot
    for i, unit in enumerate(units):
        # Get the nutrients for this unit
        unit_nutrients = combined_nutrients[unit]
        
        # Sort by amount (descending)
        sorted_nutrients = {k: v for k, v in sorted(unit_nutrients.items(), 
                                                   key=lambda item: item[1], 
                                                   reverse=True)}
        
        # Take top 10
        top_nutrients = dict(list(sorted_nutrients.items())[:10])
        
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        # Plot horizontal bar chart
        bars = ax.barh(list(top_nutrients.keys()), list(top_nutrients.values()), color='skyblue')
        
        # Add amount labels to the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', va='center')
        
        ax.set_xlabel(f'Total Amount ({unit})')
        ax.set_title(f'Top Nutrients ({unit})')
        
        # Add a note about the number of nutrients displayed
        total_for_unit = len(unit_nutrients)
        if total_for_unit > 10:
            ax.text(0.5, -0.15, f'Showing top 10 of {total_for_unit} nutrients', 
                   ha='center', transform=ax.transAxes, fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust for the suptitle
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined micronutrient visualization saved to {output_file}")
    return output_file

def visualize_mineral_comparison(df: pd.DataFrame, output_file="mineral_comparison.png"):
    """
    Create a visualization comparing mineral content across all images.
    Shows sodium, potassium, phosphorus, calcium, zinc, and iron levels.
    One subplot per mineral, with bars for each image.
    """
    # Get all available nutrient names for reference
    all_nutrients = set()
    for _, row in df.iterrows():
        nutrients = parse_micronutrients(row['combined_micronutrients'])
        all_nutrients.update(nutrients.keys())
    
    print("\nAvailable nutrient names in the data:")
    print(sorted(all_nutrients))
    
    # List of minerals to compare - with exact names from the data
    mineral_variations = {
        'Sodium': ['na'],
        'Potassium': ['k'],
        'Phosphorus': ['p'],
        'Calcium': ['ca'],
        'Zinc': ['zn'],
        'Iron': ['fe']
    }
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    fig.suptitle('Mineral Content Comparison Across All Meals', fontsize=16, y=0.95)
    
    # Process each mineral in a separate subplot
    for i, (mineral_title, possible_names) in enumerate(mineral_variations.items()):
        # Extract data for this mineral from all images
        values = []
        image_names = []
        units = set()
        used_name = None
        
        for idx, row in df.iterrows():
            image_name = os.path.splitext(row['image_path'])[0]  # Remove file extension
            nutrients = parse_micronutrients(row['combined_micronutrients'])
            
            # Try all possible variations of the mineral name
            found = False
            for name in possible_names:
                if name in nutrients:
                    amount, unit = nutrients[name]
                    values.append(amount)
                    image_names.append(image_name)
                    units.add(unit)
                    used_name = name
                    found = True
                    break
            
            if not found:
                # Add 0 if the mineral is not present
                values.append(0)
                image_names.append(image_name)
        
        # If we didn't find any matches, print a warning
        if not used_name:
            print(f"Warning: No data found for {mineral_title} under names {possible_names}")
                
        # Create the bar chart in this subplot
        ax = axes[i]
        unit = next(iter(units)) if units else "mg"  # Default to mg
        
        bars = ax.bar(image_names, values, color='lightgreen')
        
        # Add amount labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if there's a value
                ax.text(bar.get_x() + bar.get_width()/2, height * 1.01,
                       f'{height:.1f}', ha='center', fontsize=8)
        
        # Set labels and title
        ax.set_ylabel(f'Amount ({unit})')
        ax.set_title(f'{mineral_title} ({used_name if used_name else "No data"})')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Add gridlines
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Adjust for suptitle
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mineral comparison visualization saved to {output_file}")
    return output_file

def main():
    # Load the meal analysis data
    df = load_meal_data()
    print(f"Loaded data for {len(df)} meals")
    
    # Combine all micronutrients from all meals
    print("\nCombining micronutrients from all meals...")
    combined_nutrients = combine_all_nutrients(df)
    
    # Create a single visualization for combined micronutrients
    output_file = visualize_combined_nutrients(combined_nutrients)
    
    # Create a mineral comparison visualization
    print("\nCreating mineral comparison visualization...")
    mineral_output_file = visualize_mineral_comparison(df)
    
    # Print summary of the combined nutrients
    print("\nSummary of combined micronutrients:")
    for unit, nutrients in combined_nutrients.items():
        print(f"\n{unit} unit:")
        # Sort by amount
        sorted_nutrients = {k: v for k, v in sorted(nutrients.items(), 
                                                   key=lambda item: item[1], 
                                                   reverse=True)}
        # Print top 5
        for i, (name, amount) in enumerate(sorted_nutrients.items()):
            if i < 5:
                print(f"  {name}: {amount:.2f} {unit}")
            else:
                break
        print(f"  ... and {len(nutrients) - 5} more nutrients.")
    
    print(f"\nAnalysis complete. Visualizations saved to {output_file} and {mineral_output_file}")

if __name__ == "__main__":
    main() 