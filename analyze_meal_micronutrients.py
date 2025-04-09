import base64
import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re

load_dotenv()

# Define data models
class MealComponent(BaseModel):
    component_name: str
    component_amount: int
    explanation: str

class MealComponentList(BaseModel):
    components: List[MealComponent]

class MicronutrientInfo(BaseModel):
    name: str
    amount: float
    unit: str

# Global embedding cache
embedding_cache = {}

def save_embedding_cache(cache_path: str = "embedding_cache.pkl"):
    """Save the embedding cache to disk."""
    with open(cache_path, 'wb') as f:
        pickle.dump(embedding_cache, f)
    print(f"Embedding cache saved to {cache_path}")

def load_embedding_cache(cache_path: str = "embedding_cache.pkl") -> Dict[str, List[float]]:
    """Load embedding cache from disk."""
    global embedding_cache
    try:
        with open(cache_path, 'rb') as f:
            embedding_cache = pickle.load(f)
        print(f"Loaded {len(embedding_cache)} embeddings from cache file")
        return embedding_cache
    except (FileNotFoundError, pickle.PickleError):
        print("No embedding cache found or error loading cache. Starting with empty cache.")
        return {}

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def load_nutrient_database(database_path: str = "simplified_nutrients.json") -> Dict[str, Any]:
    """Load the nutrient database from the simplified JSON file."""
    try:
        with open(database_path, 'r') as f:
            processed_db = json.load(f)
        
        print(f"Loaded {len(processed_db)} food items from {database_path}")
        return processed_db
            
    except FileNotFoundError:
        print(f"Warning: Nutrient database file '{database_path}' not found.")
        print("Trying to process the original USDA database...")
        
        # Try to run the preprocessing script if the simplified database doesn't exist
        try:
            from preprocess_nutrient_database import preprocess_usda_database
            if preprocess_usda_database():
                # Try loading again
                with open(database_path, 'r') as f:
                    processed_db = json.load(f)
                print(f"Loaded {len(processed_db)} food items from {database_path}")
                return processed_db
            else:
                print("Failed to process the original database.")
                return {}
        except ImportError:
            print("Could not import preprocessing module.")
            return {}
    except json.JSONDecodeError:
        print(f"Error: Could not parse '{database_path}' as JSON.")
        return {}

def get_embedding(text: str, model: str = "text-embedding-3-small", client=None) -> List[float]:
    """Get embedding for a text string using OpenAI's API with caching."""
    # Normalize the text
    text = text.lower().strip()
    
    # Check if the embedding is in the cache
    cache_key = f"{text}_{model}"
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    # If not in cache, get from API
    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.embeddings.create(
        model=model,
        input=text,
        encoding_format="float"
    )
    
    embedding = response.data[0].embedding
    
    # Cache the result
    embedding_cache[cache_key] = embedding
    
    return embedding

def normalize_food_name(name: str) -> str:
    """Normalize food name for better matching with USDA database."""
    # Convert to lowercase
    name = name.lower().strip()
    
    # Remove common words that don't affect the core meaning
    words_to_remove = ['slice', 'slices', 'piece', 'pieces', 'roll', 'rolls', 'cup', 'of']
    pattern = r'\b(?:' + '|'.join(words_to_remove) + r')\b'
    name = re.sub(pattern, '', name).strip()
    
    # Common food mappings for better USDA matching
    replacements = {
        'bread roll': 'bread',
        'cheese slice': 'cheese',
        'ham slice': 'ham',
        'cherry tomato': 'tomatoes, cherry',
        'boiled egg': 'egg',
        'brown bread': 'bread, whole wheat',
        'white bread': 'bread, white',
        'butter': 'butter, salted'
    }
    
    for old, new in replacements.items():
        if name == old:
            return new
    
    return name

def find_most_similar_component(component_name: str, nutrient_db: Dict[str, Any], client=None, threshold: float = 0.50) -> Tuple[str, float]:
    """Find the most similar component in the USDA nutrient database using embeddings."""
    # Try with normalized name first
    normalized_name = normalize_food_name(component_name)
    print(f"Normalized '{component_name}' to '{normalized_name}'")
    
    # Check for exact match
    if normalized_name in nutrient_db:
        print(f"Found exact match for normalized name '{normalized_name}'")
        return normalized_name, 1.0
    
    # For USDA database, we need to search more flexibly
    # Check if any database entry contains the normalized name as a substring
    substring_matches = []
    for food_name in nutrient_db.keys():
        if normalized_name in food_name:
            similarity = len(normalized_name) / len(food_name)  # Simple similarity based on length ratio
            substring_matches.append((food_name, similarity))
    
    # If we found substring matches, return the best one
    if substring_matches:
        substring_matches.sort(key=lambda x: x[1], reverse=True)
        best_match = substring_matches[0]
        print(f"Found substring match: '{best_match[0]}' (similarity: {best_match[1]:.2f}) for '{normalized_name}'")
        return best_match
    
    # If no exact match, use embeddings to find the closest match
    component_embedding = get_embedding(normalized_name, client=client)
    
    # Get embeddings for all keys in the database (limiting to 100 to avoid excessive API calls)
    db_keys = list(nutrient_db.keys())[:100]  # Take first 100 foods for efficiency
    
    # Calculate similarities
    similarities = []
    for key in db_keys:
        key_embedding = get_embedding(key, client=client)
        similarity = cosine_similarity([component_embedding], [key_embedding])[0][0]
        similarities.append((key, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Print top matches for debugging
    print(f"Top 3 matches for '{normalized_name}':")
    for i, (key, sim) in enumerate(similarities[:3]):
        print(f"  {i+1}. '{key}' (similarity: {sim:.2f})")
    
    # Return most similar if above threshold
    if similarities and similarities[0][1] >= threshold:
        return similarities[0]
    
    return None, 0.0

def get_micronutrients(component_name: str, amount: int, nutrient_db: Dict[str, Any], client=None) -> Tuple[str, Dict[str, MicronutrientInfo]]:
    """Look up micronutrients for a given food component."""
    # Use embedding search to find the most similar component
    similar_component, similarity = find_most_similar_component(component_name, nutrient_db, client)
    
    if similar_component is None:
        print(f"Warning: Component '{component_name}' not found in nutrient database")
        return None, {}
    
    if similarity < 1.0:  # Not an exact match
        print(f"Found similar component: '{similar_component}' (similarity: {similarity:.2f}) for '{component_name}'")
    
    # Get nutrient data per 100g
    nutrient_data = nutrient_db[similar_component]
    
    # Scale nutrients based on actual amount
    scale_factor = amount / 100.0
    
    micronutrients = {}
    for nutrient_name, data in nutrient_data.get("micronutrients", {}).items():
        # Scale the amount based on the weight of the component
        scaled_amount = data["amount"] * scale_factor
        
        micronutrients[nutrient_name] = MicronutrientInfo(
            name=nutrient_name,
            amount=scaled_amount,
            unit=data["unit"]
        )
    
    return similar_component, micronutrients

def analyze_meal_components(meal_components: List[MealComponent], nutrient_db: Dict[str, Any], client=None) -> Tuple[Dict[str, Dict[str, MicronutrientInfo]], Dict[str, MicronutrientInfo]]:
    """
    Analyze micronutrient content for all components in a meal.
    
    Returns:
        Tuple containing:
        1. Dictionary mapping component names to their micronutrients
        2. Dictionary with combined micronutrients across all components
    """
    # Dictionary to store micronutrients per component
    component_micronutrients = {}
    
    # Dictionary to store total micronutrients
    total_micronutrients = {}
    
    # Create OpenAI client once for all components if not provided
    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Process each component
    for component in meal_components:
        print(f"Analyzing component: {component.component_name} ({component.component_amount}g)")
        matched_name, component_micros = get_micronutrients(
            component.component_name, 
            component.component_amount, 
            nutrient_db, 
            client
        )
        
        # Store the component's micronutrients with the matched database name
        if matched_name:
            component_key = f"{component.component_name} ({matched_name})"
        else:
            component_key = component.component_name
            
        component_micronutrients[component_key] = component_micros
        
        # Add each micronutrient to the totals
        for micro_name, micro_info in component_micros.items():
            if micro_name in total_micronutrients:
                # Add to existing amount
                total_micronutrients[micro_name].amount += micro_info.amount
            else:
                # Create new entry
                total_micronutrients[micro_name] = micro_info
    
    return component_micronutrients, total_micronutrients

def analyze_image(image_path: str, nutrient_db: Dict[str, Any], client=None) -> Tuple[List[MealComponent], Dict[str, Dict[str, MicronutrientInfo]], Dict[str, MicronutrientInfo]]:
    """
    Analyze a meal image for components and micronutrients.
    
    Returns:
        Tuple containing:
        1. List of identified meal components
        2. Dictionary mapping component names to their micronutrients
        3. Dictionary with combined micronutrients across all components
    """
    # Initialize OpenAI client if not provided
    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Prepare the instruction for GPT-4 Vision
    instruction = """Please list all the components of the meal in the image.
    Please list every component in its most basic form. For example, if there is a bread with butter, list the bread and butter separately.
    Give the amount in grams.
    Do not output any other text than the JSON response.
    Format your response in the following JSON format:
    {"components": [{"component_name": "", "component_amount": 0, "explanation": ""}]}"""
    
    # Encode the image
    base64_image = encode_image(image_path)
    
    # Get meal components from the image
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        response_format=MealComponentList,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
    )
    
    # Convert response to MealComponentList
    estimates = json.loads(completion.choices[0].message.content)
    components = [MealComponent(**comp) for comp in estimates["components"]]
    
    print("\nIdentified meal components:")
    for comp in components:
        print(f"- {comp.component_name}: {comp.component_amount}g ({comp.explanation})")
    
    # Analyze micronutrients
    print("\nAnalyzing micronutrients...")
    component_micronutrients, total_micronutrients = analyze_meal_components(components, nutrient_db, client)
    
    return components, component_micronutrients, total_micronutrients

def create_comprehensive_csv(analysis_results, output_file="meal_analysis_results.csv"):
    """
    Create a simplified CSV file with all analysis results in 4 columns.
    
    Args:
        analysis_results: Dictionary mapping image names to their analysis results
        output_file: Path to save the CSV file
    """
    print(f"\nCreating simplified CSV report at {output_file}...")
    
    # Define the 4 columns
    columns = ['image_path', 'meal_components', 'combined_micronutrients', 'micronutrients_by_component']
    
    # Prepare data for the CSV
    rows = []
    
    # Process each image's results
    for img_name, (components, component_micros, total_micros) in analysis_results.items():
        # Format meal components as a string list
        components_str = ", ".join([f"{c.component_name} ({c.component_amount}g)" for c in components])
        
        # Format combined micronutrients as a string list
        combined_micros_str = ", ".join([f"{name}: {info.amount:.2f} {info.unit}" for name, info in sorted(total_micros.items())])
        
        # Format micronutrients by component as a nested string list
        by_component_list = []
        for comp_name, micros in component_micros.items():
            if micros:
                micros_str = ", ".join([f"{name}: {info.amount:.2f} {info.unit}" for name, info in sorted(micros.items())])
                by_component_list.append(f"{comp_name}: [{micros_str}]")
            else:
                by_component_list.append(f"{comp_name}: [No data]")
        
        by_component_str = " | ".join(by_component_list)
        
        # Add row to data
        rows.append({
            'image_path': img_name,
            'meal_components': components_str,
            'combined_micronutrients': combined_micros_str,
            'micronutrients_by_component': by_component_str
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"Analysis results saved to {output_file}")
    
    return df

def main():
    # Load embedding cache if available
    load_embedding_cache()
    
    # Check for images directory
    if not os.path.exists('images'):
        print("Error: 'images' directory not found.")
        return
    
    # Get all images
    images = [f for f in os.listdir('images') if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print("No images found in the 'images' directory.")
        return
    
    print(f"Found {len(images)} images to process.")
    
    # Load the nutrient database once
    nutrient_db = load_nutrient_database()
    if not nutrient_db:
        print("Error: Could not load nutrient database.")
        return
    
    # Create OpenAI client once for all images
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Store results for all images
    analysis_results = {}
    
    # Process each image
    for img in images:
        image_path = os.path.join('images', img)
        print(f"\n\nProcessing image: {img}")
        
        # Analyze the image (reusing the database and client)
        components, component_micros, total_micros = analyze_image(image_path, nutrient_db, client)
        
        # Store results
        analysis_results[img] = (components, component_micros, total_micros)
        
        # Display total micronutrient results
        print("\nMicronutrient Analysis Results:")
        print("-" * 50)
        print(f"{'Micronutrient':<20} {'Amount':>10} {'Unit':<5}")
        print("-" * 50)
        for name, info in sorted(total_micros.items()):
            print(f"{name:<20} {info.amount:>10.2f} {info.unit:<5}")
    
    # Create comprehensive CSV with all results
    if analysis_results:
        df = create_comprehensive_csv(analysis_results)
        
        # Save embedding cache for future runs
        save_embedding_cache()
    else:
        print("\nNo results to save to CSV.")

if __name__ == "__main__":
    main() 