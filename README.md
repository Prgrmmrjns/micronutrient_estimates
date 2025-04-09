# Image-Based Meal Micronutrient Analysis

This system analyzes food images to identify meal components and estimate their micronutrient content.

## Features

- Identifies meal components from images using GPT-4 Vision
- Estimates weights of food components in grams
- Calculates micronutrient content using a nutrition database
- Compares micronutrient content across multiple meals
- Exports results to CSV files for further analysis

## Requirements

- Python 3.8+
- OpenAI API key

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install openai pandas scikit-learn numpy matplotlib
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Make sure you have a folder called `images` with food images to analyze

## How to Use

### Analyzing Images

Run the main script to analyze all images in the `images` folder:

```bash
python analyze_meal_micronutrients.py
```

The script will:
1. Identify meal components in each image
2. Look up the micronutrient content for each component
3. Calculate total micronutrients for each meal
4. Compare results across all analyzed meals
5. Save results to CSV files

### Output Files

- `micronutrient_estimates.csv`: Wide-format data with one row per image
- `micronutrient_estimates_readable.csv`: Long-format data with one row per micronutrient-image pair
- `micronutrient_comparison.csv`: Comparison table of all micronutrients across images
- `embedding_cache.pkl`: Cache of text embeddings (for faster subsequent runs)

## Extending the Database

The `nutrients_database.json` file contains micronutrient data for common foods. To add more foods:

1. Open the file in a text editor
2. Add new entries following the existing format:
   ```json
   "food_name": {
     "micronutrients": {
       "vitamin_name": {"amount": 0.0, "unit": "mg"},
       ...
     }
   }
   ```

## Food Component Matching

The system uses several strategies to match detected food components to database entries:
1. Exact matches (e.g., "bread" matches "bread")
2. Normalized matches (removing words like "slice", "piece", etc.)
3. First-word matches (e.g., "wheat bread" could match "bread")
4. Embedding-based similarity search for less exact matches

## Troubleshooting

If you see many "Component X not found in nutrient database" warnings:
1. Check if the component has a similar name in the database
2. Add the component to the database
3. Add a mapping in the `normalize_food_name` function 