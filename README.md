# Predicting Recipe Steps: An Analysis of Food.com Recipes

Predicting Recipe Steps: An Analysis of Food.com Recipes is a data science project conducted at University of Michigan. The project encompasses various stages of analysis, starting from exploratory data analysis to hypothesis testing, creation of baseline models, and concluding with fairness analysis. The primary focus of the project is to investigate the significance of the "first blood" event in League of Legends matches and its impact on match statistics and outcomes.

Authors: Naman Jain, Anay Moitra

## Introduction
### General Introduction
The Food.com Recipes and Ratings dataset contains over 80,000 recipes and 700,000 ratings collected from the Food.com website since 2008. Each recipe includes details such as the preparation time, ingredients, steps, nutritional information, tags, and user ratings. This rich dataset provides an excellent opportunity to explore various factors that influence recipe complexity and user preferences.

In the culinary world, understanding what makes a recipe complex or straightforward can significantly impact home cooks, chefs, and recipe developers. The number of steps in a recipe often correlates with the recipe's complexity, preparation time, and required skill level.

The central question we aim to answer is: **What factors influence the number of steps in a recipe?**. 

By analyzing various features such as preparation time, nutritional content, and tags associated with recipes, we hope to uncover patterns that can help predict recipe complexity. This analysis could assist home cooks in selecting recipes that match their skill level and available time, and help recipe developers create recipes that cater to their target audience.

### Introduction of Columns
The cleaned dataset consists of 83,782 recipes with the following relevant columns:

- `name`: Name of the recipe.

- `id`: Unique identifier for each recipe.

- `minutes`: Total preparation time in minutes.

- `tags`: List of tags associated with the recipe.

- `nutrition`: Nutritional information in the form of a list containing calories and % daily values. 

- `n_steps`: Number of steps in the recipe.

- `description`: User-provided description of the recipe.
  
- `num_tags`: Number of tags associated with the recipe (engineered feature).

- `calories`: Calories per serving.

- `total_fat_PDV, sugar_PDV, sodium_PDV, protein_PDV, saturated_fat_PDV, carbohydrates_PDV`: Nutritional information as percentage of daily values (engineered features).
  
- `average_rating`: Average user rating of the recipe.

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning

To prepare the dataset for analysis, several data cleaning steps were performed:

  Parsing the nutrition Column:
  - The nutrition column contained strings representing lists.
  - Extracted individual nutritional components into separate columns: `calories`, `total_fat_PDV`, `sugar_PDV`, `sodium_PDV`, `protein_PDV`, `saturated_fat_PDV`, `carbohydrates_PDV`.

  Handling the tags Column:
  - Converted the string representations of lists into actual lists.
  - Created a new feature `num_tags` representing the number of tags associated with each recipe.

  Merging Average Ratings:
  - Loaded the interactions dataset and calculated the average rating for each recipe.
  - Merged the `average ratings` back into the recipes dataset to create the average_rating column.

  Handling Missing Values:
  - Identified that `average_rating` has missing values where recipes have no ratings.
  - Decided *not* to impute missing `average_rating` values to avoid introducing bias, as missing ratings may indicate new or less popular recipes.

  Ensuring Correct Data Types:
  - Converted the `submitted` column to datetime format.
  - Ensured that `description` and `name` are strings.

  Removing Duplicates:
  - Checked for duplicate recipe IDs and removed them to ensure data integrity.
