# Predicting Recipe Steps: An Analysis of Food.com Recipes

Predicting Recipe Steps: An Analysis of Food.com Recipes is a data science project conducted at University of Michigan. The project encompasses various stages of analysis, starting from exploratory data analysis to creating various models.
Authors: Naman Jain, Anay Moitra

## Introduction
### General Introduction
The Food.com Recipes and Ratings dataset contains over 80,000 recipes and 700,000 ratings collected from the Food.com website since 2008. Each recipe includes details such as the preparation time, ingredients, steps, nutritional information, tags, and user ratings. This rich dataset provides an excellent opportunity to explore various factors that influence recipe complexity and user preferences.

In the culinary world, understanding what makes a recipe complex or straightforward can significantly impact home cooks, chefs, and recipe developers. The number of steps in a recipe often correlates with the recipe's complexity, preparation time, and required skill level.

The central question we aim to answer is: **What factors influence the number of steps in a recipe?**. 

By analyzing various features such as preparation time, nutritional content, and tags associated with recipes, we hope to uncover patterns that can help predict recipe complexity. This analysis could assist home cooks in selecting recipes that match their skill level and available time, and help recipe developers create recipes that cater to their target audience.

### Introduction of Columns
The cleaned dataset consists of 83,782 rows of recipes with the following relevant columns:

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

- `ingredients`: List of all ingredients in the recipe
 
- `n_ingredients`: Number of ingredients in the recipe

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

This is what our cleaned data looks like:
| name                                 |     id |   minutes | tags                                                                                                                                                                                                                                                                                               |   n_steps | ingredients                                                                                                                                                                                                                             |   n_ingredients |   calories |   total_fat_PDV |   sugar_PDV |   sodium_PDV |   protein_PDV |   saturated_fat_PDV |   carbohydrates_PDV |
|:-------------------------------------|-------:|----------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|-----------:|----------------:|------------:|-------------:|--------------:|--------------------:|--------------------:|
| 1 brownies in the world    best ever | 333281 |        40 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings']                                                                        |        10 | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa powder', 'vanilla extract', 'brewed espresso', 'kosher salt', 'all-purpose flour']                                                          |               9 |      138.4 |              10 |          50 |            3 |             3 |                  19 |                   6 |
| 1 in canada chocolate chip cookies   | 453467 |        45 | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                                                                                                      |        12 | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour', 'baking soda', 'chocolate chips']                                                                             |              11 |      595.1 |              46 |         211 |           22 |            13 |                  51 |                  26 |
| 412 broccoli casserole               | 306168 |        40 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                                                                                               |         6 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']                                                                   |               9 |      194.8 |              20 |           6 |           32 |            22 |                  36 |                   3 |
| millionaire pound cake               | 286009 |       120 | ['time-to-make', 'course', 'cuisine', 'preparation', 'occasion', 'north-american', 'desserts', 'american', 'southern-united-states', 'dinner-party', 'holiday-event', 'cakes', 'dietary', 'christmas', 'thanksgiving', 'low-sodium', 'low-in-something', 'taste-mood', 'sweet', '4-hours-or-less'] |         7 | ['butter', 'sugar', 'eggs', 'all-purpose flour', 'whole milk', 'pure vanilla extract', 'almond extract']                                                                                                                                |               7 |      878.3 |              63 |         326 |           13 |            20 |                 123 |                  39 |
| 2000 meatloaf                        | 475785 |        90 | ['time-to-make', 'course', 'main-ingredient', 'preparation', 'main-dish', 'potatoes', 'vegetables', '4-hours-or-less', 'meatloaf', 'simply-potatoes2']                                                                                                                                             |        17 | ['meatloaf mixture', 'unsmoked bacon', 'goat cheese', 'unsalted butter', 'eggs', 'baby spinach', 'yellow onion', 'red bell pepper', 'simply potatoes shredded hash browns', 'fresh garlic', 'kosher salt', 'white pepper', 'olive oil'] |              13 |      267   |              30 |          12 |           12 |            29 |                  48 |                   2 |

### Univariate Analysis
Data of the number of steps.
<iframe src="assets/fig_steps.html" width="800" height="600" frameborder="0" ></iframe>

### Bivariate Analysis

<iframe src="assets/fig_steps_ingredients.html" width="800" height="600" frameborder="0" ></iframe>

### Interesting Aggregates

We created a pivot table to ...

Pivot Table: n_ingredients and n_steps
|   n_ingredients |   n_steps |
|----------------:|----------:|
|               1 |   7.57143 |
|               2 |   5.93173 |
|               3 |   5.61315 |
|               4 |   6.32002 |
|               5 |   7.12584 |

## Framing a Prediction Problem

With a solid understanding of our dataset and initial exploratory analyses, we now define a clear prediction problem that aligns with our overall theme of understanding recipe complexity.

### Prediction Problem Statement

We aim to predict the **number of steps** in a recipe (**`n_steps`**) using features that are available before the recipe steps are fully finalized. By estimating the complexity of a recipe from its attributes, we can help home cooks choose recipes appropriate for their time constraints and skill levels, and assist recipe creators in refining their dishes for targeted audiences.

### Problem Type

Since our target variable, `n_steps`, is a continuous numerical value, this is a **regression problem**. We are not classifying recipes into discrete categories; rather, we are predicting a numerical outcome (the number of steps).

### Response Variable

- **Response Variable**: `n_steps` (Number of steps in a recipe)

We chose `n_steps` because it serves as a direct measure of recipe complexity. Understanding what influences complexity can reveal insights into how preparation time, nutritional factors, and the diversity of tags (e.g., dietary restrictions, meal types, cuisines) shape the effort required to complete a dish.

### Features and Time of Prediction Justification

We will use the following features to predict `n_steps`:

- **`minutes`**: Total preparation time.  
  *Justification*: Estimated preparation time can be determined from the ingredients, planned cooking methods, and basic recipe structure before finalizing the detailed steps.

- **`num_tags`**: Number of tags associated with the recipe.  
  *Justification*: Tags are assigned at the time of recipe creation and categorization, so this information is known upfront.

- **Nutritional Information**:  
  - `calories`
  - `total_fat_PDV`
  - `sugar_PDV`
  - `sodium_PDV`
  - `protein_PDV`
  - `saturated_fat_PDV`
  - `carbohydrates_PDV`  
  *Justification*: Nutritional values can be estimated from the ingredients and serving sizes before writing out the complete steps. Since recipe developers know what ingredients they plan to use, they can approximate nutritional content early in the process.

By relying solely on information available prior to detailing the step-by-step instructions, we ensure that our prediction does not leak information from the future. In other words, we are not using features that depend on knowing the final number of steps or user feedback that would only be available after the recipe is published.

### Evaluation Metric
We will use both **Mean Absolute Error** and **R-squared (R²)** as our evaluation metric. 
R² is a standard metric for regression tasks and has several advantages:
- **Measures Explained Variance**: R² indicates the proportion of the variance in the target variable (the number of steps) that is explained by the model, providing a clear understanding of the model's explanatory power. 
- **Interpretability**: R² values range from 0 to 1 (or can be negative for poor models), making it easy to interpret how well the model fits the data. A higher R² means the model's predictions are more aligned with the actual values.
- 
R² allows us to assess how well our model is capturing the relationship between the number of ingredients, calories, nutrition, and the number of steps in a recipe. A higher R² indicates that the model is effectively explaining the variability in the number of steps.
MAE offers key advantages for our dataset as well
- **Interpretability**: MAE is easy to understand because it's the average of absolute differences between predicted and actual values. This gives a direct, intuitive sense of how far off the model’s predictions are, on average.
Together, R² will help us understand how well the model captures the overall variance in the data, while MAE will provide a sense of the average error magnitude, ensuring both model fit and prediction accuracy are considered.

---

With our prediction problem defined, our response variable chosen, and our metric justified, we have a clear path forward. Next, we will build baseline and final models to predict `n_steps` and evaluate how well our model performs in capturing the complexity of recipes.

## Baseline Model
For the baseline model, we created a regression model, with the following two features: `total_fat_PDV` and `protein_PDV`. Both of these features are quantitative and we chose these two features initially as we predict meals with high proportions of these nutritional values may be more complicated and have a high number of steps.

After fitting the model, r-squared value is **0.1**. This is quite a low r-squared value showing that these two variables are quite poor predictors of number of steps. However, our Mean Absolute Error is **4.5363** showing our model estimated steps with only about a 4.5 step error. Our model still has large improvement space, and we will improve it through adding more features, and tuning hyperparameters in the next section. 

## Final Model
In our final model, we added two more features: `n_ingredients` and `calories`. We believe these two features have a high impact on the number of steps as when increasing the number of ingredients of a dish, we expect utilizing all those ingredients would typically increase the number of steps. We also added calories for a similar reason as using total fat and protein.

Our final model also uses a Linear Regression in alignment with the baseline model. The two additional features we added (`n_ingredients` and `calories`) are both quantitative features, so we used StandardScaler Transformer to perform encodings of these two columns. We utilized GridSearchCV for hyperparameter turing as well.

Our new Mean Absolute Error is down to **3.984** showing our estimations are only about 4 steps off on average. Our R-squared value is now **0.2193**, meaning our model has improved quite a lot in this aspect as well. We have achieved some improvement in both of our evaluation metrics, suggesting that we made an effective adjustment in our final model.
