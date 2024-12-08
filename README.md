# Predicting Recipe Steps: An Analysis of Food.com Recipes

Predicting Recipe Steps: An Analysis of Food.com Recipes is a data science project conducted at University of Michigan. The project encompasses various stages of analysis, starting from exploratory data analysis to hypothesis testing, creation of baseline models, and concluding with fairness analysis.

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

We will use **Mean Squared Error (MSE)** as our evaluation metric. MSE is a standard metric for regression tasks and has several advantages:

- **Penalizes Larger Errors**: MSE gives higher weight to larger errors, ensuring that large discrepancies between predicted and actual `n_steps` are penalized more.
- **Widespread Use**: MSE is a common benchmark for regression models, making results easy to interpret and compare.
- **Continuous Target**: Since `n_steps` is a continuous target, metrics designed for regression (like MSE) are more appropriate than classification metrics (like accuracy or F1-score).

MSE allows us to quantify how close our predictions are to the actual complexity of recipes. A lower MSE indicates that our model predictions are closely aligned with the true number of steps.

---

With our prediction problem defined, our response variable chosen, and our metric justified, we have a clear path forward. Next, we will build baseline and final models to predict `n_steps` and evaluate how well our model performs in capturing the complexity of recipes.

## Baseline Model
For the baseline model, we created a regression model, with the following two features: `n_ingredients` and `calories`. Both of these features are quantitative.

FIX:
After fitting the model, our accuracy score on the training data is **0.7910**. This means that our model is able to correctly predict **79.10%** of data. This accuracy score may sound really high, but it is quite misleading since our data is unbalanced. The F-1 score of this model is **7.68%** which is extremely low. Such a low F-1 score is due to a low Recall of 0.043, as our model has many false negatives. Our model still has large improvement space, and we will improve it through adding more features, and tuning hyperparameters in the next section. 

## Final Model
In our final model, we added two more features: `monsterkills` and `minionkills`. We are adding these two features into our model because we believe in the LOL game, the champions in jungle positions usually have higher damage, so they are able to kill more minions compared to other positions at the same amount of time. Moreover, the main job of the jungle position in the game is to kill the monsters, so we believe jungle positions should have a relatively high `monsterkills` number. Additionally, `minion kills` reflect a player's ability to efficiently farm gold and experience, which are crucial for scaling and gaining advantages in the game. Therefore, we expect that both `monsterkills` and `minionkills` will provide valuable predictive power in our final model, allowing us to better understand the factors influencing victory in League of Legends matches.

Our final model also uses a Random Forest Classifier in alignment with the baseline model. The two additional features we added (`monsterkills` and `minionkills`) are both quantitative features, so we used StandardScaler Transformer to perform encodings of these two columns. In terms of tuning hyperparameters, the two hyperparameters we chose are: max depth and the number of estimators for the random forest classifier. We are testing max depth of 2 through 200, with each of 20 steps. For the number of estimators, we are testing from 2 to 100, with each of 10 steps. Using the technique of grid search to find the best hyperparameters, we found out that the best max depth is 22 and the best number of estimators is 42. 

The accuracy score is now **0.9993**, meaning our model is able to correctly predict **99.93%** of our data. This score is super high! If we now take a look into the F-1 score, it is 99.84%, meaning both of our precision and recall are close to 1. We have achieved huge improvement in both evaluation metrics, and this improvement suggests that our adjustment to the model is effective in terms of prediction power.
