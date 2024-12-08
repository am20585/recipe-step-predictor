# Predicting Recipe Steps: An Analysis of Food.com Recipes

Predicting Recipe Steps: An Analysis of Food.com Recipes is a data science project conducted at University of Michigan. The project encompasses various stages of analysis, starting from exploratory data analysis to creating various models.

Authors: Naman Jain, Anay Moitra

## Introduction
### General Introduction
The Food.com Recipes and Ratings dataset contains over 80,000 recipes and 700,000 ratings collected from the Food.com website since 2008. Each recipe includes details such as the preparation time, ingredients, steps, nutritional information, tags, and user ratings. This rich dataset provides an excellent opportunity to explore various factors that influence recipe complexity and user preferences.

In the culinary world, understanding what makes a recipe complex or straightforward can significantly impact home cooks, chefs, and recipe developers. The number of steps in a recipe often correlates with the recipe's complexity, preparation time, and required skill level.

The central question we aim to answer is: **What factors influence the number of steps in a recipe?**. 

By analyzing various features such as ingredients, nutritional content, and tags associated with recipes, we hope to find patterns that can help predict recipe complexity. This analysis could assist home cooks in selecting recipes that match their skill level and available time, and help recipe developers create recipes that cater to their target audience.

### Introduction of Columns
The cleaned dataset consists of 83,782 rows of recipes with the following relevant columns:

- `name`: Name of the recipe.

- `id`: Unique identifier for each recipe.

- `minutes`: Total preparation time in minutes.

- `tags`: List of tags associated with the recipe.

- `nutrition`: Nutritional information in the form of a list containing calories and % daily values. 

- `n_steps`: Number of steps in the recipe.

- `calories`: Calories per serving.

- `total_fat_PDV, sugar_PDV, sodium_PDV, protein_PDV, saturated_fat_PDV, carbohydrates_PDV`: Nutritional information as percentage of daily values (engineered features).

- `ingredients`: List of all ingredients in the recipe
 
- `n_ingredients`: Number of ingredients in the recipe

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning

To prepare the dataset for analysis, a few data cleaning steps were performed:

  Parsing the `nutrition` Column:
  - The `nutrition` column contained strings representing lists.
  - Extracted individual nutritional components into separate columns: `calories`, `total_fat_PDV`, `sugar_PDV`, `sodium_PDV`, `protein_PDV`, `saturated_fat_PDV`, `carbohydrates_PDV`.

  By extracting individual components from the `nutrition` column, these variables allowed us to include specific nutritional factors as predictors in our models. For example, understanding the relationship between `protein_PDV` and `n_steps` helped identify how recipe complexity correlates with nutrition. Without this step, the nutrition data would have been unusable due to its string-based format.
  
  Handling the `tags` Column:
  - Converted the string representations of lists into actual lists.
  - Flattened all tags to identify unique tags across recipes.
  - Created a sorted list of unique tags to prepare for later one-hot encoding.

  Converting the `tags` column into lists and identifying unique tags helped us to engineer categorical features that describe the nature of recipes. For example, one-hot encoding tags like `easy` and `dessert` can provide valuable information for predicting `n_steps`. This step has qualitative aspects of the recipes, which was important in exploring patterns such as whether simpler tags are associated with fewer steps.

  Merging Average Ratings:
  - Loaded the `interactions` dataset and calculated the average rating for each recipe.
  - Merged the `average ratings` back into the `recipes` dataset to create the `average_rating` column.

  Handling Missing Values:
  - Identified that `average_rating` has missing values where recipes have no ratings.
  - Decided *not* to impute missing `average_rating` values to avoid introducing bias, as missing ratings may indicate new or less popular recipes.

  Adding `average_rating` provided an important metric for evaluating the popularity or perceived quality of recipes. Although it had missing values for unrated recipes, leaving these values as `NaN` avoided introducing bias by imputing artificial ratings. This decision preserved the integrity of the analysis and made sure that if we had done predictions involving `average_rating`, they would not skewed by incorrect assumptions about missing values.

This is what our cleaned data looks like:
| name                                 |     id |   minutes | tags                                                                                                                                                                                                                                                                                               |   n_steps | ingredients                                                                                                                                                                                                                             |   n_ingredients |   calories |   total_fat_PDV |   sugar_PDV |   sodium_PDV |   protein_PDV |   saturated_fat_PDV |   carbohydrates_PDV |
|:-------------------------------------|-------:|----------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|-----------:|----------------:|------------:|-------------:|--------------:|--------------------:|--------------------:|
| 1 brownies in the world    best ever | 333281 |        40 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings']                                                                        |        10 | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa powder', 'vanilla extract', 'brewed espresso', 'kosher salt', 'all-purpose flour']                                                          |               9 |      138.4 |              10 |          50 |            3 |             3 |                  19 |                   6 |
| 1 in canada chocolate chip cookies   | 453467 |        45 | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                                                                                                      |        12 | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour', 'baking soda', 'chocolate chips']                                                                             |              11 |      595.1 |              46 |         211 |           22 |            13 |                  51 |                  26 |
| 412 broccoli casserole               | 306168 |        40 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                                                                                               |         6 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']                                                                   |               9 |      194.8 |              20 |           6 |           32 |            22 |                  36 |                   3 |
| millionaire pound cake               | 286009 |       120 | ['time-to-make', 'course', 'cuisine', 'preparation', 'occasion', 'north-american', 'desserts', 'american', 'southern-united-states', 'dinner-party', 'holiday-event', 'cakes', 'dietary', 'christmas', 'thanksgiving', 'low-sodium', 'low-in-something', 'taste-mood', 'sweet', '4-hours-or-less'] |         7 | ['butter', 'sugar', 'eggs', 'all-purpose flour', 'whole milk', 'pure vanilla extract', 'almond extract']                                                                                                                                |               7 |      878.3 |              63 |         326 |           13 |            20 |                 123 |                  39 |
| 2000 meatloaf                        | 475785 |        90 | ['time-to-make', 'course', 'main-ingredient', 'preparation', 'main-dish', 'potatoes', 'vegetables', '4-hours-or-less', 'meatloaf', 'simply-potatoes2']                                                                                                                                             |        17 | ['meatloaf mixture', 'unsmoked bacon', 'goat cheese', 'unsalted butter', 'eggs', 'baby spinach', 'yellow onion', 'red bell pepper', 'simply potatoes shredded hash browns', 'fresh garlic', 'kosher salt', 'white pepper', 'olive oil'] |              13 |      267   |              30 |          12 |           12 |            29 |                  48 |                   2 |

### Univariate Analysis
We start with performing univariate analysis to examine the distribution of single variables. We looked at the distribution of steps in recipes which is show below:

<iframe src="assets/fig_steps.html" width="800" height="600" frameborder="0" ></iframe>

The plot shows that most recipes have a relatively low number of steps, with a sharp decline as the number of steps increases. This suggests that simpler recipes (with fewer steps) are far more common, and complex recipes with many steps are rare. This observation also highlights the prevalence of simple recipes and indicates that factors like ingredient count may play a significant role in determining the complexity of a recipe.

### Bivariate Analysis
We then performed bivariate analysis to examine the relationship of different variables. We looked at the relationship between the number of steps and the number of ingredients in recipes which is show below:

<iframe src="assets/fig_steps_ingredients.html" width="800" height="600" frameborder="0" ></iframe>

The trend shows that as the number of ingredients increases, the number of steps tends to spread out, with some recipes having significantly more steps. However, the relationship is not strictly linear, which suggests that factors other than ingredient count, such as preparation complexity or cooking techniques, may also influence the number of steps in a recipe. This supports our investigation into what drives recipe complexity.

### Interesting Aggregates

We also created a pivot table to examine the relationship between the number of ingredients and the average number of steps in recipes. This analysis helps identify how the recipe complexity changes with varying ingredient counts. The first few rows of the pivot table are shown below:

Pivot Table: n_ingredients and n_steps
|   n_ingredients |   n_steps |
|----------------:|----------:|
|               1 |   7.57143 |
|               2 |   5.93173 |
|               3 |   5.61315 |
|               4 |   6.32002 |
|               5 |   7.12584 |

This pivot table shows that recipes with fewer ingredients tend to have fewer steps on average, but the relationship is not strictly linear. For example, recipes with one ingredient have a higher average number of steps than those with two or three ingredients. This could indicate that recipes with very few ingredients might involve more complex cooking techniques, while those with moderate ingredient counts might be simpler to prepare.

### Imputation

We also saw that some of our columns had missing values. These columns were `name`, `description`, and `average rating`. We decided to **not** impute these values as none of these columns were relavent to us and imputing or not imputing values would have **no** effect on our analysis.

## Framing a Prediction Problem

With a solid understanding of our dataset and initial exploratory analyses, we now defined a clear prediction problem that aligns with our overall theme of understanding recipe complexity.

### Prediction Problem Statement

We aim to predict the **number of steps** in a recipe (**`n_steps`**) using features that are available before the recipe steps are fully finalized. By estimating the complexity of a recipe from its attributes, we can help home cooks choose recipes appropriate for their time constraints and skill levels, and assist recipe creators in refining their dishes for targeted audiences. Since our response variable, `n_steps`, is a continuous numerical value, this is a **regression problem**, since we are predicting a numerical outcome.

We chose `n_steps` because it serves as a direct measure of recipe complexity. Understanding what influences complexity can reveal insights into how preparation time, nutritional factors, and the diversity of tags (dietary restrictions, meal types, cuisines) shape the effort required to prepare a dish.

### Features and Time of Prediction Justification

Here are some of the features we will use to predict `n_steps`:

- **`n_ingredients`**: Number of ingredients associated with the recipe. Ingredients are chosen at the time of recipe creation and categorization, so this information is known upfront.

- **Nutritional Information**:  
  - `calories`
  - `total_fat_PDV`
  - `sugar_PDV`
  - `sodium_PDV`
  - `protein_PDV`
  - `saturated_fat_PDV`
  - `carbohydrates_PDV`. Nutritional values can be estimated from the ingredients and serving sizes before writing out the complete steps. Since recipe developers know what ingredients they plan to use, they can approximate nutritional content early in the process.
 
- **`tags`**: Tags associated with the recipe. Here, we assume that tags are predefined as part of the recipe metadata and are generated by the user when they submit the recipe or are assigned automatically based on recipe characteristics. So the tags of a certain recipe is known.
 
By relying solely on information available prior to detailing the step-by-step instructions, we made that our prediction does not leak information from the future. Basically, we are not using features that depend on knowing the final number of steps or user feedback that would only be available after the recipe is published.

### Evaluation Metric
We will use both **Mean Absolute Error** and **R-squared (R²)** as our evaluation metric.

R² is a standard metric for regression tasks and has several advantages. One of them is R² indicates the proportion of the variance in the target variable that is explained by the model, which provides a clear understanding of the model's explanatory power. R² values range from 0 to 1 (or can be negative for poor models), which also makes it easy to interpret how well the model fits the data. A higher R² means the model's predictions are more aligned with the actual values. MAE is a metric used to measure the average magnitude of errors between predicted values and actual values and also has many advantages in this scenario. One of them is MAE is easy to understand because it's the average of absolute differences between predicted and actual values. This gives a direct, intuitive sense of how far off the model’s predictions are, on average. 

Together, R² will help us understand how well the model captures the overall variance in the data, while MAE will provide a sense of the average error magnitude, which will make sure both model fit and prediction accuracy are considered.

With our prediction problem defined, our response variable chosen, and our metric justified, we have a clear path forward. Next, we will build baseline and final models to predict `n_steps` and evaluate how well our model performs in capturing the complexity of recipes.

## Baseline Model
Our baseline model is a **linear regression** model aimed at predicting the number of steps (`n_steps`). We used two quantitative features: `total_fat_PDV` and `protein_PDV`. These features were chosen based on the assumption that recipes with higher fat and protein content might be more complex and require a greater number of steps.

After fitting the model, the R² value on the test set was 0.027, which indicates that only 2.7% of the variance in `n_steps` was explained by these features. Although this is quite low, the Mean Absolute Error (MAE) was 4.5363, which suggests the model’s predictions were off by an average of about 4.5 steps. This result shows that the selected features provide only a weak relationship with recipe complexity but still offer a rough approximation of the number of steps.

While the baseline model’s performance leaves a lot of room for improvement, it establishes a foundation for development. For our final model, we will explore adding more features and applying hyperparameter tuning to enhance predictive performance and better capture the complexity of recipes.

## Final Model
In our final model, we added two more features: `n_ingredients` and `calories`. We believe these two features have a high impact on the number of steps as when increasing the number of ingredients of a dish, we expect utilizing all those ingredients would typically increase the number of steps. We also added calories for a similar reason as using total fat and protein.

Our final model also uses a Linear Regression in alignment with the baseline model. The two additional features we added (`n_ingredients` and `calories`) are both quantitative features, so we used StandardScaler Transformer to perform encodings of these two columns. We utilized GridSearchCV for hyperparameter turing as well.

Our new Mean Absolute Error is down to **3.984** showing our estimations are only about 4 steps off on average. Our R-squared value is now **0.2193**, meaning our model has improved quite a lot in this aspect as well. We have achieved some improvement in both of our evaluation metrics, suggesting that we made an effective adjustment in our final model.
