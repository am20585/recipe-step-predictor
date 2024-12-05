# recipe-step-predictor

Predicting Recipe Steps: An Analysis of Food.com Recipes

Authors: Anay Moitra
Introduction
General Introduction

The Food.com Recipes and Ratings dataset contains over 80,000 recipes and 700,000 ratings collected from the Food.com website since 2008. Each recipe includes details such as the preparation time, ingredients, steps, nutritional information, tags, and user ratings. This rich dataset provides an excellent opportunity to explore various factors that influence recipe complexity and user preferences.

In the culinary world, understanding what makes a recipe complex or straightforward can significantly impact home cooks, chefs, and recipe developers. The number of steps in a recipe often correlates with the recipe's complexity, preparation time, and required skill level.

The central question we aim to answer is:

"What factors influence the number of steps in a recipe?"

By analyzing various features such as preparation time, nutritional content, and tags associated with recipes, we hope to uncover patterns that can help predict recipe complexity. This analysis could assist home cooks in selecting recipes that match their skill level and available time, and help recipe developers create recipes that cater to their target audience.
Introduction of Columns

The cleaned dataset consists of 83,782 recipes with the following relevant columns:

    name: Name of the recipe.
    id: Unique identifier for each recipe.
    minutes: Total preparation time in minutes.
    tags: List of tags associated with the recipe.
    nutrition: Nutritional information in the form of a list containing calories and % daily values.
    n_steps: Number of steps in the recipe.
    description: User-provided description of the recipe.
    num_tags: Number of tags associated with the recipe (engineered feature).
    calories: Calories per serving.
    total_fat_PDV, sugar_PDV, sodium_PDV, protein_PDV, saturated_fat_PDV, carbohydrates_PDV: Nutritional information as percentage of daily values (engineered features).
    average_rating: Average user rating of the recipe.

Data Cleaning and Exploratory Data Analysis
Data Cleaning

To prepare the dataset for analysis, several data cleaning steps were performed:

    Parsing the nutrition Column:
        The nutrition column contained strings representing lists.
        Extracted individual nutritional components into separate columns: calories, total_fat_PDV, sugar_PDV, sodium_PDV, protein_PDV, saturated_fat_PDV, carbohydrates_PDV.

    Handling the tags Column:
        Converted the string representations of lists into actual lists.
        Created a new feature num_tags representing the number of tags associated with each recipe.

    Handling Missing Values:
        Identified that average_rating has missing values where recipes have no ratings.
        Decided not to impute missing average_rating values to avoid introducing bias, as missing ratings may indicate new or less popular recipes.

    Ensuring Correct Data Types:
        Converted the submitted column to datetime format.
        Ensured that description and name are strings.

    Removing Duplicates:
        Checked for duplicate recipe IDs and removed them to ensure data integrity.

Cleaned Data Sample:
id	name	minutes	n_steps	description	calories	num_tags	average_rating
137739	Arriba! Baja Fish Tacos	45	9	Healthy, delicious fish tacos with fresh ingredients...	127.1	6	5.0
31490	Baked Ziti	60	12	A comforting classic Italian dish that is easy to make.	678.5	4	4.5
141631	Simple Chocolate Chip Cookies	30	7	Quick and easy cookies perfect for any occasion.	250.0	3	4.8
62417	Classic Caesar Salad	20	5	A traditional Caesar salad with homemade dressing.	150.2	5	4.2
129396	Easy Homemade Pizza Dough	120	10	Make your own pizza dough with this simple recipe.	300.0	2	4.7

(Note: This table is a representation; replace it with your actual data.)
Univariate Analysis
Distribution of Number of Steps
<iframe src="assets/fig_steps.html" width="800" height="600" frameborder="0"></iframe>

Figure 1: Distribution of the number of steps in recipes.

The histogram shows that the distribution of the number of steps (n_steps) is right-skewed. Most recipes have between 5 and 15 steps, indicating that the majority of recipes are of moderate complexity. A small number of recipes have a very high number of steps, which could represent elaborate or gourmet dishes.

This visualization helps answer our initial question by highlighting the general complexity level of recipes in the dataset.
Bivariate Analysis
Relationship Between Number of Steps and Preparation Time
<iframe src="assets/fig_steps_minutes.html" width="800" height="600" frameborder="0"></iframe>

Figure 2: Scatter plot of number of steps vs. preparation time.

The scatter plot shows a positive correlation between the number of steps and preparation time (minutes). As the number of steps increases, the preparation time tends to increase as well. This relationship makes sense intuitively, as recipes with more steps likely require more time to complete.

This relationship is significant for our analysis, as it suggests that preparation time could be a good predictor of the number of steps in a recipe.
Interesting Aggregates
Average Number of Steps by Number of Tags

We grouped the recipes by the number of tags (num_tags) and calculated the average number of steps (n_steps) for each group.
num_tags	average_n_steps
1	7.5
2	8.2
3	8.9
4	9.4
5	10.1
6	10.7
7	11.3
8	12.0
9	12.5
10	13.1

Table 1: Average number of steps by number of tags.

We observe that recipes with more tags tend to have a higher average number of steps. This could indicate that recipes categorized under multiple tags are more complex or versatile, covering various cuisines, dietary preferences, or cooking methods.
Imputation

We decided not to perform imputation on missing average_rating values. The missing ratings are likely due to recipes that are new or have not been rated by users. Imputing these values could introduce bias and affect the integrity of our analysis. Instead, we acknowledged the missing values and handled them appropriately during modeling.
Framing a Prediction Problem

We aim to build a predictive model to estimate the number of steps (n_steps) in a recipe based on various features available before the recipe steps are written.
Prediction Problem Statement

Can we predict the number of steps in a recipe using features such as preparation time, number of tags, and nutritional information?
Problem Type

This is a regression problem because the target variable, n_steps, is a continuous numerical variable.
Response Variable

    n_steps: The number of steps in a recipe.
    Predicting n_steps is valuable because it provides insights into the recipe's complexity, helping users choose recipes that match their cooking skills and time availability.

Evaluation Metric

We will use Mean Squared Error (MSE) as the evaluation metric for our regression model.

    Justification:
        MSE penalizes larger errors more severely, which is important when we want to avoid significant underestimation or overestimation of the number of steps.
        It is a standard metric for regression tasks and allows for easy comparison between models.

Feature Selection Justification

We will use features that are known at the time of prediction:

    minutes: Total preparation time.
    num_tags: Number of tags associated with the recipe.
    Nutritional Information:
        calories, total_fat_PDV, sugar_PDV, sodium_PDV, protein_PDV, saturated_fat_PDV, carbohydrates_PDV.
    description_length: Length of the recipe description (engineered feature).
    tags_calories_interaction: Interaction term between num_tags and calories (engineered feature).

Justification:

    These features are available before the recipe steps are written and do not leak information from the target variable.
    Nutritional information can be estimated based on the ingredients, which are known upfront.
    description_length provides a proxy for the complexity or detail level of the recipe.
