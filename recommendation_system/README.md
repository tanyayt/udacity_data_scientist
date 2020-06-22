# Recommendation System 

## Motivation

The project analyzes the interactions that users have with articles on the [IBM Watson Studio platform](https://dataplatform.cloud.ibm.com/), and offer recommendations to them about new articles they may like. 

## Tasks

I. **Exploratory Data Analysis**

Findings: 

*   

II. **Rank Based Recommendations**

To get started in building recommendations, 

*   find the most popular articles simply based on the most interactions. Since there are no ratings for any of the articles, it is easy to assume the articles with the most interactions are the most popular. These are then the articles we might recommend to new users (or anyone depending on what we know about them).

III. **User-User Based Collaborative Filtering**

In order to build better recommendations for the users of IBM's platform, we could look at users that are similar in terms of the items they have interacted with. These items could then be recommended to the similar users. This would be a step in the right direction towards more personal recommendations for the users. You will implement this next.

V. **Matrix Factorization**

Finally, you will complete a machine learning approach to building recommendations. Using the user-item interactions, you will build out a matrix decomposition. Using your decomposition, you will get an idea of how well you can predict new articles an individual might interact with (spoiler alert - it isn't great). You will finally discuss which methods you might use moving forward, and how you might test how well your recommendations are working for engaging users.

## Files 

### Testing Files

To make sure intermediate outputs are correct. These files are downloaded from Udacity. 

*   `project_tests_py`

*   `top_5.p`

*   `top_10.p`

*   `to_20.p`

*   `user_item_matrix`

### Data (`/data`)

*   `articles_comunity.csv`

*   `user-item-interaction.csv`

### Notebook 

*   `Recommendations_with_IBM.ipynb` 

## Acknowledgement 

This project is part of [Udacity](www.udacity.com)'s Data Scientist Nanodegree program. The notebook template is provided by Udacity. 