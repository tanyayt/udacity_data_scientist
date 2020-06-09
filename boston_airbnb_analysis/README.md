# Motivation

Airbnb has transformed the way that people travel since 2008. Proper pricing can be a challenge to many hosts, who are not full-time landlords. This project illustrates how prices vary in different neighborhood, and how availability changes throughout the year. This will help hosts to make better-informed decisions. 

# Libraries Used

The folllowing Python libraries are used: 

`pandas`

`numpy`

`matplotlib`

`seaborn`

`sklearn`

`statsmodels`

# Files 

Data Files: [Download](https://www.kaggle.com/airbnb/boston/download) 

*   `calendar.csv` includes listing id, price and availability of each date
*   `listing.csv` includes full descriptions and average review scores 
*   `reviews.csv` includes unique id for each reviewer and comments

Analysis: 

*   `boston_airbnb_analysis.ipynb` Jupyter Notebook that documents all the steps involved to load, clean, transform, and analyze data
*   All png files are generated from `boston_airbnb_analysis.ipynb` 

# Summary of Results 

The results suggest in Boston, Airbnb rentals are most available in early December, and least available in mid-September; in terms of days of week, Airbnb rentals are most available on Mondays and Tuesdays, and least available on Fridays and Saturdays. Rental rates vary by neighborhood. South Boston, downtown and south ends are the most expensive neighbourhood of entire apartment rental, private room rental, and shared room rental respectively. 

Our initial attempt to build a prediction model is satisfactory. Using room type, neighbourhood, bedrooms, bathrooms, beds, and review ratings as predictors, we are able to capture about 60% of the variance in the data. 

View more details in this [blog post](https://tanyayt.github.io/Boston-airbnb-market/)

# Acknowledgements

The dataset is provided by [Airbnb Inside](http://insideairbnb.com/get-the-data.html) and some visualization ideas are inspired by the original source [here](http://insideairbnb.com/boston/)