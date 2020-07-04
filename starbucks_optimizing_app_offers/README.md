# Project: Optimizing App Offers with Starbucks 

*Read more in my [blog post](https://tanyayt.github.io/Optimizing-App-Offers-Starbucks/)*

Starbucks sends their promotional offers to customers via email, social media channels and mobile Apps. It is important to understand customer behaviors in response to these offers to optimize the offer selections. In this project, I took a deep dive into Starbucks App data and built a machine learning model to predict whether a user will complete a promotional offer. 

## Dataset overview

-   The program used to create the data simulates how people make purchasing decisions and how those decisions are influenced by promotional offers.
-   Each person in the simulation has some hidden traits that influence their purchasing patterns and are associated with their observable traits. People produce various events, including receiving offers, opening offers, and making purchases.
-   As a simplification, there are no explicit products to track. Only the amounts of each transaction or offer are recorded.
-   There are three types of offers that can be sent: buy-one-get-one (BOGO), discount, and informational. In a BOGO offer, a user needs to spend a certain amount to get a reward equal to that threshold amount. In a discount, a user gains a reward equal to a fraction of the amount spent. In an informational offer, there is no reward, but neither is there a requisite amount that the user is expected to spend. Offers can be delivered via multiple channels.
-   The basic task is to use the data to identify which groups of people are most responsive to each type of offer, and how best to present each type of offer.

## Data Dictionary

*   `profile.json` 

    Rewards program users (17000 users x 5 fields)
    -   gender: (categorical) M, F, O, or null
    -   age: (numeric) missing value encoded as 118
    -   id: (string/hash)
    -   became_member_on: (date) format YYYYMMDD
    -   income: (numeric)/

*   `porfolio.json`

    Offers sent during 30-day test period (10 offers x 6 fields)

    -   reward: (numeric) money awarded for the amount spent

    -   channels: (list) web, email, mobile, social

    -   difficulty: (numeric) money required to be spent to receive reward

    -   duration: (numeric) time for offer to be open, in days

    -   offer_type: (string) bogo, discount, informational

    -   id: (string/hash)

### `transcript.json` 

Event log (306648 events x 4 fields)

-   person: (string/hash)
-   event: (string) offer received, offer viewed, transaction, offer completed
-   value: (dictionary) different values depending on event type
    -   offer id: (string/hash) not associated with any "transaction"
    -   amount: (numeric) money spent in "transaction"
    -   reward: (numeric) money gained from "offer completed"
-   time: (numeric) hours after start of test

## File Organization

*   `/data` folder has all the raw data files
*   `Starbucks_Capstone_notebook.ipynb`: contains codes and output of exploration, cleaning, processing, and modelling process of the analysis. 
*   `/image` folder contains all image files, generated from `Starbucks_Capstone_notebook.ipynb` 
*   `/model` contains trained model Pickle files. These files can be reproduced, using `Starbucks_Capstone_notebook.ipynb` 

# Software Requirement and Packages 

The project is run on Python with the following packages imported: 

```python
import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats

%matplotlib inline 
#there must be no space between % and matplotlib to make it work 

# multi-line output 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# hide warnings - comment to disable
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle
```



## Acknowledgement 

I would like to thank Udacity's Data Scientist Nanodegree program for providing guidance in this project, and Starbucks for providing the data and business context. 