# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # read in file
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #keep category_ids as a seperate dataframe 
    categories_ids=categories['id']
    
    # clean data - split category to multiple columns
    categories = categories['categories'].str.split(";",expand=True)
    
    #use the non-numeric portion as the column names
    row = categories.iloc[0]
    category_colnames =row.apply(lambda x:x[:-2]) 
    categories.columns = category_colnames
    
    # convert each column to numeric 
    for column in categories.columns:
           categories[column] = categories[column].apply(lambda x:x.split('-')[1])
   
    categories[column] = pd.to_numeric(categories[column])
    
    # concatenate categories_ids back to the numeric columns
    categories_transformed = pd.concat([categories_ids,categories],axis=1)
    
    # join messages and categories_transformed on id
    df=messages.merge(categories_transformed,on="id")

    # load to database
    engine = create_engine('sqlite:///disaster_response.db')
    df.to_sql('messages', engine, index=False)

    # define features and label arrays
    X=None
    y=None

    return X, y


def build_model():
    # text processing and model pipeline


    # define parameters for GridSearchCV


    # create gridsearch object and return as final model pipeline

    model_pipeline=None
    return model_pipeline


def train(X, y, model):
    # train test split


    # fit model


    # output model test results

    model=None
    return model


def export_model(model):
    # Export model as a pickle file
    pass 


def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline