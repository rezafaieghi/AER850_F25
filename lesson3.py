# all the imports
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#read data
data = pd.read_csv("data/housing.csv")

#some quick look at the data
#print(data.head())
#print(data.columns)
#print(data['ocean_proximity'])
#data['ocean_proximity'].hist()
#data.hist()


#one-hot encoding step for categorical data
enc = OneHotEncoder(sparse_output=False)
enc.fit(data[['ocean_proximity']])
encoded_data = enc.transform(data[['ocean_proximity']])
category_names = enc.get_feature_names_out()
encoded_data_df = pd.DataFrame(encoded_data, columns=category_names)
data = pd.concat([data, encoded_data_df], axis = 1)
data = data.drop(columns = 'ocean_proximity')
#data.to_csv("revised_data.csv")



import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
"""Data Splitting"""
#this is the very basic method of data splitting
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size = 0.2,
#                                                     random_state = 42)

#the use of stratified sampling is strongly recommended
data["income_categories"] = pd.cut(data["median_income"],
                          bins=[0, 2, 4, 6, np.inf],
                          labels=[1, 2, 3, 4])
my_splitter = StratifiedShuffleSplit(n_splits = 1,
                               test_size = 0.2,
                               random_state = 42)
for train_index, test_index in my_splitter.split(data, data["income_categories"]):
    strat_data_train = data.loc[train_index].reset_index(drop=True)
    strat_data_test = data.loc[test_index].reset_index(drop=True)
strat_data_train = strat_data_train.drop(columns=["income_categories"], axis = 1)
strat_data_test = strat_data_test.drop(columns=["income_categories"], axis = 1)


print(data.shape)
print(strat_data_train.shape)
print(strat_data_test.shape)