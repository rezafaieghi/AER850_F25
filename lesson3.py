import pandas as pd
data = pd.read_csv("data/housing.csv")


print(data.head())
print(data.columns)
print(data['ocean_proximity'])
data['ocean_proximity'].hist()
data.hist()



from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse_output=False)
enc.fit(data[['ocean_proximity']])
encoded_data = enc.transform(data[['ocean_proximity']])

category_names = enc.get_feature_names_out()

encoded_data_df = pd.DataFrame(encoded_data, columns=category_names)
data = pd.concat([data, encoded_data_df], axis = 1)

data = data.drop(columns = 'ocean_proximity')


data.to_csv("revised_data.csv")