# Read data
import pandas as pd
data = pd.read_csv("data/housing.csv")

# Some quick look at the data
# print(data.head())
# print(data.columns)
# print(data['ocean_proximity'])
# data['ocean_proximity'].hist()
# data.hist()

# Remove missing data
data = data.dropna().reset_index(drop=True)



#One-hot encoding step for categorical data
from sklearn.preprocessing import OneHotEncoder
enc  = OneHotEncoder(sparse_output=False)
enc.fit(data[['ocean_proximity']])
encoded_data = enc.transform(data[['ocean_proximity']])
category_names = enc.get_feature_names_out()
encoded_data_df = pd.DataFrame(encoded_data, columns=category_names)
data = pd.concat([data, encoded_data_df], axis = 1)
data = data.drop(columns = 'ocean_proximity')
# data.to_csv("revised_data.csv", index=False)


# Data splitting
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
# This is the very basic method of data splitting
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size = 0.2,
#                                                     random_state = 42)

# The use of stratified sampling is strongly recommended
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



# Variable selection
# Identify y and X. Remember the goal is to find f(.) such that y=f(X)
y_train = strat_data_train['median_house_value']
X_train = strat_data_train.drop(columns=['median_house_value'])
y_test = strat_data_test['median_house_value']
X_test = strat_data_test.drop(columns=['median_house_value'])

# Looking at the colinearity of variables
corr_matrix = strat_data_train.corr()

# Plot the correlationtrix
import seaborn as sns
sns.heatmap(np.abs(corr_matrix))

# Mask correlation values above a threshold
masked_corr_matrix = np.abs(corr_matrix) < 0.8
sns.heatmap(masked_corr_matrix)

# Drom correlation matrix, we identify colinear variables, and select one from them
# Usually, we keep the variable with the highest correlation with y, but this
# does not generate the best results all the time. So, trial and error is needed.
print(np.abs(y_train.corr(X_train['longitude'])))
print(np.abs(y_train.corr(X_train['latitude'])))
print(np.abs(y_train.corr(X_train['total_rooms'])))
print(np.abs(y_train.corr(X_train['total_bedrooms'])))
print(np.abs(y_train.corr(X_train['population'])))
print(np.abs(y_train.corr(X_train['households'])))

# Based on correlation values, we drop the following from X_train
X_train = X_train.drop(columns=['longitude'])
X_train = X_train.drop(columns=['total_bedrooms'])
X_train = X_train.drop(columns=['population'])
X_train = X_train.drop(columns=['households'])

# We can also drop the above columns from X_test. This is safe to do so, because
# we decided the columns to drop based on train data only.
X_test = X_test.drop(columns=['longitude'])
X_test = X_test.drop(columns=['total_bedrooms'])
X_test = X_test.drop(columns=['population'])
X_test = X_test.drop(columns=['households'])


# # Data scaling
# # NOTE: If we want to K-fold cross validation, we should not be scale the data here.
# # Applying standard scaler. Note that we generally apply this to numerical data, 
# # however, applying it to one-hot encoded data does not break anything since they
# # are only 0s and 1s.
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# sc.fit(X_train)
# X_train = sc.transform(X_train)

# # At this point, we can also scale X_test. This is safe to do so, because the 
# # the standard scaler above has been fit to the train data only.
# X_test = sc.transform(X_test)



# Training the first model, a linear regression model.
from sklearn.linear_model import LinearRegression
# mdl1 = LinearRegression()
# mdl1.fit(X_train, y_train)

# Print predictions for the first few data points.
# y_pred_train1 = mdl1.predict(X_train)
# for i in range(5):
#     print("Predictions:", y_pred_train1[i], "Actual values:", y_train[i])

# Using metrics to evaluate the model.
from sklearn.metrics import mean_absolute_error
# mae_train1 = mean_absolute_error(y_pred_train1, y_train)
# print("Model 1 training MAE is: ", round(mae_train1,2))


# Training the second model, a random forest model.
from sklearn.ensemble import RandomForestRegressor
# mdl2 = RandomForestRegressor(n_estimators=30, random_state=42)
# mdl2.fit(X_train, y_train)
# y_pred_train2 = mdl2.predict(X_train)

# # MAE for the second model
# mae_train2 = mean_absolute_error(y_pred_train2, y_train)
# print("Model 2 training MAE is: ", round(mae_train2,2))

# for i in range(5):
#     print("Mode 1 Predictions:",
#           round(y_pred_train1[i],2),
#           "Mode 2 Predictions:",
#           round(y_pred_train2[i],2),
#           "Actual values:",
#           round(y_train[i],2))

# We better use k-fold cross validation for more robust evaluation.
from sklearn.model_selection import cross_val_score
# cv_scores_model1 = cross_val_score(mdl1, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
# cv_mae1 = -cv_scores_model1.mean()
# print("Model 1 Mean Absolute Error (CV):", round(cv_mae1, 2))
# # Note that there is data leak in this implementation, because the standard scaler
# # is fit on the whole training data. We should use pipelines instead.

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipeline1 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())])
cv_scores1 = cross_val_score(pipeline1, X_train, y_train,
                            cv=5, scoring='neg_mean_absolute_error')
cv_mae1 = -cv_scores1.mean()
print("Model 1 CV MAE:", round(cv_mae1, 2))


pipeline1.fit(X_train, y_train)
y_pred_test1 = pipeline1.predict(X_test)
mae_test1 = mean_absolute_error(y_test, y_pred_test1)
print("Model 1 Test MAE:", round(mae_test1, 2))


pipeline2 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=42))])
cv_scores2 = cross_val_score(pipeline2, X_train, y_train,
                            cv=5, scoring='neg_mean_absolute_error')
cv_mae2 = -cv_scores2.mean()
print("Model 2 CV MAE:", round(cv_mae2, 2))

pipeline2.fit(X_train, y_train)
y_pred_test2 = pipeline2.predict(X_test)
mae_test2 = mean_absolute_error(y_test, y_pred_test2)
print("Model 2 Test MAE:", round(mae_test2, 2))

from sklearn.metrics import mean_squared_error
mse_test2 = mean_squared_error(y_test, y_pred_test2)
print("Model 2 Test RMSE:", round(np.sqrt(mse_test2), 2))



# Using grid search
# from sklearn.model_selection import GridSearchCV, KFold
# param_grid = {
#     'model__n_estimators': [10, 30, 50],
#     'model__max_depth': [None, 10, 20, 30],
#     'model__min_samples_split': [2, 5, 10],
#     'model__min_samples_leaf': [1, 2, 4],
#     'model__max_features': ['sqrt', 'log2'],
# }
# cv = KFold(n_splits=5, shuffle=True, random_state=42)
# grid = GridSearchCV(
#     estimator=pipeline2,
#     param_grid=param_grid,
#     scoring='neg_mean_absolute_error',
#     cv=cv,
#     n_jobs=-1,
#     refit=True,           
#     verbose=1,
#     return_train_score=True
# )
# grid.fit(X_train, y_train)

# print("Best CV MAE:", -grid.best_score_)
# print("Best params:", grid.best_params_)
# y_pred = grid.predict(X_test)
# print("Test MAE:", mean_absolute_error(y_test, y_pred))
