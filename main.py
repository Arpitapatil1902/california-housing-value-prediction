
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1 load data
housing=pd.read_csv("housing.csv")

# 2. Create a stratified test set based on income category
housing["income_cat"]=pd.cut(housing["median_income"],
                             bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                             labels=[1, 2, 3, 4, 5])

split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set=housing.loc[train_index].drop("income_cat",axis=1) # we will work on this training set
    strat_test_set=housing.loc[test_index].drop("income_cat",axis=1)   # set aside test set for later evaluation

#  Work on a copy of training data
housing=strat_train_set.copy()

# 3. Separate predictors and labels
housing_labels=housing["median_house_value"].copy()
housing=housing.drop("median_house_value",axis=1)   

print(housing.head())

# 4 seperate numerical and categorical columns
num_attri=housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attri=["ocean_proximity"]

# 5 Create a pipeline 
# for numerical attributes
num_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("std_scaler",StandardScaler())
])
# for categorical attributes
cat_pipeline=Pipeline([
    ("one_hot",OneHotEncoder(handle_unknown="ignore"))
])
# construct the full pipeline
full_pipeline=ColumnTransformer([
    ("num", num_pipeline, num_attri),
    ("cat", cat_pipeline, cat_attri),
])
# 6 transform the data
housing_prepared=full_pipeline.fit_transform(housing)

print(housing_prepared.shape)

#7 train model
#linear regression model
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)# housing_prepared is preprocessed data
lin_predictions=lin_reg.predict(housing_prepared)
lin_rmse=root_mean_squared_error(housing_labels, lin_predictions)
print(f"Linear Regression RMSE: {lin_rmse:.2f}")

# Decision Tree model
dec_tree_reg=DecisionTreeRegressor()
dec_tree_reg.fit(housing_prepared, housing_labels)
dec_tree_predictions=dec_tree_reg.predict(housing_prepared)
#dec_tree_rmse=root_mean_squared_error(housing_labels, dec_tree_predictions)
dec_rmses=-cross_val_score(dec_tree_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
#print(f"Decision Tree RMSE: {dec_rmses.mean():.2f}")
print("Decision Tree RMSE:")
print(pd.Series(dec_rmses).describe())

# Random Forest model
ran_forest=RandomForestRegressor()
ran_forest.fit(housing_prepared, housing_labels)
ran_forest_predictions=ran_forest.predict(housing_prepared)
#ran_forest_rmse=root_mean_squared_error(housing_labels, ran_forest_predictions)
ran_forest_rmses=-cross_val_score(ran_forest, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
#print(f"Random Forest RMSE: {ran_forest_rmse:.2f}")
print("Random Forest RMSE:")
print(pd.Series(ran_forest_rmses).describe())