import os
import numpy as np
import pandas as pd
import joblib
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

# joblib will make these files
MODEL_FILE="model.pkl"
PIPELINE_FILE="pipeline.pkl"

def build_pipeline(num_attri, cat_attri):
    #preprocessing for numerical attribute
    num_pipeline=Pipeline([
      ("imputre", SimpleImputer())  ,
      ("std_scaler", StandardScaler())
    ])
     #preprocessing for categorial attri
    cat_pipeline=Pipeline( [
        ("onehot_encoder", OneHotEncoder(handle_unknown="ignore"))

   ])
    #construct full pipeline
    full_pipeline=ColumnTransformer([
        ("numerical attribute", num_pipeline, num_attri),

        ("categorical attribute", cat_pipeline, cat_attri)
    ])
    return full_pipeline

#if else logic
if not os.path.exists(MODEL_FILE):
    #if model file do not exist then lets train the model
    # 1 load data
    data=pd.read_csv("housing.csv")

    # 2. Create a stratified test set based on income category
    data["income_cat"]=pd.cut(data["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])

    split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["income_cat"]):
        train_set=data.loc[train_index].drop("income_cat",axis=1) # we will work on this training set
        test_set=data.loc[test_index].drop("income_cat",axis=1).to_csv("input_data.csv", index=False)   # set aside test set for later evaluation

    housing_labels=train_set["median_house_value"].copy()
    housing_features=train_set.drop("median_house_value",axis=1)

    num_attri=housing_features.drop("ocean_proximity",axis=1).columns.tolist()
    cat_attri=["ocean_proximity"]

    pipeline=build_pipeline(num_attri, cat_attri)
    housing_prepared= pipeline.fit_transform(housing_features)

    # model
    model=RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    #creating inference using JOBLIB
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("MODEL IS TRAINED . CONGRATS!!")

else:
    #if model file exist then load the model and pipeline
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)

    input_data=pd.read_csv("input_data.csv")
    transformed_data=pipeline.transform(input_data)
    predictions=model.predict(transformed_data)

    input_data['medianhousevalue']=predictions
    input_data.to_csv("output.csv", index=False)
    print("inference is completed , results saved to output.csv file")

    
