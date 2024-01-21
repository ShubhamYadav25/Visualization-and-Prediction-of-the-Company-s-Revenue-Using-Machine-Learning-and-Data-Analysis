#------------------------------------------------------------- Importing packages ----------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#----------- Plot -----------------------------------------------
from sklearn.tree import plot_tree

#---------- Models ----------------------------------------------
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#---------- Model evaluation metrics ----------------------------
from sklearn import metrics 
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse 
from math import sqrt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#--------------------------------------------------- DATASET ANALYSIS --------------------------------------------------------------------
# Reading dataset 
df = pd.read_csv (r'dataset.csv');

# Dimension of dataset
print("Dimension of dataset",df.shape)  

#Printing top 25 data
print("Top 25 data from dataset\n")
df.head(25)

# features and its data type
df.info()

print()

# Categorical data and its values 
# categorical_df = df.loc[:,df.dtypes==np.object]
# print("Categorical data features : ", df.select_dtypes(include=np.object).columns.tolist())

# Numerical data features
print("Numerical data features ", df.select_dtypes(include=np.number).columns.tolist())

#------------------------------------------------ DATA ANALYSIS -------------------------------------------------------
# Finding statistics of dataset 
print(df.describe())
# 25% of values are less than 1 

print()

# finding missing values 
print("Missing values count\n", df.isnull().sum());

# count month and year wise sales 
df['Date']= pd.to_datetime(df['Date']) 
print("\nYear wise Month wise purchase amount:")
result = df.groupby([df['Date'].dt.year, df['Date'].dt.month]).agg({'Total':sum})
print(result)

# ----------- Pie chart with month wise sale
x1 = result['Total']
mylabels = ["1", "2", "3", "4","5","6","7","8","9","10","11","12"]

total = sum(x1)

plt.pie(x1, labels = mylabels, autopct=lambda p: '{:.0f}'.format(p * total / 100))
plt.title("Month wise total sales in year 2020")
plt.show() 

#------------------------------------------------------------ Distribution plot ----------------------------------------------
# Numerical features distribution plot
sns.set()
plt.figure(figsize=(6,6))
plt.title("Rate distribution plot ")
sns.distplot(df['Rate'],bins=15)
plt.show()

# count plot of rates
plt.title("Total distribution plot ")
# sns.countplot(x='Discount' , data=df,bins=15)
sns.distplot(df['Total'],bins=15)
plt.show()

# Categorical data distributions
sns.set()
plt.title("countplot of Category in dataset")
sns.countplot(x='Category' , data=df)
plt.show()

# #Date
# sns.set()
# sns.countplot(x='Date' , data=df.head(50000))
# plt.show()

#--------------------------------  heatmap -------------------------------
# Heatmaps describe relationships between variables in form of colors instead of numbers
#get top 70000 rows 
# dd = df.iloc[:70000]

# Plot the heatmap
dd=df.head(25)
# plt.figure(figsize=(10,10))
# heat_map = sns.heatmap( dd, linewidth = 1 , annot = True)
# plt.title( "HeatMap of top data" )
# plt.show()

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(dd.corr(), center=0, annot=True)
ax.set_title("Heatmap of dataset") 
plt.show()

#----------------------------------------------- DATA PRE-PROCESSING ----------------------------------------------------------
#there is may x axis values are similiar in categorical like same date or same category
# so first we count the values in category colm
print(df['Category'].value_counts())
# if some colm values same like wines or wine like that so we just replace function 
# and replace the wine or wines into 1 fix name df.replace()

# -------------  LABEL ENCODING ----------------------
# here we convert categorical data into numerical values
encoder = LabelEncoder()

#now it will transform the categorical data into fix numerical values 
df['Category'] = encoder.fit_transform(df['Category']);

#now check top 5 values
#print(df.head())           
#it will convert BEVERAGE ,wines ... into 0 ,1 like these values

# dat is also categorical
df['Date'] = encoder.fit_transform(df['Date']);

# bill number 
#df['BillNumber'] = encoder.fit_transform(df['BillNumber']);
# print(df['Bill Number'])


# items  
df['ItemDesc'] = encoder.fit_transform(df['ItemDesc']);

# time  Time
df['Time'] = encoder.fit_transform(df['Time']);

# now we check
print(df.head())

#------------------------ split ------------------------------------------
# target is our total revenue  colm

#storing all colm in list
colms = []
for col in df.columns:
    colms.append(col)
    # print(col)

#target is Total revenue
#now we drop it and store it into x 

# x will have all features expect target colm
# if we dropping colm so axis =1 , for row axis =0
x =df.drop(columns='Total',axis=1)
y = df['Total']
# print(x)

# x - It stores all the colm value except Total
# y- It stores Total colm value

#---now we split training and testing data 
# 20% as test 
x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size=0.3 , random_state=2)

# now we check how many are in train and test
print(x.shape  ,  x_train.shape , x_test.shape)

#----------------------------------------- MODEL TRAINING AND TESTING --------------------------------------------------------

#-- now we train the model
regressor = XGBRegressor()

# now we train our model
# by x_train and y_train
# it will find pattern between x_train to corresponding price value in y_train
regressor.fit(x_train , y_train)

#--------------------------------------------------------------- training --------------------------------------------------
# now we use our model for prediction 
# lets predict the value of x_train
#predict of training data 
# it will give predicted value 
# and y_train is original value of x_train
train_predict = regressor.predict(x_train)

# print(train_predict)

print("TRAINING PERFORMANCE MEASURE..........................\n")
# now we use R squared value for checking the distance between original value and predicted value
# we use func r2_score(orginal_value , predicted_value)
# original value is y_train  corressponding to x_train and train_predict is predicted value of x_train
# range of r2_score 0 to 1
r2_value_train =metrics.r2_score(y_train,train_predict)
print("R square value of training:" ,r2_value_train)


#---- MAE of train
MAE_VALUE = mae(y_train, train_predict);
print("Mean Absolute Error of training is: ",MAE_VALUE);


#---- MSE of train
MSE_TRAIN = mse(y_train, train_predict);
print("Mean Squared Error of training is: ",MSE_TRAIN);

# --- RMSE of train
print("Root Mean Square Error  of training is: ",sqrt(MSE_TRAIN));

print()

#--------------------------------------------------------  testing --------------------------------------------------------------- 
test_predict = regressor.predict(x_test)
# so y_test is origional value of x_test   and test_predict is predicted value of x_test

print("TESTING PERFORMANCE MEASURE..........................\n")

print("Predicted values\n")
print(test_predict)

print()

# now we again check the r2 sqaure
r2_value_test =metrics.r2_score(y_test,test_predict)
print("R square value of testing :" ,r2_value_test)

# MAE of testing
MAE_VALUE_test = mae(y_test, test_predict);
print("Mean Absolute Error of testing is: ",MAE_VALUE_test);


#---- MSE of TEST
MSE_TEST = mse(y_test, test_predict);
print("Mean Squared Error of testing is: ",MSE_TEST);

# --- RMSE of test
print("Root Mean Square Error  of testing is: ",sqrt(MSE_TEST));

#-------------------------------------------- COMPARATIVE ANALYSIS ------------------------------------------------

#============= Implement Linear Regression =====================
# Model intialize
linearRegressionModel = LinearRegression()

# fit regression model 
linearRegressionModel.fit(x_train, y_train)

# Predicted value
linearRegressionPredictedValue = linearRegressionModel.predict(x_test)

# Model evaluation
mseValueOfLinearRegression = mse(y_test, linearRegressionPredictedValue)
r2ScoreValueOfLinearRegression = metrics.r2_score(y_test, linearRegressionPredictedValue)

print("MSE value of LinearRegression: ", mseValueOfLinearRegression)
print("R2_score value of LinearRegression: ", r2ScoreValueOfLinearRegression)

print("")

#=========================== Implement Ridge Regression =====================

# Model intialize with alpha 0.5, if alhpa = 0 it means linear regression
ridgeRegressionModel = Ridge(alpha = 0.5)

# Fit ridge model 
ridgeRegressionModel.fit(x_train, y_train)

# Predict value 
ridgeRegressionPredictedValue = ridgeRegressionModel.predict(x_test)

# Model evaluation
mseValueOfRidgeRegression = mse(y_test, ridgeRegressionPredictedValue)
r2ScoreValueOfRidgeRegression = metrics.r2_score(y_test, ridgeRegressionPredictedValue)

print("MSE value of RidgeRegression: ", mseValueOfRidgeRegression)
print("R2_score value of RidgeRegression: ", r2ScoreValueOfRidgeRegression)
print("")

#=========================== Implement Lasso Regression =====================

# Model intialize with alpha 0.1
lassoRegressionModel = Lasso(alpha = 0.1)

# Fit lasso model 
lassoRegressionModel.fit(x_train, y_train)

# Predict value 
lassoRegressionPredictedValue = lassoRegressionModel.predict(x_test)

# Model evaluation
mseValueOfLassoRegression = mse(y_test, lassoRegressionPredictedValue)
r2ScoreValueOfLassoRegression = metrics.r2_score(y_test, lassoRegressionPredictedValue)

print("MSE value of LassoRegression: ", mseValueOfLassoRegression)
print("R2_score value of LassoRegression: ", r2ScoreValueOfLassoRegression)
print("")

#=========================== Implement Decision Tree Regressor =====================

# Model intialize 
decisionTreeRegressor = DecisionTreeRegressor(random_state=44)

# Fit model 
decisionTreeRegressor.fit(x_train, y_train)

# Predict value 
decisionTreeRegressorPredictedValue = decisionTreeRegressor.predict(x_test)

# Model evaluation
mseValueOfdecisionTreeRegressor = mse(y_test, decisionTreeRegressorPredictedValue)
r2ScoreValueOfdecisionTreeRegressor = metrics.r2_score(y_test, decisionTreeRegressorPredictedValue)

print("MSE value of DecisionTreeRegressor: ", mseValueOfdecisionTreeRegressor)
print("R2_score value of DecisionTreeRegressor: ", r2ScoreValueOfdecisionTreeRegressor)
print("")

#------------  If user want to see the DT made by model
# plt.figure(figsize=(10,8), dpi=150)
# plot_tree(decisionTreeRegressor, feature_names=x.columns);
# plt.show()


#=========================== Implement Random forest Regressor =====================

# Model intialize 
randomForestRegressor = RandomForestRegressor(max_depth=4, min_samples_split=5, n_estimators=500,
                      oob_score=True, random_state=42, warm_start=True)

# Fit model 
randomForestRegressor.fit(x_train.values, y_train)

# Predict value 
randomForestRegressorPredictedValue = randomForestRegressor.predict(x_test)

# Model evaluation
mseValueOfRandomForestRegressor = mse(y_test, randomForestRegressorPredictedValue)
r2ScoreValueOfRandomForestRegressor = metrics.r2_score(y_test, randomForestRegressorPredictedValue)

print("MSE value of RandomForestRegressor: ", mseValueOfRandomForestRegressor)
print("R2_score value of RandomForestRegressor: ", r2ScoreValueOfRandomForestRegressor)
print("")





