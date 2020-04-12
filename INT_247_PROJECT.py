

# ## Name : Vishal Kumar

# House Price Prediction
# Description : This is a notebook for visualization of various features which the sales price of houses. Then data is taken from the "Kaggle House Price Prediction" challenge.




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
from scipy.stats import skew,norm
from scipy.stats.stats import pearsonr


# # Loading the Data




train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# ## Preview the data




train.head()





test.head()




train.describe()





train.info()





test.info()


# ## Data Manipulation and Visualization
# Lets check for NaN (null) values in the data




train.isnull().sum()





test.isnull().sum()


# Lets check for the mean, standard deviation for Sales price




train['SalePrice'].describe()





# Determining the Skewness of data 
print ("Skew is:", train.SalePrice.skew())





plt.hist(train.SalePrice)
plt.show()
sns.distplot(train.SalePrice,fit=norm)
plt.ylabel =('Frequency')
plt.title = ('SalePrice Distribution')
#Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
#QQ plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
print("skewness: %f" % train['SalePrice'].skew())
print("kurtosis: %f" % train ['SalePrice'].kurt())


# Sales price is right skewed. So, we perform log transformation so that the skewness is nearly zero




train["skewed_SP"] = np.log1p(train["SalePrice"])
print ("Skew is:", train['skewed_SP'].skew())
plt.hist(train['skewed_SP'], color='blue')
plt.show()
sns.distplot(train.SalePrice,fit=norm)
plt.ylabel=('Frequency')
plt.title=('SalePrice distribution');

(mu,sigma)= norm.fit(train['SalePrice']);

fig =plt.figure()
res =stats. probplot(train['SalePrice'], plot=plt)
plt.show() 


# Exploring the variables




#correration matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat,vmax=0.9, square=True)
plt.show()





# Data Transformation 
print ("Original: \n") 
print (train.Street.value_counts(), "\n")





train['SaleCondition'].value_counts()





train['SaleType'].value_counts()





# Turn into one hot encoding 
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)





# Encoded 
print ('Encoded: \n') 
print (train.enc_street.value_counts())





def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)
data = train.select_dtypes(include=[np.number]).interpolate().dropna()





sum(data.isnull().sum() != 0)





train.OverallQual.unique()





# Linear Model for the  train and test
y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)





from sklearn import linear_model
from sklearn import ensemble
lr = ensemble.GradientBoostingRegressor()





model = lr.fit(X_train, y_train)





print ("R^2 is: \n", model.score(X_test, y_test))





y_pred = model.predict(X_test)





from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, y_pred))





plt.scatter(y_pred, y_test, alpha=.75,color='b') 







