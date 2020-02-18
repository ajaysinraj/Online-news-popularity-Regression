#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[112]:


data=pd.read_csv('OnlineNewsPopularity.csv')
data.head()


# In[34]:


def fun(x):
    if (x<=1400):
        return 0
    else :
        return 1
data['popularity']=data[' shares'].apply(lambda x:fun(x))


# In[20]:


data.info() # already all the columns are numerical 


# In[113]:


data.columns=data.columns.str.replace(' ','')


# In[114]:


data.drop(columns=['url','timedelta'],axis=1,inplace=True)


# In[23]:


data.columns # is_weekend column should be remove it is repetitive


# # descriptive stastistics 

# # central tendency 
# meand ,median 

# In[125]:


#some feature  has outlier there 
# some feature has high standard deaviation ,means spreadness is high 
#n_tokens_contentn_non_stop_unique_tokens


# In[123]:


data.describe().T


# In[128]:


data.max()-data.min() # this show the range of each feature


# In[143]:


plt.hist(data['shares'],bins=3,log=False)  # there is right skew outlier are present that affect the model
plt.axvline(data['shares'].mean(),color='#ed0909')
plt.axvline(data['shares'].median(),color='#2a22c9')


# In[144]:


plt.hist(data['shares'],bins=3,log=True)
plt.axvline(data['shares'].mean(),color='#ed0909')# some small change after log transformation 
plt.axvline(data['shares'].median(),color='#2a22c9')


# In[162]:


np.percentile(data['shares'],99)


# In[163]:


# understanding the summary of statistics 
plt.figure(figsize=(10,10))
plt.boxplot(data['shares'])
plt.text(y=1400,x=0.80,s='median')
plt.text(y=946,x=0.70,s='first quartile')   # from this we can say that this column 99 percentil value are come in  31656
plt.text(y=2800,x=0.70,s='third quartile')  # it has to be transformed or remove the outlier
plt.text(y=31656,x=0.70,s='99 percentile ') # 
plt.show()


# In[164]:


data.var() # high variance means the distance between data points and the mean are so high
          # spreadness is high data is so disperesed


# In[165]:


data.std() # same thing come from variance and std , how far data points from the mean 
   # data has much dispersion and some feature has low std they are closely pack or data are less dispersed


# In[203]:


plt.figure(figsize=(15,15))
sns.heatmap(data.corr()[(data.corr()>0.5 )],cmap="YlGnBu")
plt.show()


# In[181]:


# from this map we can able to predict much information 
# some feature has high correlation  most of feature has no correlation but some has amazing values 
data.corr()[(data.corr()>0.5 )]
#n_non_stop_words  vs n_unique_tokens ,vs 
#(n_non_stop_unique_tokens vs [n_non_stop_words) vs n_non_stop_unique_tokens]
#n_unique_tokens vs n_non_stop_unique_tokens
#global_subjectivity vs average_token_length,max_positive_polarity
#rate_positive_words,avg positive word vs average_token_length


# In[22]:


# we make the groups of data like no of words 
words=data.iloc[:,0:5]
words['popularity']=data['popularity']
words['average_token_length']=data['average_token_length']
words['num_keywords']=data['num_keywords']


# In[23]:


words.head()


# In[14]:


data.columns


# In[191]:


# from this we can say that dense region of share are where , there has 10 -15 words in title 
# from this we can't interpret better which show good relatin ship
# no of shares are work well with limited average length of word in content 
#no of shares are increase with increase in no of keyword 


# In[122]:


# no of links ans videos , images,other links
links=data.iloc[:,5:9]
links['shares']=data['shares']


# In[202]:


sns.pairplot(links)


# In[ ]:


# dense area says less no of links videos images are high shares 
# high no of links ,videos ,images, other links are not much impact on shares 


# In[211]:


category=data.iloc[:,11:17]
category['LDA_00']=data['LDA_00']
category['LDA_01']=data['LDA_01']
category['LDA_02']=data['LDA_02']
category['LDA_03']=data['LDA_03']
category['LDA_04']=data['LDA_04']
category['shares']=data['shares']


# In[212]:


sns.pairplot(category)


# In[ ]:


# from graph we can conclude that business technology world news has good impact on shares 
#from lda feature with shares all have same relation as we know that its is the collection of top 5 articles which is different
#category,it tells that how much close our article to this 


# In[218]:


min_min=data.iloc[:,17:29]
min_min['shares']=data['shares']


# In[219]:


sns.pairplot(min_min)


# In[ ]:


#worst keyword min shares vs shares we can say in this case shares is not much high
# IN all worst keyword are same sitution no of shares is minimum 
#when we compare worst keyword with best keyword then it is clear no of shares is affected 
# shares of reference article is not clear


# In[224]:


day=data.iloc[:,29:37]
day['shares']=data['shares']


# In[225]:


sns.pairplot(day)


# In[ ]:


#from above graph we can say  weekday is affecte the shares when article publish on weekday then most probable 
#it will got high shares 
# on weekend no of shares is not much affected ,if we compare sunday saturday then sunday is low shares affector
# from this we can say article will publish on weekdays
# and avoid weekends for publishing article


# In[12]:


polarity=data.iloc[:,42:58]
polarity['shares']=data['shares']


# In[228]:


sns.pairplot(polarity)


# In[9]:


data['global_sentiment_polarity'].max()


# In[10]:


data['global_sentiment_polarity'].min()


# In[19]:


for i in polarity.columns:
    plt.hist(data[i])
    plt.xlabel(i)
    plt.show()


# In[20]:


data.head()


# In[6]:


data.isnull().sum().sum() # data has zero null values 


# In[32]:


data.head()


# In[99]:


X=data.drop(columns=['shares'],axis=1)
y=data['shares']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaletr=sc.fit_transform(x_train)
scalet=sc.transform(x_test)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
from sklearn.metrics import r2_score

model=lr.fit(x_train,y_train)
ypred=lr.predict(x_test)
print(r2_score(y_test,ypred))
print(lr.score(x_train,y_train))
print(lr.score(x_test,y_test))
print(np.sqrt(mean_squared_error(y_test,ypred)))


# In[115]:


data['shares']=np.sqrt(data['shares'])


# In[105]:


X=data.drop(columns=['shares'],axis=1)
y=data['shares']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaletr=sc.fit_transform(x_train)
scalet=sc.transform(x_test)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
from sklearn.metrics import r2_score

model=lr.fit(x_train,y_train)
ypred=lr.predict(x_test)
print(r2_score(y_test,ypred))
print(lr.score(x_train,y_train))
print(lr.score(x_test,y_test))
print(np.sqrt(mean_squared_error(y_test,ypred)))


# In[48]:


model=lr.fit(x_train,y_train)
ypred=lr.predict(x_test)


# In[56]:


from sklearn.metrics import r2_score

model=lr.fit(x_train,y_train)
ypred=lr.predict(x_test)
print(r2_score(y_test,ypred))
print(lr.score(x_train,y_train))
print(lr.score(x_test,y_test))
print(np.sqrt(mean_squared_error(y_test,ypred)))


# In[57]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaletr=sc.fit_transform(x_train)
scalet=sc.transform(x_test)
from sklearn.metrics import r2_score

model=lr.fit(scaletr,y_train)
ypred=lr.predict(scalet)
print(r2_score(y_test,ypred))
print(lr.score(scaletr,y_train))
print(lr.score(scalet,y_test))
print(np.sqrt(mean_squared_error(y_test,ypred)))


# In[53]:


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
model=dt.fit(x_train,y_train)
ypred=dt.predict(x_test)
print(dt.score(x_train,y_train))
print(dt.score(x_test,y_test))
from sklearn.metrics import r2_score,mean_squared_error
print(r2_score(y_test,ypred))
print(np.sqrt(mean_squared_error(y_test,ypred)))


# In[151]:


X=data.drop(columns=['shares'],axis=1)
y=data['shares']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
model=dt.fit(x_train,y_train) # sqrt
ypred=dt.predict(x_test)
print(dt.score(x_train,y_train))
print(dt.score(x_test,y_test))
from sklearn.metrics import r2_score,mean_squared_error
print(r2_score(y_test,ypred))
print(np.sqrt(mean_squared_error(y_test,ypred)))


# In[61]:



from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
model=dt.fit(x_train,y_train)
ypred=dt.predict(x_test)
print(dt.score(x_train,y_train))
print(dt.score(x_test,y_test))
from sklearn.metrics import r2_score,mean_squared_error
print(r2_score(y_test,ypred))
print(np.sqrt(mean_squared_error(y_test,ypred)))


# In[54]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
model=rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
print(rf.score(x_train,y_train))
print(rf.score(x_test,y_test))
print(np.sqrt(mean_squared_error(y_test,y_pred)))


# In[111]:


X=data.drop(columns=['shares'],axis=1)
y=data['shares']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42) #sqrt
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
model=rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
print(rf.score(x_train,y_train))
print(rf.score(x_test,y_test))
print(np.sqrt(mean_squared_error(y_test,y_pred)))


# In[62]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
model=rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
print(rf.score(x_train,y_train))
print(rf.score(x_test,y_test))
print(np.sqrt(mean_squared_error(y_test,y_pred)))


# In[55]:


print(np.sqrt(mean_squared_error(y_test,y_pred)))


# In[45]:


pd.DataFrame(dt.feature_importances_,index=x_test.columns)


# In[ ]:





# # split the data

# In[5]:


X=data.drop('shares',axis=1)
y=data['shares']


# In[24]:


from sklearn.model_selection import train_test_split


# In[ ]:


# we split the data beacause from this we evaluate our model like example if we train full data and apply model 
# how we confident that our model will work or not for this situation we split the data into train and test 
# a small portion of data split into test , it is unseen from every thing finally when our model ready than we pass this test 
# data to our model if our model work well on test data means our model also work well on production 
# there are many option to split the data 70:30,60:40,80:20,90:10
# it depends on data if data is large you can choose 90:10,80:20 but 70:30 is common proportion for split


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[26]:


# we choose our basic model linear regression 
# before applying linear regression we have to check assumption of linear regression 


# In[29]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[30]:


model=lr.fit(x_train,y_train)
ypred=lr.predict(x_test)


# In[31]:


from sklearn.metrics import r2_score


# In[ ]:


# rsquare tells us how much variation of y variable explain by x variables or how much x variables compatiable with y variable
# we calculate rsquare manually and by inbuilt funtion 
# our rsquare is not good
# 1-SSE/SST


# In[37]:


rsquare=1-(sum(np.square(y_test-ypred))/(sum(np.square(y_test-np.mean(y_test)))))
rsquare


# In[33]:


r2_score(y_test,ypred)


# In[42]:


print('score on train',lr.score(x_train,y_train))
print('score on test',lr.score(x_test,y_test))


# # Use statsmodels 

# In[42]:


from statsmodels.api import OLS


# In[43]:


model=OLS(endog=y,exog=X).fit()
model.summary()


# In[ ]:





# # our model is suffer from high bias error that why is underfitting 
# # and it is not able to learn train as well as test data 

# In[ ]:


# when I split our data  into 80:20 I got better rsquare than before


# In[44]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# we choose our basic model linear regression 
# before applying linear regression we have to check assumption of linear regression 

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

model=lr.fit(x_train,y_train)
ypred=lr.predict(x_test)

from sklearn.metrics import r2_score

# rsquare tells us how much variation of y variable explain by x variables or how much x variables compatiable with y variable
# we calculate rsquare manually and by inbuilt funtion 
# our rsquare is not good
# 1-SSE/SST

rsquare=1-(sum(np.square(y_test-ypred))/(sum(np.square(y_test-np.mean(y_test)))))
print('manual rsquare',rsquare)

print('rsquare',r2_score(y_test,ypred))

print('score on train',lr.score(x_train,y_train))
print('score on test',lr.score(x_test,y_test))


# In[ ]:


# when we split our data into 90:10 it is worse than first so 80:20 is better for this model 


# In[46]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# we choose our basic model linear regression 
# before applying linear regression we have to check assumption of linear regression 

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

model=lr.fit(x_train,y_train)
ypred=lr.predict(x_test)

from sklearn.metrics import r2_score

# rsquare tells us how much variation of y variable explain by x variables or how much x variables compatiable with y variable
# we calculate rsquare manually and by inbuilt funtion 
# our rsquare is not good
# 1-SSE/SST

rsquare=1-(sum(np.square(y_test-ypred))/(sum(np.square(y_test-np.mean(y_test)))))
print('manual rsquare',rsquare)

print('rsquare',r2_score(y_test,ypred))

print('score on train',lr.score(x_train,y_train))
print('score on test',lr.score(x_test,y_test))


# In[ ]:


# overfitting : when our model suffer form high variance error and it work well on train data
# but not well test data , it capture every pattern from train data and not able to apply on test data
# we can say our model is generalisation it is very complex that why it has high variance error

# underfitting : when our model suffer from high bias error and it is not learning any thing from train as well as test data
# as you can see in this we apply linear regression model , score is very low approx 2 %  on both means our model has not 
# sufficient power to learn this data 

# rsquare is very low , it tells error proportion is very high 
# model is suffer form underfitting so its solution is we have to make complex model that learn well on train data
# then we can regularize the model and we can trade of between bias and variance error

# many reason of this situation is like data is normal or it has outlier , it need transformation 
# final conclusin in  our basic model that it is very simple ,it give us to idea about the base error that where we can start the project


# In[49]:


from sklearn.metrics import mean_squared_error
print('RMSE ',np.sqrt(mean_squared_error(y_test,ypred)))


# # RMSE is very high it tells us that our your predicted value is how much far from actual value in this case it tell predicted shares is 13580 far from actual      shares

# In[50]:


from sklearn.preprocessing import PolynomialFeatures
pl=PolynomialFeatures()


# In[53]:


X_train=pl.fit_transform(x_train)
X_test=pl.transform(x_test)


# In[55]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()

model=lr.fit(X_train,y_train)
ypred=lr.predict(X_test)


# In[63]:


from sklearn.metrics import r2_score

# rsquare tells us how much variation of y variable explain by x variables or how much x variables compatiable with y variable
# we calculate rsquare manually and by inbuilt funtion 
# our rsquare is not good
# 1-SSE/SST
# model is very worst 

rsquare=1-(sum(np.square(y_test-pd.Series(ypred)))/(sum(np.square(y_test-np.mean(y_test)))))
print('manual rsquare',rsquare)

print('rsquare',r2_score(y_test,pd.Series(ypred)))

print('score on train',lr.score(X_train,y_train))
print('score on test',lr.score(X_test,y_test))


# # waste data

# In[8]:


print(data[data['n_tokens_content']==0].shape) # no of words are zero , it is noise 
print(data.shape)
# if article has zero words means there is no article exist


# In[9]:


data=data[data['n_tokens_content']!=0] # after removing this 
data.shape


# In[10]:


data['n_unique_tokens']=data['n_unique_tokens'].replace(701,np.nan) #one value has greater than 1
data['n_non_stop_words']=data['n_non_stop_words'].replace(1042,np.nan)
data.dropna(inplace=True)


# In[171]:


data.describe().T


# # Checking outlier 

# # n_tokens_content,num_hrefs,num_self_hrefs,num_imgs,num_videos,shares

# In[97]:


for i in data.columns:
    sns.boxplot(data[i])
    plt.show()


# In[ ]:


# from the boxplot we can say that some feature has so many outliers that badly affect our model 
# so we have to remove this outliers lets make those features
# we removing the outlier using z score method


# In[11]:


woutlier=data.copy()


# In[103]:


def outliers_indices(feature):

    mid = data[feature].mean()
    sigma = data[feature].std()
    return data[(data[feature] < mid - 3*sigma) | (data[feature] > mid + 3*sigma)].index


# In[104]:


#wrong_shares1 = outliers_indices('kw_avg_avg')
#wrong_shares2= outliers_indices('kw_max_avg')
#wrong_shares3= outliers_indices('kw_avg_max')
#wrong_shares4= outliers_indices('kw_max_max')
#wrong_shares5= outliers_indices('kw_min_max')
#wrong_shares6= outliers_indices('kw_avg_min')
#wrong_shares7= outliers_indices('kw_max_min')
#wrong_shares8= outliers_indices('self_reference_avg_sharess')
#wrong_shares9= outliers_indices('self_reference_max_shares')
#wrong_shares10= outliers_indices('self_reference_min_shares')
wrong_shares = outliers_indices('shares')
wrong_vid = outliers_indices('num_videos')
wrong_img = outliers_indices('num_imgs')
wrong_content = outliers_indices('n_tokens_content')
wrong_hrefs = outliers_indices('num_hrefs')
wrong_self_hrefs = outliers_indices('num_self_hrefs')
out = set(wrong_vid) | set(wrong_img) | set(wrong_content) |set(wrong_hrefs)|set(wrong_self_hrefs)|set(wrong_shares)

data.drop(out,inplace=True)


# In[14]:


data.shape


# In[15]:


woutlier.shape


# # No of outlier present in the data 

# In[16]:


woutlier.shape[0]-data.shape[0] #those affect the result


# In[58]:


data['constant']=1


# # model after removing of outlier

# In[59]:


from sklearn.model_selection import train_test_split
X=data.drop('shares',axis=1)
y=data['shares']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# we choose our basic model linear regression 
# before applying linear regression we have to check assumption of linear regression 

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

model=lr.fit(x_train,y_train)
ypred=lr.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error

# rsquare tells us how much variation of y variable explain by x variables or how much x variables compatiable with y variable
# we calculate rsquare manually and by inbuilt funtion 
# our rsquare is not good
# 1-SSE/SST

rsquare=1-(sum(np.square(y_test-ypred))/(sum(np.square(y_test-np.mean(y_test)))))
print('')
print('AFTER REMOVING OF OUTLIER')
print('manual rsquare',rsquare)

print('rsquare',r2_score(y_test,ypred))

print('score on train',lr.score(x_train,y_train))
print('score on test',lr.score(x_test,y_test))
print('RMSE ',np.sqrt(mean_squared_error(y_test,ypred)))
print(' ')
print('BEFORE REMOVING OF OUTLIER')

print('manual rsquare 0.026180071647286973')
print('rsquare 0.026180071647297076')
print('score on train 0.021916499363106468')
print('score on test 0.026180071647297073')
print('RMSE 13580.349105800738')


# # There is huge change in RMSE after removing of outlier 
# # it s working well 

# In[118]:


plt.figure(figsize=(15,15))
sns.heatmap(data.corr()[data.corr()>0.5]) # those feature has greater than 0.5 value that positive correlation 
# 


# In[121]:


for i in polarity.columns:
    sns.scatterplot(polarity[i],polarity['shares'])
    plt.show()


# In[ ]:


# in global subjectivity there is specific range like 0.3 to 0.8 has much shares 
#global rate negative words are low means share is high 
# rate of positve words are high then shares is high 
# polarity also affect the shares of article 


# In[124]:


for i in links.columns:
    sns.scatterplot(links[i],links['shares'])
    plt.show()


# In[125]:


#  no of links , videos , images and links of other article  has shown if these are less shares are high
# no of links are less than 30 has much shares we can say that if article has less than 30 links so it can be affect the shares
# if less no of links of other article  than shares is high 
# no of videos are also less 


# In[126]:


for i in data.columns:
    sns.distplot(data[i])
    plt.show()


#   # Assumption of Linear Regression 

# #### Linearity ,Normality, Multicollinearity,Homoscedasticity,Autocorrelation 

# ##  Linearity
# Linear regression needs the relationship between the independent and dependent variables to be linear. Let's use a pair plot to check the relation of independent variables with the Shares variable

# In[183]:


sns.pairplot(data[words.columns])


# In[132]:


import statsmodels.stats.api as smi
import statsmodels.api as sm


# In[84]:



model=sm.OLS(endog=y,exog=X).fit()
model.summary()


# In[47]:


from statsmodels.stats.api  import  linear_rainbow
residual=y_test-ypred


# #
# H0:  The Null hypothesis is that the regression is correctly modelled as linear
# 
# Ha: Alternate is that is no linear

# In[80]:


tstatstic,pvalue=linear_rainbow(model)
print(pvalue)
# pvalue is greater than alpha 0.05 (significance level )
# that we reject null hypothisis data is not linear 
# linearity is not met 


# In[22]:


from scipy.stats import probplot


# In[40]:


sns.residplot(residual,ypred,lowess=True)  # residual are random in nature 
plt.ylabel('residual')
# there is no pattern in residuals


# # After log transformation

# In[52]:


def fun(x):
    return np.log(x)


# In[85]:


woutlier.describe().T
#we make it linear by apply using  log transformation but it convert into nan and infinity value 
# its not well
# we cant make it linear and also we can conclude from pairplot of the data that there is no feature that has linear
# relation with target variable 
# so linearity assumption is not met 


# In[84]:


woutlier['n_tokens_content']=woutlier['n_tokens_content'].apply(lambda x:fun(x))
woutlier['shares']=woutlier['shares'].apply(lambda x:fun(x))
woutlier['n_tokens_title']=woutlier['n_tokens_title'].apply(lambda x:fun(x))
woutlier['num_hrefs']=woutlier['num_hrefs'].apply(lambda x:fun(x))
woutlier['num_self_hrefs']=woutlier['num_self_hrefs'].apply(lambda x:fun(x))
woutlier['num_imgs']=woutlier['num_imgs'].apply(lambda x:fun(x))
woutlier['num_videos']=woutlier['num_videos'].apply(lambda x:fun(x))


# # Normality Test
# ## we can using QQ plot 
# ## mean=median=mode and
# ## statstical test like shapiro wilk and anderson darling test

# In[89]:


from scipy.stats import probplot
import pylab


# In[101]:


probplot(residual,plot=pylab)
plt.show()
# from the graph we can conclude data is not normal but graph can be decieve 
# so we can check different matrix
# we can also conclude from skewness that data is normal or not 
print(data.skew())


# 2. Mean of Residuals
# 
# Residuals as we know are the differences between the true value and the predicted value. 
# 
# One of the assumptions of linear regression is that the mean of the residuals should be zero.

# In[106]:


np.mean(residual) # residual is not zero


# In[1]:


from scipy.stats import jarque_bera,anderson


# In[35]:


for i in data.columns:
    print(i,anderson(data[i],dist='norm')[0]<1) # from this we can conclude that is not normal


# In[33]:


anderson(data['shares'],dist='norm')[0:2]


# # Multicollinearity test

# ### Multicollinear means predictors are correlated each other . Presence of correlation in 
# ### independent variable leads multicollinearity
# 
# ### If it present then for model it will be difficult to identify that which variable has true effect 
# ### on dependent variable 
# 
# ### Multicollinear measure by VIF (variance influence factor vif=(1/1-R^2)  it tells us predictors
# ### are correlated  if vif=1 then multicollinear is not present and if vif between 5 to 10 then correlation may be present but vif >10 then we confirm that correlation between predictore is present 
# 
# ### we can drop those features or we can merge into single features

# In[37]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[60]:


vif=[variance_inflation_factor(data.values,i) for i in range(data.shape[1])]


# In[61]:


vifd=pd.DataFrame({'vif':vif[0:]},index=data.columns)


# In[62]:


vifd[vifd>10] # from this we can say that multicollinarity is present 
 # we can drop or merge depends on features 
    # we can drop these features like n_unique_tokens ,kw_avg_avg
    # we can merge these features kw_max_min,kw_avg_min,rate_positive_words,rate_negative_words
    #self_reference_max_shares,self_reference_avg_sharess


# ## Autocorrelation Test 
# 

# ### autocorrelation means errors are correlated each other 
# 
# ### when errors are correlated each other then something wrong in result 
# #### example metrologist forecast the temperature then in winter when they  predict value then the  actual value is always less  and in summer actual  always greater than predicted value it is happen when autocorrealtion is present 
# 
# # test is durbin watson
# 
# ### H0: there is no serial correlation between errors
# 
# ### Ha: there is serial correlation between errors
# 
# ### statstic =2(1-r)
# 
# #### if statstic is 2 when r=0 r is sample autocorrelation ,statstic between 0 to 2 positive correlation and 2 to 4 negative correlation

# In[44]:


from statsmodels.stats.stattools import durbin_watson


# In[48]:


durbin_watson(residual)# statstic is less than 2 it indicate that positive serial correlation 
# autocorrelation is present 


# ### Hetroscedasticity test 
# 
# #### means data has different dispresion or we can say data has unequal variance across different range values
# ### it is not good for model because OLS regression assumes that residual has equal variance across different range of values or follow homoscedasticity but if it is present it ruins the result and get biased coefficient 
# 
# ### statstical test are breuch pagan and gold feld quant

# In[52]:


from statsmodels.stats.api import het_breuschpagan,het_goldfeldquandt,
# H0: equal variance 
# Ha: unequal variance


# In[74]:


lm,lmpvalue,fstats,pvalue=het_breuschpagan(residual,x_test)


# In[129]:


import statsmodels as sm


# In[130]:


pvalue # we reject null hypothisis that hetroscedasticity is present


# Feature selection

# In[140]:


def back_feature_elem (data_frame,dep_var,col_list):
    """ Takes in the dataframe, the dependent variable and a list of column names, runs the regression repeatedly eliminating feature with the highest
    P-value above alpha one at a time and returns the regression summary with all p-values below alpha"""

    while len(col_list)>0 :
        modelr=sm.OLS(dep_var,data_frame[col_list])
        result=modelr.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.05):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)

result=back_feature_elem(X_train,y_train,X.columns)


# In[139]:


from sklearn.model_selection import train_test_split
X=data.drop('shares',axis=1)
y=data['shares']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[147]:


result.summary()


# In[150]:


np.sqrt(result.mse_model)


# In[ ]:





# In[ ]:





# In[65]:


X=data.drop('shares',axis=1)
y=data['shares']


# RFE

# In[77]:


from sklearn.feature_selection import RFE
rfe=RFE(estimator=lr,n_features_to_select=30)
selct=rfe.fit(X,y)
sd=pd.DataFrame(selct.ranking_,index=X.columns)
lk=sd[sd[0]==1]
lk.index


# In[116]:


X=data[['n_unique_tokens', 'n_non_stop_words', 'data_channel_is_lifestyle',
       'data_channel_is_entertainment', 'data_channel_is_bus',
       'data_channel_is_tech', 'data_channel_is_world', 'weekday_is_monday',
       'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday',
       'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday',
       'is_weekend', 'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04',
       'global_subjectivity', 'global_sentiment_polarity',
       'global_rate_positive_words', 'global_rate_negative_words',
       'rate_positive_words', 'avg_positive_polarity', 'min_positive_polarity',
       'avg_negative_polarity', 'max_negative_polarity',
       'title_sentiment_polarity']]
y=data['shares']


# In[81]:


from sklearn.model_selection import train_test_split
y=data['shares']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
model=rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
print(rf.score(X_train,y_train))
print(rf.score(X_test,y_test))
print(np.sqrt(mean_squared_error(y_test,y_pred)))


# In[108]:


pd.DataFrame(selct.ranking_,index=['n_tokens_title', 'n_tokens_content', 'n_unique_tokens',
       'n_non_stop_words', 'n_non_stop_unique_tokens', 'num_hrefs',
       'num_self_hrefs', 'num_imgs', 'num_videos', 'average_token_length',
       'num_keywords', 'data_channel_is_lifestyle',
       'data_channel_is_entertainment', 'data_channel_is_bus',
       'data_channel_is_socmed', 'data_channel_is_tech',
       'data_channel_is_world', 'kw_min_min', 'kw_max_min', 'kw_avg_min',
       'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg',
       'kw_avg_avg', 'self_reference_min_shares', 'self_reference_max_shares',
       'self_reference_avg_sharess', 'weekday_is_monday', 'weekday_is_tuesday',
       'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',
       'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend', 'LDA_00',
       'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 'global_subjectivity',
       'global_sentiment_polarity', 'global_rate_positive_words',
       'global_rate_negative_words', 'rate_positive_words',
       'rate_negative_words', 'avg_positive_polarity', 'min_positive_polarity',
       'max_positive_polarity', 'avg_negative_polarity',
       'min_negative_polarity', 'max_negative_polarity', 'title_subjectivity',
       'title_sentiment_polarity', 'abs_title_subjectivity',
       'abs_title_sentiment_polarity'])


# In[110]:


X=data[[ 'n_unique_tokens',
       'n_non_stop_words', 'data_channel_is_lifestyle',
       'data_channel_is_entertainment', 'data_channel_is_bus',
       'data_channel_is_socmed', 'data_channel_is_tech',
       'data_channel_is_world', 'weekday_is_monday', 'weekday_is_tuesday',
       'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',
       'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend', 'LDA_00',
       'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 'global_subjectivity',
       'global_sentiment_polarity', 'global_rate_positive_words',
       'global_rate_negative_words', 'rate_positive_words', 'avg_positive_polarity', 'min_positive_polarity',
        'avg_negative_polarity', 'max_negative_polarity']]


# In[118]:


from sklearn.model_selection import train_test_split
X=data[['n_unique_tokens', 'n_non_stop_words', 'data_channel_is_lifestyle',
       'data_channel_is_entertainment', 'data_channel_is_bus',
       'data_channel_is_tech', 'data_channel_is_world', 'weekday_is_monday',
       'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday',
       'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday',
       'is_weekend', 'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04',
       'global_subjectivity', 'global_sentiment_polarity',
       'global_rate_positive_words', 'global_rate_negative_words',
       'rate_positive_words', 'avg_positive_polarity', 'min_positive_polarity',
       'avg_negative_polarity', 'max_negative_polarity',
       'title_sentiment_polarity']]
y=data['shares']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

scaleXtrain=sc.fit_transform(X_train)

X_train=pd.DataFrame(scaleXtrain, columns=list(X_train))

scaleXtest=sc.transform(X_test)

X_test=pd.DataFrame(scaleXtest, columns=list(X_train))

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error,r2_score
lp=Lasso()
modelreg=lp.fit(X_train,y_train)
ypredreg=lp.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,ypredreg)))
print(r2_score(y_test,ypredreg))


# ### LASSO

# In[119]:


from sklearn.model_selection import cross_val_score
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# In[120]:


y_train.shape


# In[121]:


X=data.drop('shares',axis=1)
y=data['shares']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.linear_model import LassoCV
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y_train)
rmse_cv(model_lasso).mean()


# In[122]:


model_lasso.coef_


# In[123]:


coef = pd.Series(model_lasso.coef_, index = X_train.columns)
coef.head()


# In[124]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[125]:


coef[coef!=0].index


# In[126]:


from sklearn.model_selection import train_test_split
X=data[['n_tokens_title', 'n_tokens_content', 'n_non_stop_words', 'num_hrefs',
       'num_self_hrefs', 'num_imgs', 'num_videos', 'average_token_length',
       'num_keywords', 'data_channel_is_entertainment', 'data_channel_is_bus',
       'data_channel_is_socmed', 'data_channel_is_tech',
       'data_channel_is_world', 'kw_min_min', 'kw_max_min', 'kw_avg_min',
       'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg',
       'kw_avg_avg', 'self_reference_min_shares', 'self_reference_max_shares',
       'self_reference_avg_sharess', 'weekday_is_monday', 'weekday_is_tuesday',
       'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',
       'is_weekend', 'LDA_00', 'LDA_02', 'global_subjectivity',
       'min_negative_polarity', 'title_subjectivity']]
y=data['shares']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
lrrr=LinearRegression()
modelreg=lp.fit(X_train,y_train)
ypredreg=lp.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,ypredreg)))
print(r2_score(y_test,ypredreg))


# In[ ]:




