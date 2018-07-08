
# coding: utf-8

# # Bank Personal Loan Modelling
# 
# This is part-1 of a case study exercise based on what we learned in Supervised Learning Residency Class in July 2018.
# 
# ### Case study:
# 
# This case is about a bank (Thera Bank) which has a growing customer base. Majority of these customers are liability customers (depositors) with varying size of deposits. The number of customers who are also borrowers (asset customers) is quite small, and the bank is interested in expanding this base rapidly to bring in more loan business and in the process, earn more through the interest on loans. In particular, the management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio with minimal budget. The department wants to build a model that will help them identify the potential customers who have higher probability of purchasing the loan. This will increase the success ratio while at the same time reduce the cost of the campaign. The file Bank.xls contains data on 5000 customers. The data include customer demographic information (age, income, etc.), the customer's relationship with the bank (mortgage, securities account, etc.), and the customer response to the last personal loan campaign (Personal Loan). Among these 5000 customers, only 480 (= 9.6%) accepted the personal loan that was offered to them in the earlier campaign.
# 
# ### Steps Involved:
# 
# 1. Data pre-processing - Understand the data and treat missing values, outliers
# 
# 2. Understanding the attributes - Find relationship between different attributes (Independent variables) and choose carefully which all attributes have to be a part of the analysis and why 
# 
# 3. Model the data using Logistic regression
# 
# 4. Find the accuracy of the model using confusion matrix
# 
# 5. Use K - NN model [Hint: Try different values of k] and compare the accuracy of this model with that of Logistic regression
# 
# ### Data & Objectives:
# 
# File Bank_Personal_Loan_Modelling.xlsx is provided.
# 
# #### Project Objectives:
# While designing a new campaign, can we model the previous campaign's customer behavior to analyze what combination of parameters make a customer more likely to accept a personal loan?
# 
# There are several special products / facilities the bank offers like CD and security accounts, online services, credit cards, etc. Can we spot any association among these for finding cross-selling opportunities?
# 
# #### Data Description:
# ```
# ID                  Customer ID                                                                  UnsignedNumber
# Age                 Customer's age in years                                                      UnsignedNumber
# Experience          Years of professional experience                                             UnsignedNumber
# Income              Annual income of the customer                                                Float64
# ZIPCode 	        Home Address ZIP code                                                        Categorical
# Family              Family size of the customer                                                  UnsignedNumber
# CCAvg               Avg. spending on credit cards per month                                      Float64
# Education           Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional         Categorical
# Mortgage            Value of house mortgage if any.                                              Float64
# Personal Loan       Did this customer accept the personal loan offered in the last campaign?     Boolean(1,0)
# Securities Account  Does the customer have a securities account with the bank?                   Boolean(1,0)
# CD Account          Does the customer have a certificate of deposit (CD) account with the bank?  Boolean(1,0)
# Online              Does the customer use internet banking facilities?                           Boolean(1,0)
# CreditCard          Does the customer use a credit card issued by the bank?                      Boolean(1,0)
# ```
# *Note: Data is hypothetical*

# # Import Required Libraries Here

# In[853]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
get_ipython().magic(u'matplotlib inline')


# # Data Pre-Processing

# ## Read Excel File

# In[854]:


# Use read_excel method of Pandas Dataframe to read the data from the spreadsheet
# Note that in the SpreadSheet, there are two sheets:
# 1. README
# 2. Bank_Personal_Loan_Modelling Sheet. This is the sheet that has the data, hence we need to read only this sheet
# Kindly note for some reason the Family Members data type can't be changed to Int while reading, hence using float

bpl_data = pd.read_excel("Bank_Personal_Loan_Modelling.xlsx", 
                         sheet_name='Bank_Personal_Loan_Modelling', 
                         index_col=0,
                         dtype={'Age (in years)':np.int8,'Experience (in years)':np.int8, 
                                'Income (in K/month)':np.int32, 'ZIP Code':'category', 'Family members':np.float16,
                                'CCAvg':np.float32, 'Education':np.int8, 'Mortgage': np.int32, 
                                'Personal Loan':np.int8, 'Securities Account':np.int8, 
                                'CD Account':np.int8, 'Online':np.int8, 'CreditCard':np.int8},
                         names=['Age','Experience','Income','ZIPCode','Family','CCAvg',
                                  'Education','Mortgage','PersonalLoan','SecuritiesAccount','CDAccount',
                                  'NetBanking','CreditCard'])
bpl_data.head()


# ## Verify the Data Types
# 
# *Kindly Note that categorical data types are already labelled numerically, hence imported that with numeric data type*  
# *ZIPCode is the only one that is imported as categorical*

# In[855]:


# Verify the data types before we proceed further
bpl_data.dtypes


# ## Find the Missing Values

# In[856]:


# Is there any Missing Values ?
bpl_data.isnull().values.any()


# In[857]:


# Print the rows that has missing values
bpl_data[bpl_data.isnull().any(axis=1)]

# Observations
# Only Family Members Column has missing values and 18 rows has missing values


# ## Fill the missing values
# 
# *Kindly note only Family column has missing values*  
# *The total number of rows that contains missing values is 18*
#   
# Identify the Mean, Median of the Family Column   
# Fill the missing values with Mean or Median based on which ever is appropriate  

# In[858]:


# Describe the statistics in the dataframe
stats_desc = bpl_data.describe().transpose()
stats_desc


# In[859]:


# Number of Family Members will be always a whole number and it ranges from 1 to 4 in the dataset
# Fill the missing values in Family Members attribute with the median of the overall statistics. 
# Median will be good here as the number of possibilities is small
bpl_data['Family'].fillna((bpl_data['Family'].median()),inplace=True)
# Check whether there is any more missing values
bpl_data.isnull().values.any()


# In[860]:


# Pick randomly any missing value row and check whether the median value is populated 
bpl_data.loc[722]


# ## Understand the attributes of consumers who took Personal Loan

# In[861]:


# First lets filter the consumers who accepted personal loan
bpl_pld = bpl_data[bpl_data['PersonalLoan']==1]
# Lets understand the statistics of numeric independant variables
bpl_pld.describe().transpose()


# In[862]:


bpl_pld.groupby('Education').size().plot(kind='bar')


# In[863]:


bpl_pld.groupby('SecuritiesAccount').size().plot(kind='bar')


# In[864]:


bpl_pld.groupby('CDAccount').size().plot(kind='bar')


# In[865]:


bpl_pld.groupby('NetBanking').size().plot(kind='bar')


# In[866]:


bpl_pld.groupby('CreditCard').size().plot(kind='bar')


# ## Statistical Findings of the consumers who took the Personal Loan
# 
# Total consumers who accepted personal loan = 480  
# Their average experience level = 19.8  
# Their average age is 45  
# Their average income is 144.75K  
# Their average family size is 3  
# Their average spending on credit card is 3.9K  
# Their average mortgage is 100.8K  
# ~43% of them completed Advanced/Professional Education  
# ~87.5% of them don't have securities account  
# ~71% of them don't have CD Account  
# ~61% of them use Internet Banking facility  
# ~70% of them don't hold credit card  

# ## Find the Outliers in the data

# In[867]:


# Calculate InterQuartile Range to find outliers
# IQR = Q3 - Q1 (75% - 25%)
# Left Tail Whisker = Q1 - 1.5*IQR = 25% - 1.5*IQR
# Right Tail Whisker = Q3 - 1.5*IQR = 75% - 1.5*IQR
# Outliers: Values < Left Tail Whisker and Values > Right Tail Whisker
stats_desc['1.5IQR'] = stats_desc['75%'] - stats_desc['25%']
stats_desc['1.5IQR'] = stats_desc['1.5IQR']*1.5
stats_desc['lwhisker'] = stats_desc['25%'] - stats_desc['1.5IQR']
stats_desc['rwhisker'] = stats_desc['75%'] + stats_desc['1.5IQR']
stats_desc


# In[868]:


# Find & print the Outliers (Take into account only Continuous Variables)
bpl_data[(bpl_data['Age'] < stats_desc.loc['Age'].lwhisker) | 
    (bpl_data['Age'] > stats_desc.loc['Age'].rwhisker) | 
    (bpl_data['Experience'] < stats_desc.loc['Experience'].lwhisker) | 
    (bpl_data['Experience'] > stats_desc.loc['Experience'].rwhisker) | 
    (bpl_data['Income'] < stats_desc.loc['Income'].lwhisker) | 
    (bpl_data['Income'] > stats_desc.loc['Income'].rwhisker) |
    (bpl_data['Family'] < stats_desc.loc['Family'].lwhisker) | 
    (bpl_data['Family'] > stats_desc.loc['Family'].rwhisker) |      
    (bpl_data['CCAvg'] < stats_desc.loc['CCAvg'].lwhisker) | 
    (bpl_data['CCAvg'] > stats_desc.loc['CCAvg'].rwhisker) |
    (bpl_data['Mortgage'] < stats_desc.loc['Mortgage'].lwhisker) | 
    (bpl_data['Mortgage'] > stats_desc.loc['Mortgage'].rwhisker)].shape
# Total 602 Outlier records 


# ## Drop the outliers and print the shape
# 
# *Total Outliers = 602*  
# *Remaining Data = 4398 Rows*

# In[869]:


# Drop the outliers
bpl_nout = bpl_data.drop(bpl_data[(bpl_data['Age'] < stats_desc.loc['Age'].lwhisker) | 
    (bpl_data['Age'] > stats_desc.loc['Age'].rwhisker) | 
    (bpl_data['Experience'] < stats_desc.loc['Experience'].lwhisker) | 
    (bpl_data['Experience'] > stats_desc.loc['Experience'].rwhisker) | 
    (bpl_data['Income'] < stats_desc.loc['Income'].lwhisker) | 
    (bpl_data['Income'] > stats_desc.loc['Income'].rwhisker) |
    (bpl_data['Family'] < stats_desc.loc['Family'].lwhisker) | 
    (bpl_data['Family'] > stats_desc.loc['Family'].rwhisker) |      
    (bpl_data['CCAvg'] < stats_desc.loc['CCAvg'].lwhisker) | 
    (bpl_data['CCAvg'] > stats_desc.loc['CCAvg'].rwhisker) |
    (bpl_data['Mortgage'] < stats_desc.loc['Mortgage'].lwhisker) | 
    (bpl_data['Mortgage'] > stats_desc.loc['Mortgage'].rwhisker)].index)
bpl_nout.shape


# ## Understand the Correlation between the dependant and independant variables

# In[870]:


# Dependant Variable: PersonalLoan
# Let's identify the correlation between the independant variables in the full data
bpl_nout.corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
# Age -- Education = Very Strong Relationship
# Income -- CCAvg = Moderate  Relationship
# Income -- Mortgage = Weak Relationship
# Family -- Mortgage = Weak Relationship
# All other variables are fairly independant


# ### Relationship Observations
#   
# Dependant Variable: Personal Loan  
# Income, CCAvg, CDAccount are the threee variables which has greater strength of correlation with Personal Loan  
# Age & Experience is tightly coupled, hence it's ok to drop one of them in future calculations  
# Income & CCAvg have moderate strength of correlation

# ## Remove the variables whose correlation is between -0.1 to +0.1, as these variables will not impact prediction of Personal Loan
# 
# *Note: The reason for this is that these variables whose value close to zero doesn't have any relationship with Personal Loan, hence the change in those variable values will certainly not impact the prediction

# In[871]:


# Drop the variables whose correlation is between -0.1 to +0.1, as these variables will not impact prediction of
# Personal Loan
bpl_corr = bpl_nout.corr()['PersonalLoan'].drop(index=['PersonalLoan','Education','SecuritiesAccount',
                                                       'CDAccount','NetBanking','CreditCard'])
bpl_corr = bpl_nout.drop(columns=bpl_corr[(bpl_corr > -0.1) & (bpl_corr < 0.1)].index.values)
bpl_corr.corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


# ### Observations
# 
# *It's very clear that our manual finding matches with the data after removing the low correlated values*

# ## Understand the Variances of the attributes and remove variables with low variances

# In[872]:


# Let's use this only for Continous Variables, For Categorical Variables use Chi-Square Test
bpl_corr.var()


# In[873]:


bpl_corr.var() < 0.1


# ### Observations
# No continuous variables has low variance, hence nothing to remove

# ### Perform Chi-Square test to perform independance test among the categroical variables
# 
# Remaining Categorical Variables are:  
# -Education  
# -NetBanking  
# -CreditCard  
# -ZIPCode  
# -SecuritiesAccount  
# -CDAccount  
# -PersonalLoan  

# In[874]:


stats.chisquare(bpl_corr["Education"].value_counts())
#The p-value < 0.05 hence we conclude that Atleast one of the proportions differs


# In[875]:


stats.chisquare(bpl_corr["NetBanking"].value_counts())


# In[876]:


stats.chisquare(bpl_corr["CreditCard"].value_counts())


# In[877]:


stats.chisquare(bpl_corr["ZIPCode"].value_counts())


# In[878]:


# Goodness of Fit Test between 2 categorical variables

# H0: The two categorical variables are independent
# Ha: The two categorical variables are dependent

# Creating contingency table
cont = pd.crosstab(bpl_corr['Education'],bpl_corr['PersonalLoan'])
cont


# In[879]:


stats.chi2_contingency(cont)

#The p-value < 0.05 hence I conclude that the 2 categorical variables are dependent


# In[880]:


# Goodness of Fit Test between 2 categorical variables

# H0: The two categorical variables are independent
# Ha: The two categorical variables are dependent

# Creating contingency table
cont = pd.crosstab(bpl_corr['NetBanking'],bpl_corr['PersonalLoan'])
stats.chi2_contingency(cont)

#The p-value > 0.05 hence I conclude that the 2 categorical variables are indpendent 


# In[881]:


# Goodness of Fit Test between 2 categorical variables

# H0: The two categorical variables are independent
# Ha: The two categorical variables are dependent

# Creating contingency table
cont = pd.crosstab(bpl_corr['CreditCard'],bpl_corr['PersonalLoan'])
stats.chi2_contingency(cont)

#The p-value > 0.05 hence I conclude that the 2 categorical variables are indpendent 


# In[882]:


# Goodness of Fit Test between 2 categorical variables

# H0: The two categorical variables are independent
# Ha: The two categorical variables are dependent

# Creating contingency table
# To perform this test, 1st ZIPCode need to be converted to Numerical Labels similar to PersonalLoan
bpl_corr['ZIPCode'] = bpl_corr[bpl_corr.select_dtypes(include='category').columns].apply(
                                                        LabelEncoder().fit_transform)['ZIPCode']
cont = pd.crosstab(bpl_corr['ZIPCode'],bpl_corr['PersonalLoan'])
stats.chi2_contingency(cont)

#The p-value > 0.05 hence I conclude that the 2 categorical variables are indpendent 


# In[883]:


# Goodness of Fit Test between 2 categorical variables

# H0: The two categorical variables are independent
# Ha: The two categorical variables are dependent

# Creating contingency table
cont = pd.crosstab(bpl_corr['SecuritiesAccount'],bpl_corr['PersonalLoan'])
stats.chi2_contingency(cont)

#The p-value > 0.05 hence I conclude that the 2 categorical variables are independent 


# In[884]:


# Goodness of Fit Test between 2 categorical variables

# H0: The two categorical variables are independent
# Ha: The two categorical variables are dependent

# Creating contingency table
cont = pd.crosstab(bpl_corr['CDAccount'],bpl_corr['PersonalLoan'])
stats.chi2_contingency(cont)

#The p-value < 0.05 hence I conclude that the 2 categorical variables are dependent 


# ### Conclusions
# 
# Based on Chi-Square Test, Personal Loan is dependant only on Education & CDAccount. Hence lets drop the rest of the categorical 

# In[885]:


bpl_corr = bpl_corr.drop(['NetBanking','CreditCard','SecuritiesAccount','ZIPCode'],axis=1)


# ## Split the dependant and independant variables, Check the Distribution

# In[886]:


# Lets split dependant & indepenadant variables
dv = bpl_corr['PersonalLoan']
iv = bpl_corr.drop('PersonalLoan',axis=1)

# Scatter Plot of Independant Variables
sns.pairplot(iv, diag_kind='kde')


# ### Observations
# 
# The continuous variables Income & CCAvg distribution is not normal, hence we can try normalizing it to improve accuracy    

# ## Model the data using Logistic Regression and KNN

# ### Model without Normalizaing the distribution

# In[887]:


# Use the data where consumers accepted Personal Loan for training Logistic Regression
# It's important to ensure that the train/test data has equal proportion of consumers who accepted personal Loan
iv_train,iv_test,dv_train,dv_test=train_test_split(iv, dv, train_size=0.8, random_state=1,stratify=y)
iv_train.shape


# In[888]:


# As we need to use the previous campaign's customer behavior, let's ensure the split of accepted personal loan's 
# is evenly distributed between train & test set
print("Distribution Percentage of previous campaign data\n{}".format((dv_train.value_counts()/dv.value_counts())))


# ### Logistic Regression

# In[889]:


model = LogisticRegression()
model.fit(iv_train, dv_train)

# Train Set Scores
dv_predict = model.predict(iv_train)
model_score = model.score(iv_train, dv_train)
print(model_score)
print(confusion_matrix(dv_train, dv_predict))
print("Train Set: Accuracy Score = {}, F1 Score = {}\n\n".format(accuracy_score(dv_train,dv_predict),
                                                                 f1_score(dv_train,dv_predict)))

# Test Set Scores
dv_predict = model.predict(iv_test)
model_score = model.score(iv_test, dv_test)
print(model_score)
print(confusion_matrix(dv_test, dv_predict))
print("Test Set: Accuracy Score = {}, F1 Score = {}".format(accuracy_score(dv_test,dv_predict),
                                                            f1_score(dv_test,dv_predict)))


# ### Observations
# 
# 1. We get Accurancy of 96% only & F1 Score of 63%

# ### KNN Model

# In[890]:


# Lets try with KNN Model
kList = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 31, 33, 35, 37, 39, 41]
scores = []
# Perform Cross Validation
for k in kList:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores.append(cross_val_score(knn, iv_train, dv_train, cv=10, scoring='accuracy').mean())
scores


# In[891]:


scores_err = [1 - x for x in scores]
# plot misclassification error vs k
plt.plot(kList, scores_err)
plt.xlabel('Number of Neighbors k')
plt.ylabel('Misclassification Error')
plt.show()


# ### Observations
# 
# The accuracy of KNN is almost same as Logistic Regression ~ 96% with Optimal Neighbours as 1

# ## Normalize the Distribution and Rerun the Models

# In[892]:


# Reset Independant Variable back to Original Values
iv = bpl_corr.drop(['PersonalLoan'],axis=1)
plt.hist(iv['Income'])


# In[893]:


plt.hist(iv['CCAvg'])


# In[894]:


# Use Z-Score to transform
iv['Income'] = (iv['Income']-iv['Income'].mean())/iv['Income'].std()
iv['CCAvg'] = (iv['CCAvg']-iv['CCAvg'].mean())/iv['CCAvg'].std()


# In[895]:


plt.hist(iv['Income'])


# In[896]:


plt.hist(iv['CCAvg'])


# ### Logistic Regression

# In[897]:


iv_train,iv_test,dv_train,dv_test=train_test_split(iv, dv, train_size=0.8, random_state=1,stratify=dv)
model = LogisticRegression()
model.fit(iv_train, dv_train)

# Train Set Scores
dv_predict = model.predict(iv_train)
model_score = model.score(iv_train, dv_train)
print(model_score)
print(confusion_matrix(dv_train, dv_predict))
print("Train Set: Accuracy Score = {}, F1 Score = {}\n\n".format(accuracy_score(dv_train,dv_predict),
                                                                 f1_score(dv_train,dv_predict)))

# Test Set Scores
dv_predict = model.predict(iv_test)
model_score = model.score(iv_test, dv_test)
print(model_score)
print(confusion_matrix(dv_test, dv_predict))
print("Test Set: Accuracy Score = {}, F1 Score = {}".format(accuracy_score(dv_test,dv_predict),
                                                            f1_score(dv_test,dv_predict)))


# ### Observations
# 
# No big change in Accuracy after using zscore. 
# Accuracy is 96.4%

# ### KNN Model

# In[898]:


# Lets try with KNN Model
kList = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 31]
scores = []
# Perform Cross Validation
for k in kList:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores.append(cross_val_score(knn, iv_train, dv_train, cv=10, scoring='accuracy').mean())
scores


# In[899]:


scores_err = [1 - x for x in scores]
# plot misclassification error vs k
plt.plot(kList, scores_err)
plt.xlabel('Number of Neighbors k')
plt.ylabel('Misclassification Error')
plt.show()


# ### Observations:
# 
# Optimal Neighbour = 5
# Accuracy: 97.7%
# 
# KNN Yield better accuracy rate with z-score based distribution  
# With Optimial Neighbour of 5, the misclassification error is at the lowest
#   
# Also you can observe below the accuracy and F1 Score with KNN Model. This is comparitively much better than the Logistic Regression scores.

# In[900]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(iv_train, dv_train)    
dv_predict = knn.predict(iv_test)
print(confusion_matrix(dv_test, dv_predict))
print("Test Set: Accuracy Score = {}, F1 Score = {}".format(accuracy_score(dv_test,dv_predict),
                                                            f1_score(dv_test,dv_predict)))
dv_predict = knn.predict(iv_train)
print(confusion_matrix(dv_train, dv_predict))
print("Train Set: Accuracy Score = {}, F1 Score = {}".format(accuracy_score(dv_train,dv_predict),
                                                            f1_score(dv_train,dv_predict)))


# ## Other Distribution Models (BoxCox, ...)

# ### BoxCox

# In[901]:


# Reset Independant Variable back to Original Values
iv = bpl_corr.drop('PersonalLoan',axis=1)

# Transform income using boxcox
# We can't use boxcox with CCAvg as it has zero values
iv['Income'] = stats.boxcox(iv['Income'],0)
plt.hist(no_corr['Income'])


# ### Logistic Regression

# In[902]:


iv_train,iv_test,dv_train,dv_test=train_test_split(iv, dv, train_size=0.8, random_state=1,stratify=dv)
model = LogisticRegression()
model.fit(iv_train, dv_train)

# Train Set Scores
dv_predict = model.predict(iv_train)
model_score = model.score(iv_train, dv_train)
print(model_score)
print(confusion_matrix(dv_train, dv_predict))
print("Train Set: Accuracy Score = {}, F1 Score = {}\n\n".format(accuracy_score(dv_train,dv_predict),
                                                                 f1_score(dv_train,dv_predict)))

# Test Set Scores
dv_predict = model.predict(iv_test)
model_score = model.score(iv_test, dv_test)
print(model_score)
print(confusion_matrix(dv_test, dv_predict))
print("Test Set: Accuracy Score = {}, F1 Score = {}".format(accuracy_score(dv_test,dv_predict),
                                                            f1_score(dv_test,dv_predict)))


# ### KNN Model

# In[903]:


# Lets try with KNN Model
kList = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 31]
scores = []
# Perform Cross Validation
for k in kList:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores.append(cross_val_score(knn, iv_train, dv_train, cv=10, scoring='accuracy').mean())
scores


# In[904]:


scores_err = [1 - x for x in scores]
# plot misclassification error vs k
plt.plot(kList, scores_err)
plt.xlabel('Number of Neighbors k')
plt.ylabel('Misclassification Error')
plt.show()


# In[905]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(iv_train, dv_train)    
dv_predict = knn.predict(iv_test)
print(confusion_matrix(dv_test, dv_predict))
print("Test Set: Accuracy Score = {}, F1 Score = {}".format(accuracy_score(dv_test,dv_predict),
                                                            f1_score(dv_test,dv_predict)))
dv_predict = knn.predict(iv_train)
print(confusion_matrix(dv_train, dv_predict))
print("Train Set: Accuracy Score = {}, F1 Score = {}".format(accuracy_score(dv_train,dv_predict),
                                                            f1_score(dv_train,dv_predict)))


# ### Observations
# Logistic Regression Yielded Poor Accuracy with ~95%  
# KNN Yielded better accuracy with 97.8 & Optimal Neighbour of 3, also higher F1 Score

# ### p-Value based

# In[906]:


# Reset Independant Variable back to Original Values
iv = bpl_corr.drop('PersonalLoan',axis=1)

# Transform
iv['Income'] = stats.norm.cdf(iv['Income'],iv['Income'].mean(),iv['Income'].std())
iv['CCAvg'] = stats.norm.cdf(iv['CCAvg'],iv['CCAvg'].mean(),iv['CCAvg'].std())


# ### Logistic Regression

# In[907]:


iv_train,iv_test,dv_train,dv_test=train_test_split(iv, dv, train_size=0.8, random_state=1,stratify=dv)
model = LogisticRegression()
model.fit(iv_train, dv_train)

# Train Set Scores
dv_predict = model.predict(iv_train)
model_score = model.score(iv_train, dv_train)
print(model_score)
print(confusion_matrix(dv_train, dv_predict))
print("Train Set: Accuracy Score = {}, F1 Score = {}\n\n".format(accuracy_score(dv_train,dv_predict),
                                                                 f1_score(dv_train,dv_predict)))

# Test Set Scores
dv_predict = model.predict(iv_test)
model_score = model.score(iv_test, dv_test)
print(model_score)
print(confusion_matrix(dv_test, dv_predict))
print("Test Set: Accuracy Score = {}, F1 Score = {}".format(accuracy_score(dv_test,dv_predict),
                                                            f1_score(dv_test,dv_predict)))


# ### KNN Model

# In[908]:


# Lets try with KNN Model
kList = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 31]
scores = []
# Perform Cross Validation
for k in kList:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores.append(cross_val_score(knn, iv_train, dv_train, cv=10, scoring='accuracy').mean())
scores


# In[909]:


scores_err = [1 - x for x in scores]
# plot misclassification error vs k
plt.plot(kList, scores_err)
plt.xlabel('Number of Neighbors k')
plt.ylabel('Misclassification Error')
plt.show()


# In[910]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(iv_train, dv_train)    
dv_predict = knn.predict(iv_test)
print(confusion_matrix(dv_test, dv_predict))
print("Test Set: Accuracy Score = {}, F1 Score = {}".format(accuracy_score(dv_test,dv_predict),
                                                            f1_score(dv_test,dv_predict)))
dv_predict = knn.predict(iv_train)
print(confusion_matrix(dv_train, dv_predict))
print("Train Set: Accuracy Score = {}, F1 Score = {}".format(accuracy_score(dv_train,dv_predict),
                                                            f1_score(dv_train,dv_predict)))


# ### Observations
# With p-value based distribution also KNN Model yielded better accuracy of 97.3% with optimal neighbour of 3

# # Conclusion
# 
# With Z Score & removing outliers the accuracy considerably improved  
# KNN Model gave the better accuracy & the max of 98%  
# Logistic Regression accuracy is close to 97% max, but the F1 Score is considerably low here.  
# Hence the recommendation here is to use KNN Model  
# 
# The Personal Loan is dependant on Income, CreditCard Average Spending & Education
# 
# Based on the previous campaign, it's found that the consumers with the following attributes are more likely to   
# accept the personal loan:
# 1. Higher income, Higher credit card spend were most likely to accept the personal loand
# 2. Consumers who have Education with Advanced/Professional has higher probability

# # Thank You

# # Check with Linear Regression

# In[911]:


# Reset Independant Variable back to Original Values
iv = bpl_corr.drop('PersonalLoan',axis=1)
iv_train,iv_test,dv_train,dv_test=train_test_split(iv, dv, train_size=0.8, random_state=1,stratify=dv)


# In[912]:


# Create the Regression Model
regression_model = LinearRegression()
# Fit the data into the Regression Model
regression_model.fit(iv_train, dv_train)
# Score for Test Set
dv_pred = regression_model.predict(iv_test)
plt.scatter(dv_test,dv_pred)


# In[913]:


RSq = regression_model.score(iv_test, dv_test)
print("R-Square = {}".format(RSq))
n = len(X_test)
k = len(X_test.columns)
ARSq = 1 - (((1-RSq)*(n-1))/(n-k-1))
print("Adj. R-Square = {}".format(ARSq))


# ### Observations
# 
# Linear Regression Model doesn't yield accuracies. Linear Regression will not perform good for Classification
