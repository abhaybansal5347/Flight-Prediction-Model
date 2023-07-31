#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


train_data=pd.read_excel(r"Data_Train.xlsx")


# In[3]:


pd.set_option('display.max_columns',None)


# In[4]:


train_data.head()


# In[5]:


train_data.info()


# In[6]:


train_data["Duration"].value_counts()
#how many values are there for each time set;


# In[7]:


train_data.dropna(inplace=True)#dropping the nan values


# In[8]:


train_data.isnull().sum()


# In[9]:


##Exploratory Data Analysis


# In[10]:


train_data["Journey_day"]=pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.day


# In[11]:


train_data["Journey_month"]=pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.month


# In[12]:


train_data.head()


# In[13]:


#we can the drop the column that is of no use for now we would be dropping Date_of_Journey column
train_data.drop(["Date_of_Journey"],axis=1,inplace=True)


# In[14]:


#Departure time id when a plane leaves the gate
#Simlar to data_iof_hourney we can extract values from Dep_Time

#Extracting hours
train_data["Dep_hour"]=pd.to_datetime(train_data["Dep_Time"]).dt.hour
train_data["Dep_min"]=pd.to_datetime(train_data["Dep_Time"]).dt.minute
                                       


# In[15]:


train_data.drop(["Dep_Time"],axis=1,inplace=True)


# In[16]:


train_data.head()


# In[17]:


#Departure time id when a plane leaves the gate
#Simlar to data_iof_hourney we can extract values from Dep_Time

#Extracting hours
train_data["Arrival_hour"]=pd.to_datetime(train_data["Arrival_Time"]).dt.hour
train_data["Arrival_min"]=pd.to_datetime(train_data["Arrival_Time"]).dt.minute
                                       


# In[18]:


train_data.drop(["Arrival_Time"],axis=1,inplace=True)


# In[19]:


train_data.head()


# In[20]:


#Time taken by plane to reach destination is called Duration
#It is the differnce between departure time and arrival time
#Assigning and converting Duration column into list
duration=list(train_data["Duration"])
for i in range(len(duration)):
    if len(duration[i].split()) !=2:
        if "h" in duration[i]:
            duration[i]=duration[i].strip() +" 0m"
            #print(duration[i])
        else:
            duration[i]="0h "+duration[i]
            #print(duration[i])
duration_hours=[]
duration_mins=[]
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0])) # extract hours from duration
    duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))#extract mins from duration


# In[21]:


train_data["Duration_hours"]=duration_hours
train_data["Duration_mins"]=duration_mins
train_data.head()


# In[22]:


train_data.drop(["Duration"],axis=1,inplace=True)


# In[23]:


train_data.head()


# In[24]:


#Handling the categorical Data
#Ways to hadle categorical data
#1.Nominal Data- data are not in any order-->OneHotEncoder is used in this case
#2.Ordinal Data->data are in order->LabelEncoder is used in this case


# In[25]:


train_data["Airline"].value_counts()


# In[26]:


#From the graph we can see that jet Airways Bussinesss have the hihest Prices.
#Apart from the first Airline almost all are having similar median
#Airline VS Prices


sns.catplot(y="Price",x="Airline",data=train_data.sort_values("Price",ascending=False),kind="boxen",height=6,aspect=3)


# In[27]:


#As airlines is nomial categorical data we will perform OneHotEncoding
Airline=train_data[["Airline"]]
Airline=pd.get_dummies(Airline,drop_first=True)
Airline.head()


# In[28]:


train_data["Source"].value_counts()


# In[29]:


#Same Catplot For Source Vs Price AS we did in Airline vs Price
sns.catplot(y="Price",x="Source",data=train_data.sort_values("Price",ascending=False),kind="boxen",height=6,aspect=3)
plt.show()


# In[30]:


#As source is Nonimal Categorical data we will perform OneHotEncoding
#banglore is not there because when all the  values are 0  it means its representing banglore
Source=train_data[["Source"]]
Source=pd.get_dummies(Source,drop_first=True)
Source.head()


# In[31]:


#Same Catplot For Source Vs Price AS we did in Airline vs Price
sns.catplot(y="Price",x="Destination",data=train_data.sort_values("Price",ascending=False),kind="boxen",height=6,aspect=3)
plt.show()


# In[32]:


#As Destination is Nonimal Categorical data we will perform OneHotEncoding
Destination=train_data[["Destination"]]
Destination=pd.get_dummies(Destination,drop_first=True)
Destination.head()


# In[33]:


train_data[["Route"]]


# In[34]:


#Additional_Infor contains 90% no_info
#route and total_stops are related to each other so no use of route
train_data.drop(["Route","Additional_Info"],axis=1,inplace=True)


# In[35]:


train_data["Total_Stops"].value_counts()


# In[36]:


train_data.head()


# In[37]:


#as the no of data is ordinal categorical 
train_data.replace({"non-stop":0,"1 stop":1,"2 stops":2,"3 stops":3,"4 stops":4})

train_data.head()
# In[38]:


train_data.head()


# In[39]:


train_data.head()


# In[40]:


train_data.replace({"non-stop":0,"1 stop":1,"2 stops":2,"3 stops":3,"4 stops":4},inplace=True)
train_data.head()


# In[41]:


train_data.head()


# In[42]:


#Concatenate data frame ---> train_data+Airline+Source+Destination
data_train=pd.concat([train_data,Airline,Source,Destination],axis=1)
data_train.head()


# In[44]:


data_train.drop(["Airline","Source","Destination"],axis=1,inplace=True)


# In[45]:



data_train.head()


# In[46]:


data_train.shape


# In[47]:


#Why should we perform preprocessing for train and test data seperately why not together?
#Reason->We dont want our model to knoe the test data thats why->  Data Leakage accuracy will be decreased overfitting issue 


# TEST DATA

# In[48]:


test_data=pd.read_excel(r"Test_set.xlsx")


# In[49]:


test_data.head()


# In[50]:


# Preprocessing

print("Test data Info")
print("-"*75)
print(test_data.info())

print()
print()

print("Null values :")
print("-"*75)
test_data.dropna(inplace = True)
print(test_data.isnull().sum())

# EDA

# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)


# Categorical data

print("Airline")
print("-"*75)
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)

print()

print("Source")
print("-"*75)
print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first= True)

print()

print("Destination")
print("-"*75)
print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)

data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()

print("Shape of test data : ", data_test.shape)


# In[51]:


test_data.head()


# Feature Selection
# Finding out the best feature which will contribure and have good relaito  with target varible.
# Following are some of thr feeature selection methods
# 1. heatmap
# 2. feature_importance
# 3. SelectKBest
# 

# In[52]:


train_data


# In[53]:


data_train


# In[54]:


data_train.columns


# In[55]:


X=data_train.loc[:,['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()


# In[56]:


#dependent feature
y=data_train.iloc[:,1]
y.head()


# In[57]:


#find correlation between Indepenedent and Depenedednt attributes
plt.figure(figsize=(18,18))
sns.heatmap(train_data.corr(), annot= True,cmap = "RdYlGn")
plt.show()
#Remove two highly correlated features to remove duplicacy


# In[58]:


# Important features using ExtraTreeRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)


# In[59]:


print(selection.feature_importances_)


# In[60]:


#plot the graph of feature importacnes for better visualization
plt.figure(figsize=(12,8))
feet_importances=pd.Series(selection.feature_importances_,index=X.columns)
feet_importances.nlargest(20).plot(kind='barh')
plt.show()


# Fitting Model using Random Forest
# 
# Split dataset into train and test set in order to prediction wrt test
# 
# import model
# 
# fit the data
# 
# predict wrt X_test
# 
# In regression check RSME score
# 
# Plot the graph
# 
# 

# In[61]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[62]:


from sklearn.ensemble import RandomForestRegressor
reg_rf=RandomForestRegressor()
reg_rf.fit(X_train,y_train)


# In[63]:


y_pred=reg_rf.predict(X_test)


# In[64]:


reg_rf.score(X_train,y_train)


# In[65]:


reg_rf.score(X_test,y_test)


# In[66]:


sns.displot(y_test-y_pred)
plt.show()


# In[67]:


plt.scatter(y_test,y_pred,alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




