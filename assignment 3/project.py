#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sagemaker
import boto3


from time import gmtime, strftime
import os

region = boto3.Session().region_name
smclient = boto3.Session().client("sagemaker")

role = sagemaker.get_execution_role()

bucket = "project-ml-pipeline"
prefix = "sagemaker/DEMO-project-ml-pipeline"


# In[2]:


role


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
import json
import sagemaker.amazon.common as smac


# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


test_data = pd.read_csv('credit_card_score.csv')
test_data


# In[6]:


test_data.describe(include='all')   
# Describe function gives us a discriptive data analysis table. This can be a good starting point as it can help view discrepencies in the columns. (df.describe() by default gives discriptive summary for only the numerical columns, we can use the df.describe(include='all') to view all comumns including the categorical ones.)


# In[7]:


test_data.shape


# In[8]:


test_data.describe()


# In[9]:


test_data.info()


# In[10]:


test_data.columns.values#.columns.values gives us the names of the columns in our data in form of an array as below. 


# In[11]:


test_data.isnull().sum().sort_values(ascending=False).to_frame().T


# In[12]:


cols= ['Age',
       'Annual_Income', 'Monthly_Inhand_Salary', 'Interest_Rate', 'Num_of_Loan',
       'Delay_from_due_date', 'Num_of_Delayed_Payment',
       'Num_Credit_Inquiries', 'Credit_Mix',
       'Outstanding_Debt', 'Credit_Utilization_Ratio',
       'Credit_History_Age', 'Payment_of_Min_Amount',
       'Total_EMI_per_month', 'Amount_invested_monthly',
       'Payment_Behaviour', 'Monthly_Balance', 'Credit_Score']

needed_data = test_data[cols]


# In[13]:


# We will drop the monthly_Inhand_Salary because it is highly correlated with Annual income and it has a lot of missing values. We cn alternatively use the Anuual_Salary column to fill the missing values for monthly_salary and use the monthly_salary column for analysis. 
needed_data = needed_data.drop(['Monthly_Inhand_Salary'], axis = 1) 
needed_data.isnull().sum()  # We use .isnull().sum() to see how many of the columns have null values. Once we have an overview we can then figure our ways to get rid of the null values.


# In[14]:


# We will convert the column ['Monthly_Balance'] to string datatype to limit the number of characters in string because some of the the columns have enteries which exceed the limit of float datatype, which we will later cnvert it into for analysis. 
needed_data['Monthly_Balance'] = needed_data['Monthly_Balance'].astype(str)  
# We use this column to restrict the characters to first 10 characters. This will cover all of the enteries for monthly balance as none of the monthly values exceed 10 digits.
needed_data['Monthly_Balance'] = needed_data['Monthly_Balance'].str[:10]  


# In[15]:


# First I will convert few columns from Object to Int or Float Data Type so that we can perform analysis on the data. 


needed_data['Annual_Income'] = needed_data['Annual_Income'].apply(lambda x: str(x).replace("_"," "))  
# Annual income column has a lot of unclean data with '_' at the end or starting of the string. We will replace them with space. 
needed_data['Num_of_Delayed_Payment'] = needed_data['Num_of_Delayed_Payment'].apply(lambda x: str(x).replace("None","0")) 
# Num_of_Delayed_Payment column has None, nan for null values. WE can replace them with '0'.
needed_data['Num_of_Delayed_Payment'] = needed_data['Num_of_Delayed_Payment'].apply(lambda x: str(x).replace("_"," "))    
# Num_of_Delayed_Payment column has a lot of unclean data with '_' at the end or starting of the string. We will replace them with space. 
needed_data['Num_of_Delayed_Payment'] = needed_data['Num_of_Delayed_Payment'].apply(lambda x: str(x).replace("nan","0")) 
# Similarly we can clean data for other columns. 
needed_data['Num_of_Delayed_Payment'] = pd.to_numeric(needed_data['Num_of_Delayed_Payment'])
needed_data['Outstanding_Debt'] = needed_data['Outstanding_Debt'].apply(lambda x: str(x).replace("_"," ")) 
needed_data['Outstanding_Debt'] = pd.to_numeric(needed_data['Outstanding_Debt'])                                           
# We will use the to_numeric() function to convert the values from strng to integer.
needed_data['Amount_invested_monthly'] = needed_data['Amount_invested_monthly'].apply(lambda x: str(x).replace("_"," "))
needed_data['Amount_invested_monthly'] = needed_data['Amount_invested_monthly'].apply(lambda x: str(x).replace("nan","0"))
needed_data['Amount_invested_monthly'] = needed_data['Amount_invested_monthly'].apply(lambda x: str(x).replace("None","0"))
needed_data['Amount_invested_monthly'] = pd.to_numeric(needed_data['Amount_invested_monthly'])
needed_data['Monthly_Balance'] = needed_data['Monthly_Balance'].apply(lambda x: str(x).replace("_"," "))
needed_data['Monthly_Balance'] = needed_data['Monthly_Balance'].apply(lambda x: str(x).replace("None","0"))
needed_data['Monthly_Balance'] = needed_data['Monthly_Balance'].apply(lambda x: str(x).replace("nan","0"))
needed_data['Monthly_Balance'] = pd.to_numeric(needed_data['Monthly_Balance'])
needed_data['Annual_Income'] = pd.to_numeric(needed_data['Annual_Income'])
needed_data['Age'] = needed_data['Age'].apply(lambda x: str(x).replace("_"," "))
needed_data['Age'] = pd.to_numeric(needed_data['Age'])
needed_data['Num_of_Loan'] = needed_data['Num_of_Loan'].apply(lambda x: str(x).replace("_"," "))
needed_data['Num_of_Loan'] = pd.to_numeric(needed_data['Num_of_Loan'])


# In[16]:


needed_data.info()


# In[ ]:





# In[17]:


# We will view the data to figure out the further analysis. 
needed_data.describe(include = 'all')                                                 


# In[18]:


# For the Credit_History_Age column we will need to find a way to convert the values into integer or float.
# We can either remove the repeating parts of the string such as 'Years and Months' or we can grab the first two characters from left of the string for Year and the and 2 characters from 7th position right end. Or we can use both for easier way. 
# I will use the second option. 

# We will reate a copy of our data to perform the further analysis
nd= needed_data.copy()
nd['Credit_History_Age'] = nd['Credit_History_Age'].replace('nan', np.nan).fillna(0)   # We will fill the missing string values with 0 so we can convert to integer type later on.
nd['Credit_History_Age'] = nd['Credit_History_Age'].astype(str)                        # To convert to string type. 
nd['History_Year'] = nd['Credit_History_Age'].str[ :2]                                 # To grab the first two characters of the string.                              
nd['History_Month'] = nd['Credit_History_Age'].str[ -9:-7]                             # We will use the negative values to count the numbers from the right side of the data. 


# In[19]:


nd['History_Year'] = pd.to_numeric(nd['History_Year'])                  # We use the pd.to_numeric to convert the column into integer type.
nd['History_Month'] = pd.to_numeric(nd['History_Month']) 
nd['Credit_History_Age'] = nd['History_Year']*12 + nd['History_Month']  # Now that we have months and years we can make a combined columns with total months for Credit_History_Age.
nd = nd.drop(['History_Year'], axis = 1)                                # Now that we have a consolidated column for Credit_History_Age we can drop the History_Year and History_month columns.
nd = nd.drop(['History_Month'], axis = 1)
nd


# In[20]:


# Now we have 4 columns left as objects that have categorical or ordinal data. 
{column: list(nd[column].unique()) for column in nd.select_dtypes('object').columns}  # We use this code to view diffferent inputs in each of the coategorical column. 
nd.info()


# In[21]:


# First we will identify which columns are nominal and which ones are ordinal.
# Credit_Mix and Credit_score are ordinal.So we can provide order to them manually. 
# Payment_of_Min_Amount, Payment_Behaviour can be considered as catecorical and we can use onehotencoder to provide dummy variables.

nd = nd[nd.Payment_Behaviour != '!@9#%8']                                  # After seeing the different values in the columns we find out that some cells of column '[Payment_Behaviour]' have '!@9#%8' as a conditional values  We will delete all the observations with _ as value. 
nd = nd[nd.Payment_of_Min_Amount != 'NM']                                  # Similarly some of the cells in ['Payment_of_Min_Amount'] column are listed as 'NM'. 

# We will now define functions for one hot encoding and ordinal encoding. So I will create 2 functions which we can call later for ordinal or categorical data. We will use onehot encoding for categorical data and ordinal encoding for Ordinal. 

def ordinal_encode(df, column, ordering):
    df = df.copy()
    df[column] = df[column].apply(lambda x: ordering.index(x))
    return df

def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column],dtype=int, prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df



# In[22]:


# We will create new array to define the order of the ordinal variables so that we can use the order in the ordinal encoder() function. We will do this for all three ordinal columns i.e.: 'Credit_Mix', 'Credit_Score', 'Payment_Behaviour'.
Credit_Mix_order = [ '_',
                    'Bad',
                    'Standard',
                    'Good']

Payment_Behaviour_order = ['Low_spent_Small_value_payments',
                          'Low_spent_Medium_value_payments',
                          'High_spent_Small_value_payments',
                          'Low_spent_Large_value_payments',
                           'High_spent_Medium_value_payments',
                           'High_spent_Large_value_payments']

Credit_Score_order = ['Poor',
                      'Standard',
                      'Good' ]


nd = ordinal_encode(nd, 'Credit_Mix', ordering = Credit_Mix_order)                       # We will call the ordical_encode() function for encoding different columns. Three inputs are the name of dataset, name of column and the order array. 
nd= ordinal_encode(nd, 'Payment_Behaviour', ordering = Payment_Behaviour_order )
nd= ordinal_encode(nd, 'Credit_Score', ordering = Credit_Score_order )


nd = onehot_encode(nd, 'Payment_of_Min_Amount', prefix = 'PMA' )                         # Payment_of_Min_Amount is a categorical variable so we use onehot encoding for this.
                                  


# In[23]:


cleaned_data = nd    # We will now visualize the cleaned table and name it as cleaned_data. 
cleaned_data.info()


# In[24]:


from matplotlib import colors
from matplotlib.colors import ListedColormap

sns.set(rc={"axes.facecolor":"#CEE8C8","figure.facecolor":"#FFFFF5"})
pallet = ["#FAD89F", "#53726D", "#424141", "#FFFFF5"]
cmap = colors.ListedColormap(["#ACC7EF", "#FAD89F", "#53726D", "#424141", "#FFFFF5"])
#Plotting following features
To_Plot = [ "Age", "Interest_Rate", "Delay_from_due_date", "Num_of_Delayed_Payment", "Credit_Mix"]
print("Reletive Plot Of Features")
plt.figure(figsize=(20,20))  
sns.pairplot(cleaned_data[To_Plot], hue= "Credit_Mix", palette= (["#ACC7EF", "#FAD89F", "#53726D", "#424141"]))
#Taking hue 
plt.show()


# In[25]:


# We can clearly see that Age has few outliiers so we will limit it to 100. 

q = cleaned_data['Interest_Rate'].quantile(0.95)
cleaned_data = cleaned_data[cleaned_data['Interest_Rate']<q]

cleaned_data['Num_of_Loan'] = cleaned_data['Num_of_Loan'].apply(lambda x: str(x).replace("-","0"))
cleaned_data['Num_of_Loan'] = cleaned_data['Num_of_Loan'].str[ :1] 
cleaned_data['Num_of_Loan'] = pd.to_numeric(cleaned_data['Num_of_Loan'])  

cleaned_data['Age'] = cleaned_data['Age'].apply(lambda x: str(x).replace("-"," "))
cleaned_data['Age'] = cleaned_data['Age'].str[ :3]   
cleaned_data['Age'] = pd.to_numeric(cleaned_data['Age'])  
cleaned_data = cleaned_data[(cleaned_data["Age"]> 18)]
cleaned_data = cleaned_data[(cleaned_data["Age"]< 100)]

cleaned_data['Delay_from_due_date'] = cleaned_data['Delay_from_due_date'].apply(lambda x: str(x).replace("-"," "))
cleaned_data['Delay_from_due_date'] = pd.to_numeric(cleaned_data['Delay_from_due_date'])  

cleaned_data['Num_of_Delayed_Payment'] = cleaned_data['Num_of_Delayed_Payment'].apply(lambda x: str(x).replace("-"," "))
cleaned_data['Num_of_Delayed_Payment'] = pd.to_numeric(cleaned_data['Num_of_Delayed_Payment'])  

q = cleaned_data['Num_of_Delayed_Payment'].quantile(0.95)
cleaned_data = cleaned_data[cleaned_data['Num_of_Delayed_Payment']<q]
cleaned_data = cleaned_data.fillna('0')

cleaned_data.describe()
cleaned_data.info()


# In[26]:


sns.set(rc={"axes.facecolor":"#CEE8C8","figure.facecolor":"#FFFFF5"})
pallet = ["#FAD89F", "#53726D", "#424141", "#FFFFF5"]
cmap = colors.ListedColormap(["#ACC7EF", "#FAD89F", "#53726D", "#424141", "#FFFFF5"])
#Plotting following features
To_Plot = [ "Age", "Interest_Rate", "Delay_from_due_date", "Num_of_Delayed_Payment", "Credit_Mix"]
print("Reletive Plot Of Features")
plt.figure(figsize=(20,20))  
sns.pairplot(cleaned_data[To_Plot], hue= "Credit_Mix", palette= (["#ACC7EF", "#FAD89F", "#53726D", "#424141"]))
#Taking hue 
plt.show()


# In[27]:


#sns.set(rc={"axes.facecolor":"#CEE8C8","figure.facecolor":"#FFFFF5"})
#pallet = ["#FAD89F", "#CEE8C8", "#53726D", "#FFFFF5"]
#cmap = colors.ListedColormap(["#ACC7EF", "#FAD89F", "#CEE8C8", "#53726D", "#FFFFF5"])
corrmat= cleaned_data.corr()
plt.figure(figsize=(20,20))  
sns.heatmap(corrmat,annot=True,  center=0)


# In[28]:


cleaned_data.sample(10)


# In[29]:


target_variable = "Credit_Score"

# Split the data into features (X) and target variable (y)
X = cleaned_data.drop(target_variable, axis=1)
y = cleaned_data[target_variable]

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1729)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=1729)

# Save the splits into CSV files
train_data = pd.concat([y_train, X_train], axis=1)
validation_data = pd.concat([y_validation, X_validation], axis=1)
test_data = pd.concat([y_test, X_test], axis=1)

train_data.to_csv("train.csv", index=False, header=False)
validation_data.to_csv("validation.csv", index=False, header=False)
test_data.to_csv("test.csv", index=False, header=False)

pd.read_csv("train.csv")


# In[30]:


boto3.Session().resource("s3").Bucket(bucket).Object(
    os.path.join(prefix, "train/train.csv")
).upload_file("train.csv")
boto3.Session().resource("s3").Bucket(bucket).Object(
    os.path.join(prefix, "validation/validation.csv")
).upload_file("validation.csv")


# In[68]:


from sagemaker.image_uris import retrieve

training_image = retrieve(framework="xgboost", region=region, version="1.7-1")

s3_input_train = "s3://{}/{}/train".format(bucket, prefix)
s3_input_validation = "s3://{}/{}/validation/".format(bucket, prefix)

training_job_definition = {
    "AlgorithmSpecification": {"TrainingImage": training_image, "TrainingInputMode": "File"},
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "CompressionType": "None",
            "ContentType": "csv",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_input_train,
                }
            },
        },
        {
            "ChannelName": "validation",
            "CompressionType": "None",
            "ContentType": "csv",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_input_validation,
                }
            },
        },
    ],
    "OutputDataConfig": {"S3OutputPath": "s3://{}/{}/output".format(bucket, prefix)},
    "ResourceConfig": {"InstanceCount": 1, "InstanceType": "ml.m4.xlarge", "VolumeSizeInGB": 10},
    "RoleArn": role,
    "StaticHyperParameters": {
    
        
        "verbosity": "0",
        "objective": "multi:softmax",
        "num_class": "10",
        
        
        
    },
    "StoppingCondition": {"MaxRuntimeInSeconds": 43200},
}


# In[69]:


from time import gmtime, strftime, sleep

tuning_job_name = "xgboost-project11-" + strftime("%d-%H-%M-%S", gmtime())

print(tuning_job_name)

tuning_job_config = {
    "ParameterRanges": {
        "CategoricalParameterRanges": [],
        "ContinuousParameterRanges": [
            {
                "MaxValue": "0.5",
                "MinValue": "0.1",
                "Name": "eta",
            },
            {
                "MaxValue": "5",
                "MinValue": "0",
                "Name": "gamma",
            },
            {
                "MaxValue": "120",
                "MinValue": "0",
                "Name": "min_child_weight",
            },
            {
                "MaxValue": "1",
                "MinValue": "0.5",
                "Name": "subsample",
            },
            {
                "MaxValue": "2",
                "MinValue": "0",
                "Name": "alpha",
            },
            
            
            
            
            
            
        ],
        "IntegerParameterRanges": [
            {
                "MaxValue": "10",
                "MinValue": "0",
                "Name": "max_depth",
            },
            {
                "MaxValue": "50",
                "MinValue": "1",
                "Name": "num_round",
            },
            
           
        ],
    },
    "ResourceLimits": {"MaxNumberOfTrainingJobs": 10, "MaxParallelTrainingJobs": 3},
    "Strategy": "Bayesian",
    "HyperParameterTuningJobObjective": {"MetricName": "validation:merror", "Type": "Minimize"},
}


# In[70]:


smclient.create_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuning_job_name,
    HyperParameterTuningJobConfig=tuning_job_config,
    TrainingJobDefinition=training_job_definition,
)


# In[71]:


smclient.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)[
    "HyperParameterTuningJobStatus"
]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




