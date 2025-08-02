# Import Libraries
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/data set/train.csv')
data.shape # check number of rows and columns

#from google.colab import drive
#drive.mount('/content/drive')

#Initial Exploration
data.columns 

data.head()

data.info() 

data['Dependents'].value_counts().reset_index()

data = data.drop('Loan_ID',axis = 1)

#Basic Data Exploration

data.describe()

data.describe(include = ['object'])

data.isna().sum()

cat_cols = data.dtypes =='object'
cat_cols = list(cat_cols[cat_cols].index)

num_cols = data.dtypes !='object'
num_cols = list(num_cols[num_cols].index)
cat_cols.remove('Loan_Status')

cat_cols

num_cols

data[cat_cols].head()

data[num_cols].head()

data['Loan_Status'].value_counts()

sns.countplot(data=data, x='Loan_Status')
plt.show()

target = 'Loan_Status'
data[target].value_counts()

plt.subplot(121)
sns.distplot(data["ApplicantIncome"])

plt.subplot(122)
data["ApplicantIncome"].plot.box(figsize=(16,5))
plt.show()

plt.subplot(121)
sns.distplot(np.log(data["ApplicantIncome"]))

plt.show()

#Slice this data by Education

data.boxplot(column='ApplicantIncome', by="Education", figsize=(8,5))
plt.suptitle("")
plt.show()

plt.subplot(121)
sns.distplot(data["CoapplicantIncome"])

plt.subplot(122)
data["CoapplicantIncome"].plot.box(figsize=(16,5))
plt.show()

#Relation between "Loan_Status" and "Income"

data.groupby("Loan_Status")['ApplicantIncome'].mean()

data.groupby("Loan_Status")['ApplicantIncome'].mean().plot.bar()
plt.ylabel("Mean Income of applicant")
plt.show()

# Simple Feature Engineering

bins=[0,2500,4000,6000, 8000, 10000, 20000, 40000, 81000]
group=['Low','Average','medium', 'h1', 'h2', 'h3', 'h4' , 'Very high']
data['Income_bin']= pd.cut(data['ApplicantIncome'],bins,labels=group)

data.head()

# Incomes

pd.crosstab(data["Income_bin"],data["Loan_Status"])

loan_percent = data.groupby("Income_bin")["Loan_Status"].value_counts(normalize=True).unstack() * 100


loan_percent.plot(kind='bar', figsize=(5,4))
plt.xlabel("Income Group")
plt.ylabel("Loan Approval %")
plt.title("Loan Status by Income Group")
plt.show()

bins=[0,1000,3000,42000]
group =['Low','Average','High']
data['CoapplicantIncome_bin']=pd.cut(data["CoapplicantIncome"],bins,labels=group)

pd.crosstab(data["CoapplicantIncome_bin"],data["Loan_Status"])

loan_percent_co=data.groupby("CoapplicantIncome_bin")["Loan_Status"].value_counts(normalize=True).unstack() * 100

loan_percent_co.plot(kind='bar', figsize=(5,4))

plt.xlabel("CoapplicantIncome")
plt.ylabel("Percentage")
plt.show()

## What's the problem here? Why co-applicant having low income is getting maximum loan approved?

data['CoapplicantIncome'].value_counts().head().reset_index()

data["TotalIncome"] = data["ApplicantIncome"] + data["CoapplicantIncome"]

bins = [0,3000,5000,8000,81000]
group = ['Low','Average','High','Very High']
data["TotalIncome_bin"] = pd.cut(data["TotalIncome"],bins,labels=group)

pd.crosstab(data["TotalIncome_bin"], data["Loan_Status"])

loan_percent_TotalIncome =data.groupby("TotalIncome_bin")["Loan_Status"].value_counts(normalize=True).unstack() * 100

loan_percent_TotalIncome.plot(kind='bar', figsize=(5,4))
plt.xlabel("TotalIncome")
plt.ylabel("Percentage")
plt.show()

data = data.drop(["Income_bin","CoapplicantIncome_bin","TotalIncome_bin"],axis=1)

# Loan Amount and Loan Term

data['Loan_Amount_Term'].value_counts().reset_index()

data['Loan_Amount_Term'] = (data['Loan_Amount_Term']/12).astype('float')

sns.countplot(x='Loan_Amount_Term', data=data)
plt.xlabel("Term in years")
plt.show()

plt.figure(figsize=(16,5))
plt.subplot(121)
sns.distplot(data['LoanAmount']);

plt.subplot(122)
sns.boxplot(data=data, x='Loan_Status', y = 'LoanAmount')

plt.show()

# Approximate calc: ignoring interest rates as we dont know that.
data['Loan_Amount_per_year']  = data['LoanAmount']/data['Loan_Amount_Term']

plt.figure(figsize=(16,5))
plt.subplot(121)
sns.distplot(data['Loan_Amount_per_year']);

plt.subplot(122)
sns.boxplot(data=data, x='Loan_Status', y = 'Loan_Amount_per_year')

plt.show()

# log transform
plt.figure(figsize=(16,5))
plt.subplot(121)
log_loanAmount = np.log(data['Loan_Amount_per_year'])
sns.distplot(log_loanAmount)

plt.subplot(122)
sns.boxplot(data=data, x='Loan_Status', y = log_loanAmount)

plt.show()

# Feature : Calculate the EMI based on the Loan Amount Per year.
data['EMI'] = data['Loan_Amount_per_year']*1000/12

#Feature : Able_to_pay_EMI
data['Able_to_pay_EMI'] = (data['TotalIncome']*0.1 > data['EMI']).astype('int')

data.head()

sns.countplot(x='Able_to_pay_EMI', data = data, hue = 'Loan_Status')
#Observation:
###There is 50% chance that you may get the loan approved if you cannot pay the EMI.
###But there, is a 72% chance that you may get the loan approved if you can pay the EMI.

# Dependents and Loan **approval**

data['Dependents'].value_counts().reset_index()

data['Dependents'].replace('3+',3,inplace=True)

data['Dependents'] = data['Dependents'].astype('float')

data.dtypes

sns.countplot(data =data, x = 'Dependents', hue = 'Loan_Status')

#Observations:

## No Dependents and 2 dependents will helps you get loan easily.

# Credit History vs Loan Approval

data['Credit_History'].value_counts()

sns.countplot(data =data, x = 'Credit_History', hue = 'Loan_Status')
#Observation:
## We can clearly see that the approval rate is 80% if your credit history is aligned with the guidlines.
## Hence this is the most important question that can be considered.

# Missing Values & Data Cleaning

data.isna().sum()

# Function to create a data frame with number and percentage of missing data in a data frame

def missing_to_df(df):
    total_missing_df = df.isnull().sum().sort_values(ascending =False)
    percent_missing_df = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
    missing_data_df = pd.concat([total_missing_df, percent_missing_df], axis=1, keys=['Total', 'Percent'])
    return missing_data_df

missing_df = missing_to_df(data)
missing_df[missing_df['Total'] > 0]

data.Credit_History.unique()

# Credit History=2 for nan/missing values.
data['Credit_History'] = data['Credit_History'].fillna(2)

# Self_Employed = 'Other' for nan
data.Self_Employed.unique()

data['Self_Employed'] = data['Self_Employed'].fillna('Other')

# median imputation for numerical columns.
from sklearn.impute import SimpleImputer

num_missing = ['EMI', 'Loan_Amount_per_year',  'LoanAmount',  'Loan_Amount_Term']

median_imputer = SimpleImputer(strategy = 'median')
for col in num_missing:
    data[col] = pd.DataFrame(median_imputer.fit_transform(pd.DataFrame(data[col])))

# Highest Freq imputation for some categorical columns.
cat_missing = ['Gender', 'Married','Dependents']

freq_imputer = SimpleImputer(strategy = 'most_frequent')
for col in cat_missing:
    data[col] = pd.DataFrame(freq_imputer.fit_transform(pd.DataFrame(data[col])))

missing_df = missing_to_df(data)
missing_df[missing_df['Total'] > 0]

s = (data.dtypes == 'object')
object_cols = list(s[s].index)
object_cols

# Encoding Categorical Variables

# Loan_Status
col='Loan_Status'
data[col].value_counts()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
col='Loan_Status'
data[col] = label_encoder.fit_transform(data[col])

data[col].value_counts()



#Gender
data['Gender'].value_counts()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
col='Gender'
data[col] = label_encoder.fit_transform(data[col])

data[col].value_counts()



# Married
data['Married'].value_counts()

label_encoder = LabelEncoder()
col='Married'
data[col] = label_encoder.fit_transform(data[col])
data[col].value_counts()

# Property Area

col='Property_Area'
data[col].value_counts().reset_index()

# !pip install category_encoders

from category_encoders import TargetEncoder

ta = TargetEncoder()
data[col] = ta.fit_transform(data[col], data['Loan_Status'])

col='Property_Area'
data[col].value_counts().reset_index()

# Education

col='Education'
data[col].value_counts()

label_encoder = LabelEncoder()
data[col] = label_encoder.fit_transform(data[col])
data[col].value_counts().reset_index()

# Self Employed

col='Self_Employed'
data[col].value_counts().reset_index()

#!pip install category_encoders

from category_encoders import TargetEncoder

ta = TargetEncoder()
data[col] = ta.fit_transform(data[col], data['Loan_Status'])
data[col].value_counts().reset_index()

s = data.dtypes == 'object'
object_cols = list(s[s].index)
object_cols

data.dtypes

# Correlation Coefficients


#PCC
plt.figure(figsize=(12, 12))
sns.heatmap(data.corr(method='pearson'), square=True,annot=True)

# srcc
plt.figure(figsize=(12, 12))
sns.heatmap(data.corr(method='spearman'), square=True,annot=True)

plt.scatter(data['Credit_History'], data['Loan_Status'])
plt.show()
#sometimes scatter plots can be misleading do to catgeorical nature of the data

sns.countplot(data =data, x = 'Credit_History', hue = 'Loan_Status')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
std_data = scaler.fit_transform(data)
std_data = pd.DataFrame(std_data, columns=data.columns)
std_data.head()

data.head()

