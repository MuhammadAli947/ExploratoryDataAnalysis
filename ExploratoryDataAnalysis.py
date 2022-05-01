import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'

df=pd.read_csv(path)
df.head()

# list the data types for each column
print(df.dtypes)
# Engine size as potential predictor variable of price
sns.regplot(x="engine-size",y="price",data=df)
plt.ylim(0,)

# We can examine the correlation between 'engine-size' and 'price' and see it's approximately 0.87
df[["engine-size","price"]].corr()

# Highway mpg is a potential predictor variable of price\
sns.regplot(x="highway-mpg",y="price",data=df)

#  We can examine the correlation between 'highway-mpg' and 'price' and see it's approximately -0.704
df[['highway-mpg','price']].corr()

sns.regplot(x="peak-rpm",y="price",data=df)
df[["peak-rpm","price"]].corr()

#  Categorical variables
#  look at the relationship between "body-style" and "price"
sns.boxplot(x="body-style",y="price", data=df)

sns.boxplot("engine-location","price",data=df)

#  examine "drive-wheels" and "price".
# drive-wheels
sns.boxplot(x="drive-wheels",y="price",data=df)

#3. Descriptive Statistical Analysis
df.describe()
df.describe(include=['object'])

# Value counts
df['drive-wheels'].count()
df['drive-wheels'].value_counts().to_frame()
#Let's repeat the above steps but save the results to the dataframe "drive_wheels_counts" and rename the column
#'drive-wheels' to 'value_counts'.

drive_wheel_counts=df['drive-wheels'].value_counts().to_frame()
drive_wheel_counts.rename(columns={'drive-wheels':'value_counts'},inplace=True)
drive_wheel_counts

# We can repeat the above process for the variable 'engine-location'
engine_loc_counts=df['engine-location'].value_counts().to_frame()

engine_loc_counts.rename(columns={'engine-location':'value_counts'},inplace=True)
engine_loc_counts.index.name='engine-location'
engine_loc_counts.head(10)

#   4. Basics of Grouping
df['drive-wheels'].unique()

#  We can select the columns 'drive-wheels', 'body-style' and 'price', then assign it to the variable "df_group_one".
df_group_one=df[['drive-wheels','body-style','price']]
df_group_one=df_group_one.groupby(['drive-wheels'],as_index=False).mean()

# grouping results
df_gptest=df[['drive-wheels','body-style','price']]
grp_test1=df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grp_test1

#  we will leave the drive-wheel variable as the rows of the table, and pivot body-style to become the columns of the table:
grouped_pivot= grp_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot

grouped_pivot=grouped_pivot.fillna(0) #fill missing values with 0
grouped_pivot

import matplotlib.pyplot as plt
%matplotlib inline

#  use the grouped results
#   Variables: Drive Wheels and Body Style vs Price
plt.pcolor(grouped_pivot,cmap='RdBu')
plt.colorbar()
plt.show()

#  The default labels convey no useful information to us. Let's change that:
fig, ax=plt.subplots()
im= ax.pcolor(grouped_pivot, cmap='RdBu')

#  label names
row_labels= grouped_pivot.columns.levels[1]
col_labels=grouped_pivot.index

#  move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1])+ 0.5 , minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0])+ 0.5 , minor=False)

# insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#  #rotate label if too long
plt.xticks(rotation=90)
fig.colorbar(im)
plt.show()

#  5. Correlation and Causation
df.corr()

from scipy import stats
#  Let's calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'.
#  Width vs Price

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The pearson Correlation Coefficient is",pearson_coef, "with a P-value of p =",p_value)

# Curb-weight vs Price
pearson_coef , p_value =stats.pearsonr(df['curb-weight'], df['price'])
print("The pearson Correlation coefficient is",pearson_coef, "with a P-Value of P=",p_value)

#  Engine-size vs Price
pearson_coef ,p_value=stats.pearsonr(df['engine-size'], df['price'])
print("The pearson Correlation Coefficient is",pearson_coef, "with a p-value of P=",p_value)

# Bore vs Price
pearson_coef ,p_value=stats.pearsonr(df['bore'], df['price'])
print("The pearson Correlation Coefficient is",pearson_coef, "with a p-value of P=",p_value)

#  City-mpg vs Price
pearson_coef ,p_value=stats.pearsonr(df['city-mpg'], df['price'])
print("The pearson Correlation Coefficient is",pearson_coef, "with a p-value of P=",p_value)

# Highway-mpg vs Price
pearson_coef ,p_value=stats.pearsonr(df['highway-mpg'], df['price'])
print("The pearson Correlation Coefficient is",pearson_coef, "with a p-value of P=",p_value)

# ANOVA
# ANOVA: Analysis of Variance
# Drive Wheels
grouped_test2=df_gptest[['drive-wheels','price']].groupby(['drive-wheels'])

grouped_test2.head(2)
grouped_test2.get_group('4wd')['price']

# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'],grouped_test2.get_group('rwd')['price'],grouped_test2.get_group('4wd')['price'])
print("Anova Results: F=", f_val, ", p=",p_val)

# Separately: fwd and rwd
f_val, p_val=stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])
print("Anova results: F=", f_val, ", P=",p_val)

#  4wd and rwd
f_val,p_val=stats.f_oneway(grouped_test2.get_group('4wd')['price'],grouped_test2.get_group('rwd')['price'])
print("Anova results: F=",f_val,", P=",p_val)

# 4wd and fwd
f_val,p_val=stats.f_oneway(grouped_test2.get_group('4wd')['price'],grouped_test2.get_group('fwd')['price'])
print("Anova results: F=",f_val,", P=",p_val)

#We now have a better idea of what our data looks like and which variables are important to take into account when predicting the car price. We have narrowed it down to the following variables:

#Continuous numerical variables:

#Length
#Width
#Curb-weight
#Engine-size
#Horsepower
#City-mpg
#Highway-mpg
#Wheel-base
#Bore

# Categorical Variable
# Drive-wheels



