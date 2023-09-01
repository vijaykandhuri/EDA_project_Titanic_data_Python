#import pyforest
import pandas as pd



data=pd.read_csv('train.csv')
                    # showing train.csv data
#print(data)
                    # displays rows and columns
#print(data.shape)
                    # displays nulvalues present in csv file
#print(data.isna().sum())



                    #The describe() method returns description of the data in the DataFrame 
                    #if the DataFrame contains numerical data the description contains these information for each column
                    #count - The number of not-empty values and mean - The average (mean) value
                    #std - The standard deviation.
#print(data.describe())


                    # information about data type
                    
#print(data.info())
#print(data.dtypes)





                    # rename male female to 0 and 1  for that install pip install label encoder, pip install sklearn ,
                    # pip install scikit-learn and pip install --upgrade scikit-learn , to check import sklearn and print(sklearn.__version__)

import sklearn
#print(sklearn.__version__)
from sklearn import preprocessing



                    # lablel_encoder object knows how to understand word lables
label_encoder=preprocessing.LabelEncoder()
data['Sex']=label_encoder.fit_transform(data['Sex'])
#print(data['Sex'].value_counts())          # vulues returns as 0 and 1
#print(data.Sex)




                   # Dropping unwanted columns
data=data.drop(['Ticket','Cabin','Name'],axis=1)
#print(data)




                    #filling Age null values by average age
#print(data['Age'].median())
data['Age']=data['Age'].fillna(value=28)
#print(data)
#print(data['Age'].isna().sum())     # checking nullvalues in average

#print(data.isna().sum())            # showing Null values

#print(data['Embarked'].value_counts())  # Embarked s,c,q count





g=data.groupby('Survived')
print(g)
#print(g['Embarked'].value_counts())




                    #creating null values to Embarked data 
data['Embarked']=data['Embarked'].fillna(value='S')
#print(data)
#print(data.isna().sum())




                    # lablel_encoder object knows how to understand word lables
label_encoder=preprocessing.LabelEncoder()
                    # encodes lables in column Embarked
data['Embarked']=label_encoder.fit_transform(data['Embarked'])
                    # converts S,C,Q to 2,0,1
#print(data['Embarked'].value_counts())      
#print(data)





from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
                # showing countplot by using Embarked data with Survived people
sns.countplot(x=data['Embarked'],hue=data['Survived'])
#print(plt.show())

                # bar chat sibling and parch details
data.plot(x='Survived',y=['SibSp','Parch'],kind='bar')
#plt.show()

#print(data.head(5))
#print(data.tail(6))




                    # adding siblings and parch(no of Parents/Children Aboard) to family and removing sibling and parch after adding
data['family']=data['SibSp']+data['Parch']+1
data=data.drop(['SibSp','Parch'],axis=1)
data=data.drop('PassengerId',axis=1)
data=data.drop('Embarked',axis=1)
print(data)


data.to_csv("Titanic_data.csv")     # creating a csv file by using above details