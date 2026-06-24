import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle


st.title('Passenger Survival Chance in Titanic Journey')

pclass=st.slider('Enter the Passenger class for the user',1,3)
sex=st.selectbox('Enter the Passenger Gender',['male','female'])
sibsp=st.slider('Enter the Passenger total number of Sibling and Spouse',1,8) 
parch=st.slider('Enter the Passenger total number of Parent and Child',0,6)
fare=st.number_input('Enter the Fare of the passenger')      
embarked=st.selectbox('Enter the Passenger station from where they started the journey',['Southamton','Chebourg','Queenstown'])

data=pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}])

model=load_model('titanic_model.h5')

with open ('le.pkl','rb') as file:
  label=pickle.load(file)

with open ('ohe.pkl','rb') as file:
  onehot=pickle.load(file)

with open ('ss.pkl','rb') as file:
  scaler=pickle.load(file)

data['Sex']=label.transform(data['Sex'])
Embarked=onehot.transform(data[['Embarked']])

Embarked=pd.DataFrame(Embarked,columns=onehot.get_feature_names_out())

data=pd.concat([data.drop(columns=['Embarked']), Embarked], axis=1)

data[['Pclass', 'SibSp', 'Parch', 'Fare']]=scaler.transform(data[['Pclass', 'SibSp', 'Parch', 'Fare']])

y=model.predict(data)

y=y[0][0]

def Chance(y):
  if y>0.5:
    return 'The Passenger will Survive the Titanic Journey'
  else:
    return 'The Passenger will not survive the Titanic Journey'

if st.button('Predict Survival Chance'):
  st.write('Probablity for Passenger Survival Chance',y) 
  st.write(Chance(y))

  
