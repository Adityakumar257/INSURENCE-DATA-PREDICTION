import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#importing the dataset
dataset=pd.read_csv('insurance(1).csv')
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
X["sex"]=le.fit_transform(X["sex"])

lo=LabelEncoder()
X["smoker"]=lo.fit_transform(X["smoker"])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[("Encoding",OneHotEncoder(),[5])],remainder = "passthrough")

X = ct.fit_transform(X)

from sklearn.preprocessing import StandardScaler
sc =  StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor
Regressor=RandomForestRegressor(random_state=0)
Regressor.fit(X_train,y_train)
y_pred=Regressor.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

from joblib import dump,load
dump(le,"gender.joblib")
dump(lo,"smoker.joblib")
dump(sc,"scaling.joblib")
dump(ct,"onehot.joblib")
dump(Regressor,"Regressor.joblib")

gender = load("gender.joblib")
smoker = load("smoker.joblib")
onehot = load("onehot.joblib")
scaling = load("scaling.joblib")
Regressor = load("Regressor.joblib")

def disp():
        
    new=pd.DataFrame({"age":[int(a1.get())],"sex":[(a2.get())],"bmi":[float(a3.get())],"children":[int(a4.get())],"smoker":[(a5.get())],"region":[(a6.get())]})
    
    new["sex"]=gender.transform(new["sex"])
    new["smoker"]=smoker.transform(new["smoker"])

    new = ct.transform(new)
    new = sc.transform(new)

    ans=Regressor.predict(new)
    Final_l.config(text=str(ans))

from tkinter import *
root= Tk()
root.geometry("2000x2000")
#bg = PhotoImage(file = "bg.png")
#label0= Label( root, image = bg)
#label0.place(x = 0, y = 0)
a1=StringVar()

a2=StringVar()

a3=StringVar()

a4=StringVar()

a5=StringVar()

a6=StringVar()

l = Label(root,text = "Insurance Charges Predictor",fg="black",bg="skyblue" ,font=("Pacifico","40")).place(x=400,y=30)

label1 = Label(root,text = "What is your age : ",fg="black",font=("Slabo 27px","23")).place(x= 70,y=180)
age=Entry(root,textvariable=a1,width=23).place(x=550,y=180,height=30)

label2 = Label(root,text = "What is your gender : ",fg="black",font=("Slabo 27px","23")).place(x= 70,y=270)
sex=Entry(root,textvariable=a2,width=23).place(x=550,y=270,height=30)

label3 = Label(root,text = "What is your BMI : ",fg="black",font=("Slabo 27px","23")).place(x= 70,y=360)
bmi=Entry(root,textvariable=a3,width=23).place(x=550,y=360,height=30)

label4 = Label(root,text = "How many childers you have : ",fg="black",font=("Slabo 27px","23")).place(x= 70,y=450)
child=Entry(root,textvariable=a4,width=23).place(x=550,y=450,height=30)

label5 = Label(root,text = "Do you smoke : ",fg="black",font=("Slabo 27px","23")).place(x= 70,y=540)
smokerA=Entry(root,textvariable=a5,width=23).place(x=550,y=540,height=30)

label6 = Label(root,text = "In which region do you live in : ",fg="black",font=("Slabo 27px","23")).place(x= 70,y=630)
region=Entry(root,textvariable=a6,width=23).place(x=550,y=630,height=30)
    

submit=Button(root,text="SUBMIT",fg="black",command = disp,font=("Slabo 27px",20)).place(x=850,y=600)

l1 = Label(root,text="Pred:" ,fg="black",font=("Slabo 27px","20")).place(x=850,y=300)
Final_l = Label(root,fg="black",font=("Ariel",20))
Final_l.place(x=1010,y=300)
root.mainloop()