from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def home(request):

    return render(request,'home.html')

def result(request):
    data = pd.read_csv(r"C:\Users\aboub\Desktop\IT\ML projects\diabetes\diabetes.csv")
    x = data.drop(['Outcome'],axis=1)
    y = data['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.20)
    model = LogisticRegression()
    model.fit(x_train,y_train)

    var1 = float(request.GET['v1'])
    var2 = float(request.GET['v2'])
    var3 = float(request.GET['v3'])
    var4 = float(request.GET['v4'])
    var5 = float(request.GET['v5'])
    var6 = float(request.GET['v6'])
    var7 = float(request.GET['v7'])
    var8 = float(request.GET['v8'])

    predict = model.predict(np.array([var1,var2,var3,var4,var5,var6,var7,var8]).reshape(1,-1))
    predict = round(predict[0])
    if predict == 1:
        outcome = 'Positive :('
    elif predict == 0 :
        outcome = 'Negative ;)'
    else:
        outcome = "Couldn't detect -_-"

    context = {
        'result':outcome
    }
    return render(request,'home.html',context)