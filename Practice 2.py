import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# here we import all the file which we use 
df = pd.read_csv("student_marks.csv")
# add csv file.
X=df["Hours"]
y=df["total"]
#we declare the variable what the prediction we want
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#apply train test split method for find the pridiction.
model=LinearRegression()
model.fit(X_train,y_train)
#declare train model on linear Regression
prediction=model.predict([[6]])
print(prediction)
# this is the predition statement
from sklearn.metrics import r2_score
y_pred=model.predict(X_test)
print(r2_score(y_test,y_pred))
#check Accuracy
print(model.coef_)
print(model.intercept_)
#this is for coefficient and intercept 

