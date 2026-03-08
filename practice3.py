import pandas as pd
# load the csv file
df = pd.read_csv("Student_marks.csv")
#this is for display the 5 information which is present
print(df.head())
#All the information are displayed
print(df.info())
# if the null value are present then the value are printed
print(df.isnull().sum())
# check the duplicate value if it present then displayed it
print(df.duplicated().sum())
# check basic statics mean min and max value if it is present
print(df.describe())
#displayed all the columns name which is present
print(df.columns)
