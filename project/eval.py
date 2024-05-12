import pandas as pd

from sklearn.metrics import accuracy_score, classification_report


#reading the output
path='output/trained.csv'
df= pd.read_csv(path)

#calculating accuracy of bert
accuracy_b = accuracy_score(df.pred_label,df.true)

#priniing
print("Bert Model Accuracy : ", accuracy_b)
print(classification_report(df.pred_label, df.true))