from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd

#Reading the csv file
data = pd.read_csv('Parkinson_Data.csv')
data.head
data.describe().transpose()
data.shape
X = data.drop('name',axis=1)
y = data['status']

#Spliting the dataset into Test and Train 
X_train, X_test, y_train, y_test = train_test_split(X, y,
                    train_size=0.7, random_state=200)

#Implementing the Classifier
mlp = MLPClassifier(hidden_layer_sizes=(30),
        solver='sgd',learning_rate_init=0.01,max_iter=500)

mlp.fit(X_test,y_test)
print (mlp.score(X_test,y_test))
mlp.fit(X_train,y_train)
print (mlp.score(X_train,y_train))

predictions = mlp.predict(X_test)
#print("Neural Network Classifier")
print('Confusion Matrix')
print(confusion_matrix(y_test,predictions))
print('Classification Report')
print(classification_report(y_test,predictions))



