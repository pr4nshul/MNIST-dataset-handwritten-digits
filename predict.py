import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TrainData = pd.read_csv('mnist_train.csv')
TrainData = TrainData.values
TrainLabels = TrainData[:,0]
TrainPixels = TrainData[:,1:]
TestData = pd.read_csv('mnist_test.csv')
TestData = TestData.values
TestLabel = TestData[:,0]
TestPixels = TestData[:,1:]
#KNN
def distance(X,Y):
    return np.sqrt(sum((X-Y)**2))
def KNN(X,Y,query,k=10):
    m = X.shape[0]
    vals =[]
    for i in range(m):
        d = distance(query,X[i])
        vals.append((d,Y[i]))
    vals = sorted(vals)
    vals = vals[:k]
    vals = np.array(vals)
    count = np.unique(vals[:,1],return_counts=True)
    index = count[1].argmax()
    pred  =count[0][index]
    return pred  

print('Enter values between 0 to 9999')
n =int(input())
pred=KNN(TestPixels,TestLabel,TestPixels[n,:])
img = TestPixels[n,:].reshape(28,28)
plt.imshow(img,cmap='gray')
plt.show()
print(pred)
print('Expected {}'.format(TestLabel[n]))
print('Do you want predict accuracy of Test set?')
acc = input()
y = 'yes'
if acc== y :
    print('This may take some time please wait...')
    mtest = TestLabel.shape[0]
    accuracy = 0
    for i in range(mtest):
        prediction = KNN(TrainPixels,TrainLabels,TestPixels[i,:])
        expected = TestLabel[i]
        if prediction==expected :
            accuracy+=1
            
    accuracy *=100
    print(accuracy)