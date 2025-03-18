from decimal import Decimal,getcontext
import pandas as pd
import warnings
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
class forback:
        def forward(self,x,weights,bias):
                self.outs=np.sum(weights*x,axis=1)+bias
                return self.outs
        def cost_function(self,m,y1,y2):
                return np.around((1 / (2 * m)) *(np.sum(np.power(np.subtract(y2,y1),2))),decimals=5)
        def backward(self,weights,bias,cost_func,learning_rate,num_value,x,y):
                new_weights=weights-np.around(np.array(learning_rate*((np.sum(cost_func*(-x),axis=0)))/num_value),decimals=5)
                new_bias=bias-np.around(np.array(learning_rate*((np.sum(cost_func*(-1),axis=0)))/num_value),decimals=5)
                return new_weights,new_bias
class Linear_regression(forback):
        def __init__(self,epoche):
                self.weights=None
                self.bias=None
                self.out=None
                self.acc=None
                self.epoche=epoche
                self.msclist=[]
        def fit(self,x,y):
                rows,col=x.shape
                self.bias=0
                self.weights=np.zeros(col)
                mscdict={}
                for i in range(self.epoche):
                        out=super().forward(x,self.weights,self.bias)
                        msc=super().cost_function(rows,out,y)
                        self.weights,self.bias=super().backward(self.weights,self.bias,msc,0.0001,rows,x,y)
                        self.msclist.append(msc)
                        mscdict[msc]=(np.array([self.weights]),self.bias)
                values=mscdict[min(self.msclist)]
                self.weights=values[0]
                self.bias=values[1]
        def predict(self,x):
                out=super().forward(x,self.weights,self.bias)
                out=np.array([out],dtype='float')
                mscs=min(self.msclist)
                #print(f'weights and bias {(self.weights,self.bias)}\n predicted values {np.around(out)}')
                print(f"\n mean squared error {mscs})")
                return np.around(out)
warnings.filterwarnings("ignore")
a=Linear_regression(epoche=100)
data=pd.read_csv(r"Housing.csv")
x=data.iloc[:,1:-1]
y=data.iloc[:,0]
print(y)
X_train, X_test,y_train, y_test = train_test_split(x,y , 
                                   random_state=3,  
                                   test_size=0.90,  
                                   shuffle=True) 
a.fit(X_train,y_train)
out=a.predict(X_test)
y_test=np.array([y_test])
print(out.shape,y_test.shape)

7530060544
