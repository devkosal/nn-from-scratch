# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import pandas as pd
import numpy as np
np.seterr(all='ignore')

def relu(x,der=False):
    if der is False:
        return np.maximum(0,x)
    else:
        x[x<=0] = 0
        x[x>0] = 1
        return x

def sigmoid(x, der=False):
    if der is False:
        return 1 / (1+np.e**-x)
    else:
        return sigmoid(x) * (1 - sigmoid(x))

def linear(x,der=False):
    if der is False:
        return x
    else:
        return 1


def rmse(input,pred):
    mse = ((pred-input)**2).mean()
    return np.sqrt(mse)



# Airbnb

import os
os.chdir('/Users/devsharma/Dropbox/Education/Data Science/Python Learning')

data = pd.read_csv("data/analysisData.csv")
cats = [col for col in data.columns if type(data[col][0]) == str]

data.drop(cats + ["reviews_per_month"],axis=1,inplace=True)
data = data[data["price"].notna()]
data.fillna(data.mean())

nas = data.isna().sum()
index = nas.index

na_cols = [col for col in index if nas[col] != 0]
data.drop(na_cols,axis=1,inplace=True)
x = data[[col for col in data.columns if col != "price"]].values
y = data["price"].values
y = np.array([y]).T

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler()
y_scaled = mm_scaler.fit_transform(y)

# split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y_scaled, test_size=0.3, random_state=101)

class NeuralNetwork:
    def __init__(self, x, y, neurons_list=[100], lr=0.1, wd=0.01, annealing_lr_pct = 1,momentum = .9,rms_prop = .9, act = "sigmoid", output_act = "sigmoid", test_set=None):
        self.input = x
        self.y = y
        self.neurons_list = neurons_list
        self.lr = lr
        self.wd = wd
        self.annealing_lr_pct = annealing_lr_pct
        self.momentum = momentum
        self.rms_prop = rms_prop
        self.constant = 1e-6
        self.mom_w3, self.mom_w2, self.mom_w1, self.mom_b3, self.mom_b2, self.mom_b1 = 0,0,0,0,0,0
        self.rpr_w3, self.rpr_w2, self.rpr_w1, self.rpr_b3, self.rpr_b2, self.rpr_b1 = 0, 0, 0, 0, 0, 0
        self.test_set = test_set

        self.weights1 = np.random.randn(x.shape[1], neurons_list[0])
        self.bias1 = np.full((1, neurons_list[0]), 1.)

        if len(self.neurons_list) == 1:
            self.weights2 = np.random.randn(neurons_list[0], 1)
            self.bias2 = np.full((1, 1),  1.)
        elif len(self.neurons_list) == 2:
            self.weights2 = np.random.randn(neurons_list[0], neurons_list[1])
            self.bias2 = np.full((1, neurons_list[1]), 1.)
            self.weights3 = np.random.randn(neurons_list[1], 1)
            self.bias3 = np.full((1, 1), 1.)

        if act.lower() == "linear": self.act = linear
        elif act.lower() == "relu": self.act = relu
        else: self.act = sigmoid

        if output_act.lower() == "linear": self.output_act = linear
        elif output_act.lower() == "relu": self.output_act = relu
        else: self.output_act = sigmoid

    def predict(self,x = None, y = None):
        if x is not None:
            self.batch_input = x
        if y is not None:
            self.batch_y = y
        self.z1 = (self.batch_input @ self.weights1) + self.bias1
        self.layer1 = self.act(self.z1)

        if len(self.neurons_list) == 1:
            self.z2 = (self.layer1 @ self.weights2) + self.bias2
            self.yhat = self.output_act(self.z2)
        elif len(self.neurons_list) == 2:
            self.z2 = (self.layer1 @ self.weights2) + self.bias2
            self.layer2 = self.act(self.z2)
            self.z3 = (self.layer2 @ self.weights3) + self.bias3
            self.yhat = self.output_act(self.z3)
        if x is not None and y is None:
            return self.yhat

    def backprop(self):
        self.lr = self.lr
        if len(self.neurons_list) == 1:
            self.grad_w2 = (self.layer1.T @ (2 * (self.yhat - self.batch_y) * self.output_act(self.z2, der=True))) + self.wd * self.weights2
            self.grad_w1 = (self.batch_input.T @ ((self.output_act(self.z2, der=True) * 2 * (self.yhat - self.batch_y) @ self.weights2.T) * self.act(self.z1, der=True))) + self.wd * self.weights1

            self.grad_b2 = np.sum((2 * (self.yhat - self.batch_y) * self.output_act(self.z2, der=True)), axis=0, keepdims=True)
            self.grad_b1 = np.sum(((2 * (self.yhat - self.batch_y) * self.output_act(self.z2, der=True)) @ self.weights2.T) * self.act(self.z1, der=True), axis=0, keepdims=True)

            self.mom_w2 = self.grad_w2 * (1 - self.momentum) + self.mom_w2 * self.momentum
            self.mom_w1 = self.grad_w1 * (1 - self.momentum) + self.mom_w1 * self.momentum
            self.mom_b2 = self.grad_b2 * (1 - self.momentum) + self.mom_b2 * self.momentum
            self.mom_b1 = self.grad_b1 * (1 - self.momentum) + self.mom_b1 * self.momentum

            self.rpr_w2 = self.grad_w2**2 * (1 - self.rms_prop) + self.rpr_w2 * self.rms_prop
            self.rpr_w1 = self.grad_w1**2 * (1 - self.rms_prop) + self.rpr_w1 * self.rms_prop
            self.rpr_b2 = self.grad_b2**2 * (1 - self.rms_prop) + self.rpr_b2 * self.rms_prop
            self.rpr_b1 = self.grad_b1**2 * (1 - self.rms_prop) + self.rpr_b1 * self.rms_prop

            self.weights2 = self.weights2 - (self.mom_w2 / np.sqrt(self.rpr_w2 + self.constant) * self.lr)
            self.weights1 = self.weights1 - (self.mom_w1 / np.sqrt(self.rpr_w1 + self.constant) * self.lr)

            self.bias2 = self.bias2 - (self.mom_b2 / np.sqrt(self.rpr_b2 + self.constant) * self.lr)
            self.bias1 = self.bias1 - (self.mom_b1 / np.sqrt(self.rpr_b1 + self.constant) * self.lr)

        elif len(self.neurons_list) == 2:
            self.grad_w3 = (self.layer2.T @ (2 * (self.yhat - self.batch_y) * self.output_act(self.z3, der=True))) + self.wd * self.weights3
            self.grad_w2 = (self.layer1.T @ ((2 * (self.yhat - self.batch_y) * self.output_act(self.z3, der=True) @ self.weights3.T) * self.act(self.z2, der=True))) + self.wd * self.weights2
            self.grad_w1 = (self.batch_input.T @ ((((self.output_act(self.z3, der=True) * 2 * (self.yhat - self.batch_y) @ self.weights3.T) * self.act(self.z2, der=True)) @ self.weights2.T) * self.act(self.z1, der=True))) + self.wd * self.weights1

            self.grad_b3 = np.sum((2 * (self.yhat - self.batch_y) * self.output_act(self.z3, der=True)), axis=0, keepdims=True)
            self.grad_b2 = np.sum(((2 * (self.yhat - self.batch_y) * self.output_act(self.z3, der=True)) @ self.weights3.T) * self.act(self.z2, der=True), axis=0, keepdims=True)
            self.grad_b1 = np.sum(((((2 * (self.yhat - self.batch_y) * self.output_act(self.z3, der=True)) @ self.weights3.T) * self.act(self.z2, der=True)) @ self.weights2.T) * self.act(self.z1, der=True), axis=0, keepdims=True)

            self.mom_w3 = self.grad_w3 * (1 - self.momentum) + self.mom_w3 * self.momentum
            self.mom_w2 = self.grad_w2 * (1 - self.momentum) + self.mom_w2 * self.momentum
            self.mom_w1 = self.grad_w1 * (1 - self.momentum) + self.mom_w1 * self.momentum
            self.mom_b3 = self.grad_b3 * (1 - self.momentum) + self.mom_b3 * self.momentum
            self.mom_b2 = self.grad_b2 * (1 - self.momentum) + self.mom_b2 * self.momentum
            self.mom_b1 = self.grad_b1 * (1 - self.momentum) + self.mom_b1 * self.momentum

            self.rpr_w3 = self.grad_w3 ** 2 * (1 - self.rms_prop) + self.rpr_w3 * self.rms_prop
            self.rpr_w2 = self.grad_w2 ** 2 * (1 - self.rms_prop) + self.rpr_w2 * self.rms_prop
            self.rpr_w1 = self.grad_w1 ** 2 * (1 - self.rms_prop) + self.rpr_w1 * self.rms_prop
            self.rpr_b3 = self.grad_b3 ** 2 * (1 - self.rms_prop) + self.rpr_b3 * self.rms_prop
            self.rpr_b2 = self.grad_b2 ** 2 * (1 - self.rms_prop) + self.rpr_b2 * self.rms_prop
            self.rpr_b1 = self.grad_b1 ** 2 * (1 - self.rms_prop) + self.rpr_b1 * self.rms_prop

            self.weights3 = self.weights3 - (self.mom_w3 / np.sqrt(self.rpr_w3 + self.constant) * self.lr)
            self.weights2 = self.weights2 - (self.mom_w2 / np.sqrt(self.rpr_w2 + self.constant) * self.lr)
            self.weights1 = self.weights1 - (self.mom_w1 / np.sqrt(self.rpr_w1 + self.constant) * self.lr)

            self.bias3 = self.bias3 - (self.mom_b3 / np.sqrt(self.rpr_b3 + self.constant) * self.lr)
            self.bias2 = self.bias2 - (self.mom_b2 / np.sqrt(self.rpr_b2 + self.constant) * self.lr)
            self.bias1 = self.bias1 - (self.mom_b1 / np.sqrt(self.rpr_b1 + self.constant) * self.lr)

    def fit_one_cycle(self,epochs,batch_size=50):
        mat_len = len(self.input)
        last_iter = mat_len % batch_size

        for epoch in range(epochs):
            for i in range(0,mat_len-last_iter,batch_size):
                self.predict(x=self.input[i:i + batch_size], y=self.y[i:i + batch_size])
                self.backprop()
            self.predict(x=self.input[mat_len - last_iter + 1:mat_len+1], y=self.y[mat_len - last_iter + 1:mat_len+1])
            self.backprop()
            # print(rmse(mm_scaler.inverse_transform(self.batch_y), mm_scaler.inverse_transform(self.yhat)))
            # print(rmse(self.batch_y, self.yhat))
            y_pred = self.predict(self.input)
            print("NN Train RMSE is {}".format(
                rmse(mm_scaler.inverse_transform(y_pred), mm_scaler.inverse_transform(self.y))))

            if self.test_set is not None:
                self.x_test = self.test_set[0]
                self.y_test = self.test_set[1]
                y_pred = self.predict(self.x_test)
                print("NN Test RMSE is {}".format(
                    rmse(mm_scaler.inverse_transform(y_pred), mm_scaler.inverse_transform(y_test))))




nn = NeuralNetwork(x, y_scaled, neurons_list=[10,10], lr = .02, wd = .1, annealing_lr_pct=1,momentum=.9,rms_prop=.9,act="relu",test_set=[x_test,y_test])

nn.fit_one_cycle(epochs=30,batch_size=100)

# print("NN RMSE is {}".format(rmse(y_pred,y_test)))


#submission
sub_data = pd.read_csv("data/scoringData.csv")

cats = [col for col in sub_data.columns if type(sub_data[col][0]) == str]
sub_data.drop(cats,axis=1,inplace=True)

sub_data.fillna(sub_data.mean())

nas = sub_data.isna().sum()
index = nas.index

na_cols = [col for col in index if nas[col] != 0]
sub_data.drop(na_cols,axis=1,inplace=True)
x_sub = sub_data[[col for col in sub_data.columns]].values
x_sub = scaler.transform(x_sub)

sub_pred = nn.predict(x_sub)
sub_pred = mm_scaler.inverse_transform(sub_pred)

final = pd.DataFrame({"id": sub_data["id"],"price":sub_pred.ravel()})

final.to_csv("late_submission.csv",index=False)

#Airbnb benchmark

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn import metrics

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(x_train, y_train.ravel())

forest_train_rmse = np.sqrt(metrics.mean_squared_error(mm_scaler.inverse_transform(y_train),mm_scaler.inverse_transform(forest_reg.predict(x_train).reshape(-1,1))))
print('Random Forest Train RMSE: %.4f' % forest_train_rmse)

y_pred = forest_reg.predict(x_test).reshape(-1,1)
forest_mse = metrics.mean_squared_error(mm_scaler.inverse_transform(y_pred), mm_scaler.inverse_transform(y_test))
forest_rmse = np.sqrt(forest_mse)
print('Random Forest Test RMSE: %.4f' % forest_rmse)



for i in range(27):
    if data.drop("price",axis=1).columns[i] != sub_data.columns[i]:
        print(i)