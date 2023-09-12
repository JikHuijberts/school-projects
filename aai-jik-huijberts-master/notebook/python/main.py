from nn_helper import Model, Layer
from activation import *
from loss import *


X_train = np.array([[0, 0, 1],
                    [1, 1, 1],
                    [1, 0, 1],
                    [0, 1, 1]])
y_train = np.array([[0, 1, 1, 0]]).T

model = Model(lr=0.01, lf=crossentropy)
model.add(Layer(3, 10, Relu()))
model.add(Layer(10, 5, Relu()))
model.add(Layer(5, 2, Relu()))
model.add(Layer(2, 1, Sigmoid()))


a = model.predict(X_train)
print("Fitting model")
history = model.fit(X_train, y_train, epochs=1000)
print(f"\nMean loss: {np.mean(history)}")
print("last predict")
b = model.predict(X_train)
print(f"\nExpected output: {y_train}")
print(f"\nPredictions before training: {a}")
print(f"\nPredictions after training: {b}")
