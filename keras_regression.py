import sys

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation

from regression import pull_data

dataroot = "D:/Data/csvs" if sys.platform == "win32" else "/data/Prog/data/csvs"
fl = "fcvnyers.csv"


tobacco = pull_data(fl)
fanin, outshape = tobacco.neurons_required()

network = Sequential()
network.add(Dense(100, input_dim=np.prod(fanin)))
network.add(Activation("tanh"))
network.add(Dense(30))
network.add(Activation("tanh"))
network.add(Dense(outshape))
network.add(Activation("sigmoid"))
network.compile("sgd", "mse")

network.fit(tobacco.data, tobacco.indeps, batch_size=10, nb_epoch=1000, validation_split=0.3, verbose=0)
network.predict(tobacco.testing)
print("Done!")
