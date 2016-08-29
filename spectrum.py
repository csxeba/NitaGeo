import sys

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense

from csxdata.frames import CData
from csxdata.utilities.nputils import avgpool as pool

dataroot = "D:/Data/" if sys.platform == "win32" else "/data/Prog/data/"
datapath = dataroot + "raw/nir/ntab.txt"


def pull_data(path, avgpool=0, avg_replications=True, standardize=True):
    myData = CData(path, cross_val=0.0)
    if avg_replications:
        myData.average_replications()
    if standardize:
        myData.transformation = "std"
    data = np.zeros((myData.data.shape[0], (myData.data.shape[1] // avgpool) - 2))
    for i, line in enumerate(myData.data):
        data[i] = pool(line, e=avgpool)
    return data, myData.indeps


def get_ffnn(X, Y):
    fanin, outshape = X.shape[1], Y.shape[1]
    model = Sequential()
    model.add(Dense(input_dim=fanin, output_dim=120, activation="tanh"))
    model.add(Dense(output_dim=outshape, activation="softmax"))
    model.compile("sgd", "categorical_crossentropy")
    return model


if __name__ == '__main__':
    learning, indeps = pull_data(datapath, 2, True, True)
    y = np.eye(len(indeps))
    network = get_ffnn(learning, y)
    network.fit(learning, y, batch_size=10, nb_epoch=1000)
