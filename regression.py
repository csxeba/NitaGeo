import sys

from csxnet.datamodel import CData, RData
from csxnet.brainforge.Architecture.NNModel import Network
from csxnet.brainforge.Utility.cost import *
from csxnet.brainforge.Utility.activations import *


dataroot = "D:/Data/csvs/" if sys.platform.lower() == "win32" else "/data/Prog/data/csvs/"
fcvpath = "fcvnyers.csv"
burleypath = "burleynyers.csv"

what = "fcv"

crossvalrate = 0.2
pca = 10  # full = 13

eta = 0.7
lmdb = 0.0
hiddens = (100, 30, 30)
activationH = Sigmoid
activationO = Sigmoid
cost = MSE

runs = 1
epochs = 10000
batch_size = 20


def wgs_to_distance(coords1: np.ndarray, coords2: np.ndarray):
    import math
    R = 6378.137  # radius of the Earth in kms
    dY = coords1[0] - coords2[0] * math.pi / 180
    dX = coords1[1] - coords2[1] * math.pi / 180
    a = (np.sin(dY/2)**2) \
        + np.cos(coords1[0] * math.pi / 180) \
        * np.cos(coords1[1] * math.pi / 180) \
        * (np.sin(dX/2)**2)
    a = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return a * R


def wgs_test(net: Network, on):
    m = net.data.n_testing
    d = net.data
    questions = {"d": d.data[:m], "l": d.learning[:m], "t": d.testing}[on[0]]
    ideps = d.upscale({"d": d.indeps, "l": d.lindeps, "t": d.tindeps}[on[0]][:m])
    preds = d.upscale(net.predict(questions))
    distance = wgs_to_distance(ideps, preds)
    return np.mean(distance)


def pull_data(filename):
    d = CData(dataroot + filename, cross_val=0, pca=0)
    questions = d._datacopy[..., 2:]
    targets = d._datacopy[..., :2]
    return RData((questions, targets), cross_val=crossvalrate, indeps_n=2, header=False, pca=pca)


def build_network(data):
    net = Network(data=data, eta=eta, lmbd=lmdb, cost=cost)
    for hl in hiddens:
        net.add_fc(hl, activation=activationH)

    net.finalize_architecture(activation=activationO)
    return net

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path = fcvpath if "fcv" in what.lower() else burleypath
    results = [list(), list()]
    myData = pull_data(path)
    network = build_network(myData)

    for r in range(runs):
        for e in range(epochs):
            network.learn(batch_size=batch_size)
            if e % 100 == 0 and e != 0:
                terr = wgs_test(network, "testing")
                lerr = wgs_test(network, "learning")
                results[0].append(terr)
                results[1].append(lerr)
                if e % 1000 == 0 and e != 0:
                    print("{} epochs done!".format(e))
                    print("TErr: {}".format(terr))
                    print("LErr: {}".format(lerr))

        if r % 10 == 0 and r != 0:
            print("{} runs done! Avg TAcc: {} Avg LAcc: {}"
                  .format(r, sum(results[0])/r, sum(results[1])/r))

    X = np.arange((epochs//100))[:-1] * 100
    plt.plot(X, results[1], "r", label="learning")
    plt.plot(X, results[0], "b", label="testing")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

