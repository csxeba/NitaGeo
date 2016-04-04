import sys
import time
import multiprocessing

from mylibs.datamodel import CData
from mylibs.thinkster.Architecture.NNModel import Network
from mylibs.thinkster.Utility.cost import *
from mylibs.thinkster.Utility.activations import *

dataroot = "D:/Data/csvs/" if sys.platform.lower() == "win32" else "/data/Prog/data/csvs/"
fcvpath = "fcvnyers.csv"
burleypath = "burleynyers.csv"

what = "fcv"

crossvalrate = 0.5
pca = 13  # full = 13

eta = .8
lmdb = .1
hiddens = (60, 60)
cost = Xent

runs = 10
epochs = 500
batch_size = 15


def pull_data(filename):
    d = CData(dataroot + filename, cross_val=0, pca=0)
    questions = d._datacopy[..., 2:]
    targets = d.indeps
    return CData((questions, targets), cross_val=crossvalrate, pca=pca)


def build_network(data):
    net = Network(data=data, eta=eta, lmbd=lmdb, cost=cost)

    for hl in hiddens:
        net.add_fc(hl, activation=Sigmoid)

    net.finalize_architecture(activation=Sigmoid)

    return net


if __name__ == '__main__':
    start = time.time()
    path = fcvpath if "fcv" in what.lower() else burleypath
    results = [list(), list()]
    myData = pull_data(path)
    net = build_network(myData)

    for r in range(1, runs+1):
        for __ in range(epochs):
            net.learn(batch_size=batch_size)
        results[0].append(net.evaluate("testing"))
        results[1].append(net.evaluate("learning"))
        # if r % 10 == 0:
        #     print("{} runs done! Avg TAcc: {} Avg LAcc: {}"
        #           .format(r, sum(results[0])/r, sum(results[1])/r))

    print(what, "T:", sum(results[0])/runs)
    print(what, "L:", sum(results[1])/runs)
    print("Run took {} seconds".format(int(time.time()-start)))
