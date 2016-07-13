import sys
import time

from csxnet.datamodel import CData
from csxnet.brainforge.Architecture.NNModel import Network
from csxnet.brainforge.Utility.cost import *
from csxnet.brainforge.Utility.activations import *

dataroot = "D:/Data/csvs/" if sys.platform.lower() == "win32" else "/data/Prog/data/csvs/"
fcvpath = "fcvnyers.csv"
burleypath = "burleynyers.csv"

what = "burley"

crossvalrate = 0.3
pca = 11  # full = 13

eta = 1.5
lmdb = 0.0
hiddens = (100, 60)
cost = MSE

runs = 1000
epochs = 300
batch_size = 10


def pull_data(filename):
    d = CData(dataroot + filename, cross_val=0, pca=0)
    questions = d.data[..., 2:]
    targets = d.indeps
    return CData((questions, targets), cross_val=crossvalrate, pca=pca)


def dump_predictions(net: Network, on, ID):
    m = net.data.n_testing
    d = net.data
    questions = {"d": d.data, "l": d.learning, "t": d.testing}[on[0]][:m]
    ideps = {"d": d.indeps, "l": d.lindeps, "t": d.tindeps}[on[0]][:m]
    preds = net.predict(questions)
    answers = d.translate(preds, dummy=True)
    out = np.vstack((ideps, answers)).T

    np.savetxt("logs/C" + str(ID) + on + ".txt", out, delimiter="\t", header="IDEPS, ANSWERS", fmt="%s")


def build_network(data):
    net = Network(data=data, eta=eta, lmbd1=0.0, lmbd2=lmdb, mu=0.0, cost=cost)

    for hl in hiddens:
        net.add_fc(hl, activation=Sigmoid)

    net.finalize_architecture(activation=Sigmoid)

    return net


if __name__ == '__main__':
    start = time.time()
    path = fcvpath if "fcv" in what.lower() else burleypath
    results = [list(), list()]
    curve = [[list(), list()] for _ in range(3)]
    myData = pull_data(path)

    for r in range(1, runs+1):
        network = build_network(myData)
        myData.split_data()
        for epoch in range(epochs):
            network.learn(batch_size=batch_size)
            if r <= 3:
                if epoch % (epochs // 100) == 0:
                    curve[r-1][0].append(network.evaluate("testing"))
                    curve[r-1][1].append(network.evaluate("learning"))
        tfinal = network.evaluate("testing")
        lfinal = network.evaluate("learning")
        results[0].append(network.evaluate("testing"))
        results[1].append(network.evaluate("learning"))
        if r % 10 == 0:
            print("{} runs done! Avg TAcc: {} Avg LAcc: {}"
                  .format(r, np.mean(results[0]), np.mean(results[1])))
        if r <= 3:
            dump_predictions(network, "testing", ID=r)
            dump_predictions(network, "learning", ID=r)

    print(what, "T:", np.mean(results[0]), "STD:", np.std(results[0]))
    print(what, "L:", np.mean(results[1]), "STD:", np.std(results[1]))
    print("Run took {} seconds".format(int(time.time()-start)))

    import matplotlib.pyplot as plt
    X = X = np.arange(100) * (epochs // 100)
    f, axarr = plt.subplots(3, sharex=True)
    for i, acc in enumerate(curve):
        axarr[i].plot(X, acc[0], "r", label="T")
        axarr[i].plot(X, acc[1], "b", label="L")
        axarr[i].axis([0, X.max(), 0.0, 1.0])
        if i == 0:
            axarr[i].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                            ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
