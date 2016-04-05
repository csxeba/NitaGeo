import time
import sys

import matplotlib.pyplot as plt

from csxnet.datamodel import CData, RData
from csxnet.brainforge.Architecture.NNModel import Network
from csxnet.brainforge.Utility.cost import *
from csxnet.brainforge.Utility.activations import *


dataroot = "D:/Data/csvs/" if sys.platform.lower() == "win32" else "/data/Prog/data/csvs/"
fcvpath = "fcvnyers.csv"
burleypath = "burleynyers.csv"

what = "fcv"

crossvalrate = 0.3
pca = 10  # full = 13

eta = 0.2
lmdb = 0.0
hiddens = (150, 40)
activationH = Sigmoid
activationO = Sigmoid
cost = MSE

runs = 1
epochs = 20000
batch_size = 15

logchain = ""


def wgs_test(net: Network, on):
    from csxnet.utilities import haversine
    m = net.data.n_testing
    d = net.data
    questions = {"d": d.data[:m], "l": d.learning[:m], "t": d.testing}[on[0]]
    ideps = d.upscale({"d": d.indeps, "l": d.lindeps, "t": d.tindeps}[on[0]][:m])
    preds = d.upscale(net.predict(questions))
    distance = haversine(ideps, preds)
    return int(np.mean(distance))


def dump_wgs_prediction(net: Network, on):
    m = net.data.n_testing
    d = net.data
    questions = {"d": d.data[:m], "l": d.learning[:m], "t": d.testing}[on[0]]
    ideps = d.upscale({"d": d.indeps, "l": d.lindeps, "t": d.tindeps}[on[0]][:m])
    preds = d.upscale(net.predict(questions))
    np.savetxt(on + '_ideps.txt', ideps, delimiter="\t")
    np.savetxt(on + '_preds.txt', preds, delimiter="\t")


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


def run():
    global logchain, results, eta
    path = fcvpath if "fcv" in what.lower() else burleypath
    myData = pull_data(path)
    network = build_network(myData)

    print("Autoencoding...")
    network.eta = eta / 2
    for e in range(1, 10001):
        network.autoencode(batch_size)
        if e % 1000 == 0:
            print("error @ {}:".format(e), network.error)
    network.eta = eta

    for e in range(1, epochs + 1):
        epochlog = ""
        network.learn(batch_size=batch_size)
        if e % 100 == 0 and e != 0:
            terr = wgs_test(network, "testing")
            lerr = wgs_test(network, "learning")
            results[0].append(terr)
            results[1].append(lerr)
            if e % 1000 == 0:
                epochlog += "Epochs {}\n".format(e) \
                          + "TErr: {} kms\n".format(terr) \
                          + "LErr: {} kms\n".format(lerr)
                print(epochlog)
        if e % 10000 == 0 and network.eta < 3.0:
            network.eta += 0.1
        logchain += epochlog

    logchain += "Run took {} seconds!\n".format(time.time() - start)

    dump_wgs_prediction(network, "learning")
    dump_wgs_prediction(network, "testing")


if __name__ == '__main__':

    results = [list(), list()]  # Save the start unix epoch

    start = time.time()

    # Run the network until all epochs done or until keyboard-interrupt
    try:
        run()
    except KeyboardInterrupt:
        pass

    # Print the log
    print(logchain, end="")

    # Write the log to file
    log = open("log.txt", "w")
    log.write(logchain)
    log.close()

    # Plot the learning dynamics
    X = np.arange(len(results[0])) * 100
    plt.plot(X, results[1], "r", label="learning")
    plt.plot(X, results[0], "b", label="testing")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

