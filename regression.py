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
fullpath = "fullnyers"

what = "fcv"

logchain = ""

crossvalrate, pca, eta, lmbd, hiddens, activationO, activationH, cost, epochs, batch_size = \
    0.2,      10,  0.1, 0.0, (120, 30),  Sigmoid,     Sigmoid,     MSE,  20000,  20  # FCV Hypers
#   0.3,      10,  0.2, 0.0,  (100, 30), Sigmoid,     Sigmoid,     MSE,  20000,  20  # Burley Hypers


def wgs_test(net: Network, on):
    from csxnet.nputils import haversine
    m = net.data.n_testing
    d = net.data
    questions = {"d": d.data, "l": d.learning, "t": d.testing}[on[0]][:m]
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
    np.savetxt("logs/" + on + '_ideps.txt', ideps, delimiter="\t")
    np.savetxt("logs/" + on + '_preds.txt', preds, delimiter="\t")


def pull_data(filename):
    d = CData(dataroot + filename, cross_val=0, pca=0)
    questions = d._datacopy[..., 2:]
    targets = d._datacopy[..., :2]
    return RData((questions, targets), cross_val=crossvalrate, indeps_n=2, header=False, pca=pca)


def build_network(data):
    net = Network(data=data, eta=eta, lmbd=lmbd, cost=cost)
    for hl in hiddens:
        net.add_fc(hl, activation=activationH)

    net.finalize_architecture(activation=activationO)
    return net


def run():

    runlog = ""

    def autoencode():
        print("Autoencoding...")
        network.eta = eta / 2
        for ep in range(1, 10001):
            network.autoencode(batch_size)
            if ep % 1000 == 0:
                print("error @ {}:".format(ep), network.error)
        network.eta = eta

    global logchain, results, eta
    path = fcvpath if "fcv" in what.lower() else burleypath
    myData = pull_data(path)
    network = build_network(myData)

    # autoencode()

    for e in range(1, epochs + 1):
        epochlog = ""
        network.learn(batch_size=batch_size)
        if e % 50 == 0 and e != 0:
            terr = wgs_test(network, "testing")
            lerr = wgs_test(network, "learning")
            results[0].append(terr)
            results[1].append(lerr)
            if e % 1000 == 0:
                epochlog += "Epochs {}, eta: {}\n".format(e, round(network.eta, 2)) \
                          + "TErr: {} kms\n".format(terr) \
                          + "LErr: {} kms\n".format(lerr)
                print(epochlog)
        runlog += epochlog

    dump_wgs_prediction(network, "learning")
    dump_wgs_prediction(network, "testing")

    return runlog

if __name__ == '__main__':

    results = [list(), list()]  # Save the start unix epoch

    start = time.time()

    logchain = run()
    timestr = "Run took {} seconds!\n".format(time.time() - start)
    logchain += timestr
    print(timestr)

    logchain += "Hypers: crossvalrate, pca, eta, lmbd, hiddens, activationO, activationH, cost, epochs, batch_size\n"
    logchain += str((crossvalrate, pca, eta, lmbd, hiddens, activationO, activationH, cost, epochs, batch_size)) + "\n"

    # Write the log to file
    log = open("logs/log.txt", "w")
    log.write(logchain)
    log.close()

    # Plot the learning dynamics
    X = np.arange(len(results[0])) * 50
    plt.plot(X, results[1], "r", label="learning")
    plt.plot(X, results[0], "b", label="testing")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

