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

jobs = 2

crossvalrate, pca, eta, lmbd, hiddens, activationO, activationH, cost, epochs, batch_size = \
    0.3,      10,  0.3, 0.0, (100, 30),  Sigmoid,     Sigmoid,     MSE,  5000,   20  # FCV Hypers in use
#   0.3,      10,  0.3, 0.0, (100, 30),  Sigmoid,     Sigmoid,     MSE,  5000,   20  # FCV Hypers, 1st best so far
#   0.2,      10,  0.2, 0.0, (100, 30),  Sigmoid,     Sigmoid,     MSE,  20000,  20  # FCV Hypers, 2nd best so far
#   0.3,      10,  0.2, 0.0, (100, 30),  Sigmoid,     Sigmoid,     MSE,  20000,  20  # Burley Hypers


def wgs_test(net: Network, on):
    from csxnet.nputils import haversine
    m = net.data.n_testing
    d = net.data
    questions = {"d": d.data, "l": d.learning, "t": d.testing}[on[0]][:m]
    ideps = d.upscale({"d": d.indeps, "l": d.lindeps, "t": d.tindeps}[on[0]][:m])
    preds = d.upscale(net.predict(questions))
    distance = haversine(ideps, preds)
    return int(np.mean(distance))


def dump_wgs_prediction(net: Network, on, ID):
    m = net.data.n_testing
    d = net.data
    questions = {"d": d.data, "l": d.learning, "t": d.testing}[on[0]][:m]
    ideps = d.upscale({"d": d.indeps, "l": d.lindeps, "t": d.tindeps}[on[0]][:m])
    preds = d.upscale(net.predict(questions))
    np.savetxt("logs/R" + str(ID) + on + '_ideps.txt', ideps, delimiter="\t")
    np.savetxt("logs/R" + str(ID) + on + '_preds.txt', preds, delimiter="\t")


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


def run(queue=None, ID=0):

    print("P{} starting...".format(ID))
    runlog = ""

    res = [list(), list()]
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
            res[0].append(terr)
            res[1].append(lerr)
            if e % 1000 == 0:
                epochlog += "P{}: epochs {}, eta: {}\n".format(ID, e, round(network.eta, 2)) \
                          + "TErr: {} kms\n".format(terr) \
                          + "LErr: {} kms\n".format(lerr)
                if e % 1000 == 0:
                    print(epochlog)
        runlog += epochlog

    dump_wgs_prediction(network, "learning", ID)
    dump_wgs_prediction(network, "testing", ID)

    if queue:
        queue.put((runlog, res, ID))
    else:
        return runlog, res, ID


def mp_experiment():
    import multiprocessing as mp

    start = time.time()

    myQueue = mp.Queue()
    jbs = jobs if jobs else mp.cpu_count()
    procs = [mp.Process(target=run, args=(myQueue, i), name=str("P{}".format(i))) for i in range(jbs)]
    results = []
    for proc in procs:
        proc.start()
    while len(results) != jbs:
        results.append(myQueue.get())
        time.sleep(0.1)
    for proc in procs:
        proc.join()

    timestr = "Run took {} seconds!\n".format(time.time() - start)

    X = np.arange(epochs // 50) * 50
    f, axarr = plt.subplots(jbs, sharex=True)
    for i, (log, acc, ID) in enumerate(results):
        logf = open("logs/{}log.txt".format(ID), "w")
        logf.write(log + timestr + "\n")
        logf.close()
        print("\n", timestr)

        axarr[i].plot(X, acc[0], "r", label="T")
        axarr[i].plot(X, acc[1], "b", label="L")
        axarr[i].set_title("P{}".format(ID))
        axarr[i].axis([0, X.max(), 0.0, 5000.0])
        if i == 0:
            axarr[i].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                            ncol=2, mode="expand", borderaxespad=0.)

    # plt.plot(X, results[1], "r", label="learning")
    # plt.plot(X, results[0], "b", label="testing")
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #            ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

if __name__ == '__main__':
    mp_experiment()
