import time
import sys
import multiprocessing as mp

import matplotlib.pyplot as plt

from csxnet.datamodel import CData, RData
from csxnet.brainforge.Architecture.NNModel import Network
from csxnet.brainforge.Utility.cost import *
from csxnet.brainforge.Utility.activations import *


dataroot = "D:/Data/csvs/" if sys.platform.lower() == "win32" else "/data/Prog/data/csvs/"
fcvpath = "fcvnyers.csv"
burleypath = "burleynyers.csv"
fullpath = "fullnyers"

what = "burley"

crossvalrate, pca, eta, lmbd, hiddens, activationO, activationH, cost, epochs, batch_size = \
    0.3,      10,  0.2, 0.0, (100, 30),  Sigmoid,     Sigmoid,     MSE,  20000,  20  # Burley Hypers
#   0.3,      10,  0.3, 0.0, (100, 30),  Sigmoid,     Sigmoid,     MSE,  10000,  20  # FCV Hypers, 1st best so far
#   0.3,      10,  0.3, 0.0, (100, 30),  Sigmoid,     Sigmoid,     MSE,  5000,   20  # FCV Hypers, 2nd best so far
#   0.2,      10,  0.2, 0.0, (100, 30),  Sigmoid,     Sigmoid,     MSE,  20000,  20  # FCV Hypers, 3rd best so far

runs = 10


def wgs_test(net: Network, on):
    """Test the network's accuracy

    by computing the haversine distance between the target and the predicted coordinates"""
    from csxnet.nputils import haversine
    m = net.data.n_testing
    d = net.data
    questions = {"d": d.data, "l": d.learning, "t": d.testing}[on[0]][:m]
    ideps = d.upscale({"d": d.indeps, "l": d.lindeps, "t": d.tindeps}[on[0]][:m])
    preds = d.upscale(net.predict(questions))
    distance = haversine(ideps, preds)
    return int(np.mean(distance))


def dump_wgs_prediction(net: Network, on, ID):
    """Dumps the coordinate predictions into a text file"""
    m = net.data.n_testing
    d = net.data
    questions = {"d": d.data, "l": d.learning, "t": d.testing}[on[0]][:m]
    ideps = d.upscale({"d": d.indeps, "l": d.lindeps, "t": d.tindeps}[on[0]][:m])
    preds = d.upscale(net.predict(questions))
    np.savetxt("logs/R" + str(ID) + on + '_ideps.txt', ideps, delimiter="\t")
    np.savetxt("logs/R" + str(ID) + on + '_preds.txt', preds, delimiter="\t")


def pull_data(filename):
    """Pulls the learning data from a csv file."""
    d = CData(dataroot + filename, cross_val=0, pca=0)
    questions = d._datacopy[..., 2:]
    targets = d._datacopy[..., :2]
    return RData((questions, targets), cross_val=crossvalrate, indeps_n=2, header=False, pca=pca)


def build_network(data):
    """Generates a neural network from given hyperparameters"""
    net = Network(data=data, eta=eta, lmbd=lmbd, cost=cost)
    for hl in hiddens:
        net.add_fc(hl, activation=activationH)

    net.finalize_architecture(activation=activationO)
    return net


def run1(queue=None, return_dynamics=False):
    """One run corresponds to the training of a network with randomized weights"""
    path = fcvpath if "fcv" in what.lower() else burleypath
    myData = pull_data(path)
    network = build_network(myData)
    dynamics = [list(), list()]

    for e in range(1, epochs + 1):
        network.learn(batch_size=batch_size)
        if return_dynamics and e % (epochs // 100) == 0:
            terr = wgs_test(network, "testing")
            lerr = wgs_test(network, "learning")
            dynamics[0].append(terr)
            dynamics[1].append(lerr)

    output = dynamics[0][-1], dynamics[1][-1] if not return_dynamics else dynamics

    if queue and return_dynamics:
        queue.put(output)
    elif not queue and return_dynamics:
        return output


def mp_run(jobs=mp.cpu_count(), return_dynamics=False):
    """Organizes multiple runs in parallel"""
    myQueue = mp.Queue()
    procs = [mp.Process(target=run1, args=(myQueue, return_dynamics), name="P{}".format(i)) for i in range(jobs)]
    results = []
    for proc in procs:
        proc.start()
    while len(results) != jobs:
        results.append(myQueue.get())
        time.sleep(0.1)
    for proc in procs:
        proc.join()

    return results


def logged_run():
    """This experimental setup logs the performance after several runs

    Writes the mean and STD of the recorded accuracies to Rlog.txt"""
    start = time.time()
    results = [list(), list()]
    logchain = ""
    for r in range(1, runs+1):
        res = mp_run()
        results[0].extend(res[0])
        results[1].extend(res[1])
        logchain += "Acc @ {}: T: {}\tL: {}\n".format(r, np.mean(results[0]), np.mean(results[1]))
    logchain += "---------------\nFinal Tests:\n"
    logchain += "L: mean: {} STD: {}\n".format(np.mean(results[0]), np.std(results[0]))
    logchain += "T: mean: {} STD: {}\n".format(np.mean(results[1]), np.std(results[1]))
    logchain += "Hypers: crossvalrate, pca, eta, lmbd, hiddens, activationO, activationH, cost, epochs, batch_size:\n"
    logchain += str([crossvalrate, pca, eta, lmbd, hiddens, activationO, activationH, cost, epochs, batch_size]) + "\n"
    logchain += "Run time: {}s\n".format(time.time()-start)
    logf = open("logs/Rlog{}.txt".format(runs), "w")
    logf.write(logchain)
    logf.close()
    print("Fin")


def plotted_run():
    """This experimental setup plots the run dynamics of some epochs

    The generated diagrams must be saved manually."""
    X = np.arange(epochs // 50) * 50
    f, axarr = plt.subplots(3, sharex=True)

    dynamics = mp_run(jobs=3, return_dynamics=True)

    for i in range(3):
        axarr[i].plot(X, dynamics[0], "r", label="T")
        axarr[i].plot(X, dynamics[1], "b", label="L")
        axarr[i].set_title("Run {}".format(i))
        axarr[i].axis([0, X.max(), 0.0, 5000.0])
        if i == 0:
            axarr[i].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                            ncol=2, mode="expand", borderaxespad=0.)
            plt.plot(X, dynamics[1], "r", label="learning")
        plt.plot(X, dynamics[0], "b", label="testing")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.show()


if __name__ == '__main__':
    plotted_run()
