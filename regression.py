import time
import sys
import multiprocessing as mp

import matplotlib.pyplot as plt

from nnet.datamodel import CData, RData
from nnet.brainforge.Architecture.NNModel import Network
from nnet.brainforge.Utility.cost import *
from nnet.brainforge.Utility.activations import *

dataroot = "D:/Data/csvs/" if sys.platform.lower() == "win32" else "/data/Prog/data/csvs/"
fcvpath = "fcvnyers.csv"
burleypath = "burleynyers.csv"
fullpath = "fullnyers.csv"

what = "burley"

crossvalrate, pca, eta,  lmbd,  hiddens, activationO, activationH,   cost, epochs, batch_size = \
    0.3,      10,  1.0,  0.2,   (60, 60, 30),  Sigmoid,     Sigmoid,     MSE,  10000,  20  # fcv Hypers
#   0.2,      10,  3.0,  0.0,   (30, 30),  Sigmoid,     Sigmoid,     MSE,  10000,  20  # Burley Hypers
#   0.3,      10,  0.3,  0.0,  (100, 30),  Sigmoid,     Sigmoid,     MSE,  10000,  20  # FCV Hypers, 1st best so far
#   0.3,      10,  0.3,  0.0,  (100, 30),  Sigmoid,     Sigmoid,     MSE,   5000,  20  # FCV Hypers, 2nd best so far
#   0.2,      10,  0.2,  0.0,  (100, 30),  Sigmoid,     Sigmoid,     MSE,  20000,  20  # FCV Hypers, 3rd best so far

runs = 50
no_plotpoints = 200
no_plots = 2
jobs = 2


def wgs_test(net: Network, on):
    """Test the network's accuracy

    by computing the haversine distance between the target and the predicted coordinates"""
    from csxnet.nputils import haversine
    m = net.data.n_testing
    d = net.data
    questions = {"d": d.data, "l": d.learning, "t": d.testing}[on[0]][:m]
    ideps = {"d": d.indeps, "l": d.lindeps, "t": d.tindeps}[on[0]][:m]
    usideps = d.upscale(ideps)
    preds = net.predict(questions)
    uspreds = d.upscale(preds)
    distance = haversine(usideps, uspreds)
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
    questions = np.log(d.data[..., 2:])
    targets = d.data[..., :2]
    # questions = d.data[..., 2:]
    # targets = d.data[..., :2]
    # questions = np.concatenate((d.data[..., 2:], np.log(d.data[..., 2:])))
    # targets = np.concatenate((d.data[..., :2], d.data[..., :2]))
    return RData((questions, targets), cross_val=crossvalrate, indeps_n=2, header=False, pca=pca)


def build_network(data):
    """Generates a neural network from given hyperparameters"""
    net = Network(data=data, eta=eta, lmbd=lmbd, cost=cost)
    for hl in hiddens:
        net.add_fc(hl, activation=activationH)

    net.finalize_architecture(activation=activationO)
    return net


def run1(queue=None, return_dynamics=False, dump=False, ID=0, verbose=False):
    """One run corresponds to the training of a network with randomized weights"""
    path = {"fcv": fcvpath, "bur": burleypath, "ful": fullpath}[what.lower()[:3]]
    myData = pull_data(path)
    network = build_network(myData)
    dynamics = [list(), list()]

    for e in range(1, epochs + 1):
        network.learn(batch_size=batch_size)
        if e % (epochs // no_plotpoints) == 0:
            terr = wgs_test(network, "testing")
            lerr = wgs_test(network, "learning")
            dynamics[0].append(terr)
            dynamics[1].append(lerr)
            if e % ((epochs // no_plotpoints) * 10) == 0 and verbose:
                print(str(ID) + " / e: {}: T: {} L: {}".format(e, terr, lerr))

    output = (dynamics[0][-1], dynamics[1][-1]) if not return_dynamics else dynamics

    if dump:
        dump_wgs_prediction(network, "testing", ID)

    if queue:
        queue.put(output)
    else:
        return output


def mp_run(jobs=mp.cpu_count(), return_dynamics=False, dump=False, verbose=False):
    """Organizes multiple runs in parallel"""
    myQueue = mp.Queue()
    procs = [mp.Process(target=run1, args=(myQueue, return_dynamics, dump, i, verbose),
                        name="P{}".format(i)) for i in range(jobs)]
    results = []
    for proc in procs:
        proc.start()
    while len(results) != jobs:
        results.append(myQueue.get())
        time.sleep(0.33)
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
        res = mp_run(jobs=jobs, return_dynamics=False, dump=True)
        res = list(zip(*res))
        results[0].extend(res[0])
        results[1].extend(res[1])
        ch = "Acc@{}:\tT: {}\tL: {}\n".format(r*jobs, int(np.mean(results[0])), int(np.mean(results[1])))
        logchain += ch
        print(ch[:-1])
    tfin = np.mean(results[0]), np.std(results[0])
    lfin = np.mean(results[1]), np.std(results[1])
    print("----------\nExperiment ended.")
    print("T: mean: {} STD: {}".format(*tfin))
    print("L: mean: {} STD: {}".format(*lfin))
    logchain += "---------------\nFinal Tests:\n"
    logchain += "T: mean: {} STD: {}\n".format(*tfin)
    logchain += "L: mean: {} STD: {}\n".format(*lfin)
    logchain += "Hypers: crossvalrate, pca, eta, lmbd, hiddens, activationO, activationH, cost, epochs, batch_size:\n"
    logchain += str([crossvalrate, pca, eta, lmbd, hiddens, activationO, activationH, cost, epochs, batch_size]) + "\n"
    logchain += "Run time: {}s\n".format(time.time()-start)
    logf = open("logs/Rlog{}.txt".format(runs), "w")
    logf.write(logchain)
    logf.close()


def plotted_run():
    """This experimental setup plots the run dynamics of some epochs

    The generated diagrams must be saved manually."""
    X = np.arange(no_plotpoints) * (epochs // no_plotpoints)
    f, axarr = plt.subplots(no_plots, sharex=True)

    dynamics = mp_run(jobs=no_plots, return_dynamics=True, dump=True, verbose=True)

    for i in range(no_plots):
        axarr[i].plot(X, dynamics[i][0], "r", label="T")
        axarr[i].plot(X, dynamics[i][1], "b", label="L")
        axarr[i].axis([0, X[-1], 0.0, 3000.0])
        axarr[i].annotate('%0.0f' % dynamics[i][0][-1], xy=(1, dynamics[i][0][-1]), xytext=(8, 0),
                          xycoords=('axes fraction', 'data'), textcoords='offset points')
        if i == 0:
            # axarr[i].annotate('%0.0f' % dynamics[i][0][-1], xy=(1, dynamics[i][0][-1]), xytext=(8, 0),
            #                   xycoords=('axes fraction', 'data'), textcoords='offset points')
            axarr[i].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                            ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


if __name__ == '__main__':
    plotted_run()
    print("Fin")
