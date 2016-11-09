import time
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

from csxnet import Network
from csxnet.util import cost_fns as costs
from csxnet.util import act_fns as activations

from generic import pull_old_data as pull_data

fcvdataparam, fcvnetparam, fcvrunparam = (0.2, 10), (0.3, 0.0, (100, 30),
                                                     activations.sigmoid, activations.sigmoid, costs.mse), \
                                         (10, 2, 10000, 20)
burleydataparam, burleynetparam, burleyrunparam = (0.2, 10), (0.3, 0.0, (30, 30),
                                                              activations.sigmoid, activations.sigmoid, costs.mse), \
                                                  (10, 2, 10000, 20)
displayparams = 200, 2  # no_plotpoints, no_plots

millenia = 10


class CsxModel:
    def __init__(self, dataparameters, netparameters, runparameters):
        self.netparams = netparameters  # eta, lmbd, hiddens, activationO, activationH, cost
        self.dataparams = dataparameters  # crossval, pca
        self.runparams = runparameters  # runs, jobs, epochs, batch_size

    def build_network(self):
        """Generates a neural network from given hyperparameters"""
        eta, lmbd, hiddens, activationO, activationH, cost = self.netparams
        network = Network(data=pull_data(*self.dataparams),
                          eta=eta, lmbd2=lmbd, lmbd1=0.0, mu=0.0, cost=cost)
        for hl in hiddens:
            network.add_fc(hl, activation=activationH)

        network.finalize_architecture(activation=activationO)
        return network

    def mp_run(self, jobs=mp.cpu_count(), return_dynamics=False, dump=False, verbose=False):
        """Organizes multiple runs in parallel"""
        myQueue = mp.Queue()
        procs = [mp.Process(target=self.run1, args=(myQueue, return_dynamics, dump, i, verbose),
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

    def logged_run(self):
        """This experimental setup logs the performance after several runs

        Writes the mean and STD of the recorded accuracies to Rlog.txt"""

        runs, no_plotpoints, no_plots, jobs, epochs, batch_size = self.runparams

        start = time.time()
        results = [list(), list()]
        logchain = ""
        for r in range(1, runs+1):
            res = self.mp_run(jobs=jobs, return_dynamics=False, dump=True)
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
        logchain += "crossval\tpca\teta\tlmbd\thiddens\tactivationO\t" + \
                    "activationH\tcost\truns\tjobs\tepochs\tbatch_size:\n"
        logchain += ("\t".join(self.dataparams) + "\t" +
                     "\t".join(self.netparams) + "\t" +
                     "\t".join(self.runparams) + "\n")
        logchain += "Run time: {}s\n".format(time.time()-start)
        logf = open("logs/Rlog{}.txt".format(runs), "w")
        logf.write(logchain)
        logf.close()

    def get_models(self):
        eta, lmbd, hiddens, activationO, activationH, cost = self.netparams
        runs, jobs, epochs, batch_size = self.runparams
        pca = self.dataparams[1]

        names = "Archimedes", "Avis", "Pallas"
        myData = pull_data(0.0, pca)
        models = [Network(myData, eta, 0.0, lmbd, 0.0, cost) for _ in range(3)]
        for no, net in enumerate(models):
            net.name = names[no]
            print("Building {}...".format(net.name))
            for h in hiddens:
                net.add_fc(h)
            net.finalize_architecture()
            print("Training {}...".format(net.name))
            for millenium in range(1, millenia+1):
                net.fit(batch_size, epochs=epochs // 1000, verbose=0)
                print("{} cost @ epoch {}: {}".format(net.name, millenium * 1000, net.evaluate(accuracy=False)))
            print("-----------------------\nDone Training {}".format(net.name))
            net.describe(1)
            print("Dumping {}".format(net.name))
            net.save("models/" + net.name + ".bro")
            print("Saved!")

    def plotted_run(self):
        """This experimental setup plots the run dynamics of some epochs

        The generated diagrams must be saved manually."""
        no_plotpoints, no_plots = displayparams
        runs, jobs, epochs, batch_size = self.runparams

        X = np.arange(no_plotpoints) * (epochs // no_plotpoints)
        f, axarr = plt.subplots(no_plots, sharex=True)

        dynamics = self.mp_run(jobs=no_plots, return_dynamics=True, dump=True, verbose=True)

        for i in range(no_plots):
            axarr[i].plot(X, dynamics[i][0], "r", label="T")
            axarr[i].plot(X, dynamics[i][1], "b", label="L")
            axarr[i].axis([0, X[-1], 0.0, 5000.0])
            axarr[i].annotate('%0.0f' % dynamics[i][0][-1], xy=(1, dynamics[i][0][-1]), xytext=(8, 0),
                              xycoords=('axes fraction', 'data'), textcoords='offset points')
            if i == 0:
                # axarr[i].annotate('%0.0f' % dynamics[i][0][-1], xy=(1, dynamics[i][0][-1]), xytext=(8, 0),
                #                   xycoords=('axes fraction', 'data'), textcoords='offset points')
                axarr[i].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                                ncol=2, mode="expand", borderaxespad=0.)
        plt.show()

    def run1(self, queue=None, return_dynamics=False, dump=False, ID=0, verbose=False):
        """One run corresponds to the training of a network with randomized weights"""

        network = self.build_network()

        runs, no_plotpoints, no_plots, jobs, epochs, batch_size = self.runparams
        dynamics = [list(), list()]

        for e in range(1, epochs + 1):
            network._epoch(batch_size=batch_size)
            if e % (epochs // no_plotpoints) == 0:
                terr = self.wgs_test(network, "testing")
                lerr = self.wgs_test(network, "learning")
                dynamics[0].append(terr)
                dynamics[1].append(lerr)
                if e % ((epochs // no_plotpoints) * 10) == 0 and verbose:
                    print(str(ID) + " / e: {}: T: {} L: {}".format(e, terr, lerr))

        output = (dynamics[0][-1], dynamics[1][-1]) if not return_dynamics else dynamics

        if dump:
            self.dump_wgs_prediction(network, "testing", ID)

        if queue:
            queue.put(output)
        else:
            return output

    @staticmethod
    def wgs_test(network, on):
        """Test the network's accuracy

        by computing the haversine distance between the target and the predicted coordinates"""
        from csxdata.utilities.nputils import haversine
        m = network.data.n_testing
        d = network.data
        questions = {"d": d.data, "l": d.learning, "t": d.testing}[on[0]][:m]
        ideps = {"d": d.indeps, "l": d.lindeps, "t": d.tindeps}[on[0]][:m]
        usideps = d.upscale(ideps)
        preds = network.predict(questions)
        uspreds = d.upscale(preds)
        distance = haversine(usideps, uspreds)
        return int(np.mean(distance))

    @staticmethod
    def dump_wgs_prediction(network: Network, on, ID):
        """Dumps the coordinate predictions into a text file"""
        m = network.data.n_testing
        d = network.data
        questions = {"d": d.data, "l": d.learning, "t": d.testing}[on[0]][:m]
        ideps = d.upscale({"d": d.indeps, "l": d.lindeps, "t": d.tindeps}[on[0]][:m])
        preds = d.upscale(network.predict(questions))
        np.savetxt("logs/R" + str(ID) + on + '_ideps.txt', ideps, delimiter="\t")
        np.savetxt("logs/R" + str(ID) + on + '_preds.txt', preds, delimiter="\t")

if __name__ == '__main__':
    model = CsxModel(fcvdataparam, fcvnetparam, fcvrunparam)
    model.run1()
