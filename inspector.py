import sys
import pickle

import numpy as np

from csxnet.brainforge.Architecture.NNModel import Network

dataroot = "D:/Data/" if sys.platform == "win32" else "/data/Prog/data/"


def wake(path):
    with open(path, "rb") as fl:
        network = pickle.load(fl)
        fl.close()
    print("{} is awakened!".format(network.name))
    network.describe(1)
    return network


def weight_space_distance(model1: Network, model2: Network):
    print("\nInspecting {} and {}.".format(model1.name, model2.name))
    wts1, wts2 = [[l.weights for l in m.layers[1:]] for m in (model1, model2)]
    for i, (wt1, wt2) in enumerate(zip(wts1, wts2)):
        print("Distance of L{}:".format(i+2))
        print(np.sum(np.abs(np.subtract(wt1, wt2))))

print()

brainroot = dataroot + "brains/"
archimedes = "Archimedes.bro"
avis = "Avis.bro"
pallas = "Pallas.bro"

models = wake(brainroot + archimedes), wake(brainroot + avis), wake(brainroot + pallas)
archimedes, avis, pallas = models

weight_space_distance(archimedes, avis)
weight_space_distance(archimedes, pallas)
weight_space_distance(avis, pallas)

print("Fin!")
