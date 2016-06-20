import sys
import pickle

import numpy as np

from csxnet.datamodel import RData
from csxnet.brainforge.Architecture.NNModel import Network
from csxnet.utilities import pull_table


dataroot = "D:/Data/" if sys.platform == "win32" else "/data/Prog/data/"


def wake(path):
    with open(path, "rb") as fl:
        network = pickle.load(fl)
        fl.close()
    print("{} is awakened!".format(network.name))
    network.describe(1)
    return network


def get_data(path=None):
    if path is None:
        from tkinter import Tk
        import tkinter.filedialog as tkfd
        tk = Tk()
        tk.withdraw()
        path = tkfd.askopenfilename(title="Please open the csv containing the data!",
                                    initialdir=csvroot)
        tk.destroy()

    return pull_table(path, header=True, labels=True)


def dump_wgs_prediction(net: Network, questions: np.ndarray, labels: np.ndarray=None):
    """Dumps the coordinate predictions into a text file"""
    questions = net.data.pca.transform(questions)
    preds = net.data.upscale(net.predict(questions)).astype(str)
    preds = np.concatenate((labels, preds), axis=1)
    preds.tolist()
    preds = ["\t".join(pr.tolist()) for pr in preds]
    chain = "\n".join(preds)
    with open("logs/" + net.name + "_predictions.csv", "w") as f:
        f.write(chain)
        f.close()


brainroot = dataroot + "brains/"
archimedes = "Archimedes.bro"
avis = "Avis.bro"
pallas = "Pallas.bro"

csvroot = dataroot + "csvs/"

models = wake(brainroot + archimedes), wake(brainroot + avis), wake(brainroot + pallas)
# table, header, labels = get_data(csvroot + "unknown_tobacco01.csv")
table, header, labels = get_data()
labels = np.array(labels).reshape((88, 1))

for model in models:
    dump_wgs_prediction(model, table, labels=labels)

print("Fin!")
