import time

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Highway

from generic import pull_new_data

from csxdata.utilities.high_utils import th_haversine

# Model parameters:
OPTIMIZER = "rmsprop"
ACTIVATION = "tanh"
O_ACTIVATION = "sigmoid"
HIDDENS = ["300d", "180d", "120d", 60, 30]
HIGHWAY = False
EPOCHS = 100
BSIZE = 10
VAL_RATE = 0.3
DROP_RATE = 0.5
PCA = 0
L1 = 0.0
L2 = 0.01


class KerasModel:
    def __init__(self, name=""):

        def parse_data():
            dframe = pull_new_data(crossval_rate=VAL_RATE, pca=PCA)
            if not PCA:
                dframe.transformation = "std"
            neurons_in, neurons_out = dframe.neurons_required
            return dframe, neurons_in, neurons_out

        def build_keras_network(neurons_in, neurons_out):

            def add(network, h, input_dim=None, activation=ACTIVATION):
                drop = False
                if isinstance(h, str):
                    if h[-1] == "d":
                        h = int(h[:-1])
                        drop = True
                network.add(Dense(h, input_dim=input_dim, activation=activation))
                if drop:
                    network.add(Dropout(p=DROP_RATE))
                if HIGHWAY:
                    network.add(Highway())

            def define_model():
                add(model, HIDDENS[0], input_dim=neurons_in)
                if len(HIDDENS) > 1:
                    for neurons in HIDDENS[1:]:
                        add(model, neurons)
                add(model, neurons_out, activation="sigmoid")

            model = Sequential()
            define_model()
            model.compile(optimizer=OPTIMIZER, loss=th_haversine())
            return model

        self.dataframe, fanin, outshape = parse_data()
        self.network = build_keras_network(fanin, outshape)

        self.name = name
        self.age = 0

    def run(self, X, y):
        self.network.fit(X, y, batch_size=BSIZE, nb_epoch=100, verbose=0)
        wgs_lerr = self.wgs_test("learning")
        wgs_terr = self.wgs_test("testing")
        self.age += 1
        print("WGS error     @\t{}:\tL: {}\tT: {}"
              .format(self.age, wgs_lerr, wgs_terr))
        return wgs_lerr, wgs_terr

    def run_century(self):
        learning_table = self.dataframe.table("learning")
        print("Initial error @\t0:\tL: {}\tT: {}".format(self.wgs_test("learning"), self.wgs_test("testing")))
        runstart = time.time()

        lacc, tacc = [], []
        for cent in range(1, 101):
            try:
                la, ta = self.run(*learning_table)
                lacc.append(la)
                tacc.append(ta)
            except KeyboardInterrupt:
                break

        print("Learning finished, it took {} seconds!".format(int(time.time() - runstart)))
        self.network.summary()
        return lacc, tacc

    def wgs_test(self, on):
        """Test the network's accuracy

        by computing the haversine distance between the target and the predicted coordinates"""
        from csxdata.utilities.nputils import haversine
        X, y = self.dataframe.table(on, m=self.dataframe.n_testing)
        scaled_y = self.dataframe.upscale(y)
        preds = self.network.predict(X)
        scaled_preds = self.dataframe.upscale(preds)
        distance = haversine(scaled_y, scaled_preds)
        return int(np.mean(distance))

    def wgs_predict(self, questions: np.ndarray, labels: np.ndarray=None, y: np.ndarray=None):
        """Dumps the coordinate predictions into a text file"""
        from csxdata.utilities.nputils import haversine
        preds = self.network.predict(questions)
        dist = haversine(self.dataframe.upscale(preds), y) if y is not None else None
        preds = self.dataframe.upscale(preds).astype(str)
        labels = np.atleast_2d(labels).T
        preds = np.concatenate((labels, preds), axis=1)
        if y is not None:
            preds = np.concatenate((preds, y.astype(str), np.atleast_2d(dist).T.astype(str)), axis=1)
        preds.tolist()
        preds = ["\t".join(pr.tolist()) for pr in preds]
        chain = "\n".join(preds)
        chain = chain.replace(".", ",")
        header = ["Azon", "Y", "X"]
        if y is not None:
            header += ["real_Y", "real_X", "Haversine"]
        chain = "\t".join(header) + "\n" + chain
        with open("logs/" + self.name + "_predictions.csv", "w") as f:
            f.write(chain)
            f.close()

    def __call__(self):
        return self.run_century()


if __name__ == '__main__':
    km = KerasModel("Kerberos")
    km()
