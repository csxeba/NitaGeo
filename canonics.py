from csxdata import RData, roots
from csxdata.utilities.parsers import parse_csv
from csxdata.utilities.high_utils import plot


PRETREAT = "std"


def pull_xy_data():
    X, Y, header = parse_csv(roots["csvs"] + "sum_ntab2.csv", headers=1, indeps=7)

    data = RData((X, Y[:, -2:].astype("float32")), cross_val=0.0, indeps_n=2, header=header)
    data.transformation = PRETREAT

    cities = Y[:, -3]

    return data.table("learning"), cities


def canonical_approach():
    from sklearn.cross_decomposition import CCA

    (X, Y), cities = pull_xy_data()

    cca = CCA(n_components=2)
    cca.fit(X, Y)

    ccaX, ccaY = cca.transform(X, Y)

    plot(ccaX, cities, ["CC01", "CC02", "CC03"], 1)

    return "OK What Now?"


def pls_approach():
    from sklearn.cross_decomposition import PLSRegression

    (X, Y), cities = pull_xy_data()

    pls = PLSRegression()
    pls.fit(X, Y)

    plsX, plsY = pls.transform(X, Y)

    plot(plsX, cities, ["Lat01", "Lat02", "Lat03"], ellipse_sigma=1)

    return "OK What Now?"


if __name__ == '__main__':
    canonical_approach()
