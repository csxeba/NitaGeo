def pull_old_data(crossval_rate, pca, path=None):
    """Pulls the learning data from a csv file."""
    from csxdata.frames import CData, RData
    if path is None:
        from csxdata.const import roots
        path = roots["csvs"] + "fullnyers.csv"
    d = CData(path, cross_val=0.0, pca=0, header=True, sep="\t", end="\n")
    questions = d.data[..., 2:] + 1e-8
    targets = d.data[..., :2]
    return RData((questions, targets), cross_val=crossval_rate, indeps_n=2, header=False, pca=pca)


def pull_new_data(crossval_rate, pca, path=None):
    from csxdata.frames import RData
    from csxdata.utilities.parsers import parse_csv
    if path is None:
        from csxdata.const import roots
        path = roots["csvs"] + "sum_ntab.csv"
    X, _, header = parse_csv(path, headers=1, indeps=4, sep="\t", end="\n")
    return RData((X[..., 2:], X[..., :2]), crossval_rate, indeps_n=0, header=0, pca=pca)
