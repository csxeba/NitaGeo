import numpy as np


def test_featscale():
    from csxdata.utilities.nputils import featscale

    ar = np.array([np.random.uniform(12, 21) for _ in range(1000)]).reshape(250, 4)
    fsar, feats = featscale(ar, 0.1, 0.9, 0, 1)
    rsar = featscale(ar, feats[0], feats[1], 0, 0)
    res = np.sum(rsar - ar)
    assert res == 0.0, "featscale test failed! Result: {}".format(res)


if __name__ == '__main__':
    test_featscale()
