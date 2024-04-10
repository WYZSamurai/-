import torch
import pattern


def msll(Fdb: torch.Tensor):
    maxi = Fdb.argmax()
    indexr = torch.where(Fdb[maxi:].diff() > 0)[0][0]
    indexl = torch.where(Fdb[:maxi].flip(0).diff() > 0)[0][0]
    Fdb[maxi-indexl:maxi+indexr] = -60
    mFdb = Fdb.max()
    MSLL = mFdb
    return MSLL


if __name__ == "__main__":
    theta_min = -90.0
    theta_max = 90.0
    (l, delta, theta_0) = (1, 360, 0)
    d = l/2
    G = 5
    NP = 100
    m = 1
    n = 1
    L = 20
    Pc = 0.8
    Pm = 0.050

    dna = torch.randint(0, 2, (m, n, L)).to(dtype=torch.float)

    Fdb = pattern.pattern(dna, l, d, delta, theta_0)
    MSLL = msll(Fdb)
    print(MSLL)
