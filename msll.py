import torch


def msll(Fdb: torch.Tensor):
    maxi = Fdb.argmax().item()

    point = 0
    while Fdb[maxi+point] >= Fdb[maxi+1+point]:
        point = point+1
    mrFdb = Fdb[maxi+point:].max()
    point = 0
    while Fdb[maxi-point] >= Fdb[maxi-1-point]:
        point = point+1
    mlFdb = Fdb[:maxi-point].max()
    if mlFdb > mrFdb:
        mFdb = mlFdb
    else:
        mFdb = mrFdb
    MSLL = mFdb
    return MSLL


if __name__ == "__main__":
    pass
