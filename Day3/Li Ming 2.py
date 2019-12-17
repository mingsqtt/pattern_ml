import math
import numpy as np


def makeSteps(dat, length, dist):
    total_len = len(dat)
    nrow = int(math.ceil((total_len - length + 1) / dist))
    out = np.zeros(nrow * length)
    for i in range(nrow):
        out[i*length:(i + 1)*length] = dat[i * dist:(i * dist + length)]
    return out.reshape(-1, length)


dat = np.linspace(0, 12, 13)
print(makeSteps(dat, 5, 3))
