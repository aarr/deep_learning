"""ニューラルネットワーク
"""

import numpy as np
from ..com.utils import log


def step_function(x):
    # xは配列を想定。下記式により、0より大きな要素はすべてTrueとなる。
    tmp = x > 0
    log('Temp Result : {0}'.format(tmp))
    # Trueの要素のみ1に変換
    y = tmp.astype(np.int)
    log('Result : {0}'.format(y))
    return y

