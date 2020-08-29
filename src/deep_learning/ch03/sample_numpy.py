"""Numpyを利用した行列計算のサンプル
"""

import numpy as np
from ..com.utils import log

def dot():
    """行列掛け算
    ３章より
    """
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[1, 2], [3, 4], [5, 6]])
    log('行列数 x : {0}, y : {1}'.format(x, y))
    log('行列Shape x : {0}, y : {1}'.format(x.shape, y.shape))
    result = np.dot(x, y)
    log('行列掛算 result : {0}'.format(result))

def sum_axis(val_axis):
    array = [
        [
            [1, 2, 3],
            [10, 20, 30],
            [100, 200, 300]
        ],
        [
            [3, 2, 1],
            [30, 20, 10],
            [300, 200, 100]
        ]
    ]
    log('Array : \n{0}'.format(array))
    return np.sum(array, axis=val_axis)

