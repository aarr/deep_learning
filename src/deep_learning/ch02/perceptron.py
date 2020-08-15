"""パーセプトロン
"""
import sys
from ..com.utils import log 
import numpy as np


def AND(x1, x2):
    """ANDゲートによるパーセプトロン実装
    """
    w1, w2, theta = 0.5, 0.5, 0.7   # 重み、閾値
    tmp = x1 * w1 + x2 * w2
    log("Temp Result : {0}".format(tmp))

    # 閾値と比較して、0 or 1 を返却
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


def AND_BIAS(x1, x2):
    """ANDゲートによるパーセプトロン（バイアス付き）
    """
    x = np.array([x1, x2])      # 入力
    w = np.array([0.5, 0.5])    # 重み
    b = -0.7                    # バイアス（閾値 -theta）
    log("Input * Weight : {0}".format(w * x))
    tmp = np.sum(w * x) + b
    log("Temp Result : {0}".format(tmp))
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    """NANDによるパーセプトロン
    """
    # ANDとは重みとバイアスの値が異なるだけ
    x = np.array([x1, x2])      # 入力
    w = np.array([-0.5, -0.5])  # 重み
    b = 0.7                     # バイアス
    tmp = np.sum(w * x) + b
    log("Temp Result : {0}".format(tmp))
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    """ORによるパーセプトロン
    """
    # ANDとはバイアスのみ異なるだけ
    x = np.array([x1, x2])      # 入力
    w = np.array([0.5, 0.5])    # 重み
    b = -0.2                    # バイアス
    tmp = np.sum(w * x) + b
    log("Temp Result : {0}".format(tmp))
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    log('START')
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    log('END')
    return y