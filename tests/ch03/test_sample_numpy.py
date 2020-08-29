"""Numpyサンプルテスト
"""
import src.deep_learning.ch03.sample_numpy as target
import pytest

def test_dot():
    """行列掛け算サンプル
    """
    target.dot()

def test_sum_axis():
    # axisで指定された次元が消える
    result_axis0 = target.sum_axis(0)
    print("SUM axis=0 : \n{0}".format(result_axis0))

    result_axis1 = target.sum_axis(1)
    print("SUM axis=1 : \n{0}".format(result_axis1))

    result_axis2 = target.sum_axis(2)
    print("SUM axis=2 : \n{0}".format(result_axis2))
