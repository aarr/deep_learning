"""ニューラルネットワークテスト
"""
import numpy as np
import matplotlib.pylab as plt
import src.deep_learning.ch03.neuralnetwork as target


def test_step_function():
    """ステップ関数テスト
    """
    # すべてが一致していること
    assert ([1, 1] == target.step_function(np.array([1.0, 2.0]))).all()


def test_draw_step_function():
    """ステップ関数　グラフ表示
    """
    # 表示
    x = np.arange(-5.0, 5.0, 0.1)
    y = target.step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

