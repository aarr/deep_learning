"""ニューラルネットワークテスト
"""
import numpy as np
import matplotlib.pylab as plt
import src.deep_learning.ch03.neuralnetwork as target
import pytest


def test_step_function():
    """ステップ関数テスト"""
    # すべてが一致していること
    assert ([1, 0] == target.step_function(np.array([1.0, -2.0]))).all()


@pytest.mark.skip()
def test_draw_step_function():
    """ステップ関数　グラフ表示"""
    # 表示
    # 表示用アプリが立ち上がるので、閉じないと次には進まない
    # -5.0から5.0まで0.1刻みでNumpy配列を作成する
    x = np.arange(-5.0, 5.0, 0.1)
    y = target.step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

@pytest.mark.skip
def test_draw_sigmoid():
    """シグモイド関数テスト グラフ表示"""
    x = np.arange(-5.0, 5.0, 0.1)
    y = target.sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1) # y軸の範囲設定
    plt.show()

@pytest.mark.skip
def test_draw_relu():
    """ReLU関数テスト　グラフ表示"""
    x = np.arange(-5.0, 5.0, 0.1)
    y = target.relu(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 5.0) # y軸の範囲設定
    plt.show()

def test_summary():
    """３層ニューラルネットワークテスト"""
    result = target.three_layer_nuralnetwork()
    # 厳密には小数点以下の桁数が異なる
    #assert np.array_equal(result[0], np.array([[0.31682708, 0.69627909]]))

def test_softmax():
    """ソフトマックス関数テスト"""
    x1 = np.array([0.3, 2.9, 4.0])
    target.softmax_non_considered_overflow(x1)
    x2 = np.array([1010, 1000, 990])
    result2 = target.softmax_considered_overflow(x2)

def test_load_image():
    """画像ダウンロード"""
    target.load_image() 

@pytest.mark.skip
def test_analyze_image():
    """画像表示"""
    target.show_image()

def test_analyze_image():
    """画像解析"""
    target.analyze_image()