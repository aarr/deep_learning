"""ニューラルネットワークテスト
"""

from decimal import Decimal, ROUND_HALF_UP
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


# conftest.pyで定義しているfixutureでテスト用関数を用意
def test_summary(test_utils):
    """３層ニューラルネットワークテスト"""
    result = target.three_layer_nuralnetwork()
    excepted_value = [[0.31682708, 0.69627909]]
    test_utils.assert_by_compare_array(result, excepted_value, enable_log=True)


def test_softmax(test_utils):
    """ソフトマックス関数テスト"""
    # オーバーフロー考慮なし
    x1 = np.array([0.3, 2.9, 4.0])
    result1 = target.softmax_non_considered_overflow(x1)
    test_utils.assert_by_compare_array(result1, [0.01821127, 0.24519181, 0.73659691])
    # オーバーフロー考慮あり
    x2 = np.array([1010, 1000, 990])
    result2 = target.softmax_considered_overflow(x2)
    test_utils.assert_by_compare_array(result2, [9.99954600e-01, 4.53978686e-05, 2.06106005e-09])


def test_load_image():
    """画像ダウンロード"""
    target.load_image() 


@pytest.mark.skip
def test_analyze_image():
    """画像表示"""
    target.show_image()


def test_analyze_image():
    """画像解析"""
    analyze_logic = target.AnalyzeImage()
    accuracy = analyze_logic.analyze_image()
    # データセットの内容が変わったのか本の回答とは値が合わない
    assert accuracy == 0.9352

    accuracy_batch = analyze_logic.analyze_image_batch()
    # データセットの内容が変わったのか本の回答とは値が合わない
    # 公開されているサンプルソースではsoftmax関数にて、axisを指定してmaxを取得したりしている
    # そのため、更に答えはズレが出てくると思われる
    assert accuracy_batch == 0.9352


