"""ニューラルネットワーク
ニューラルネットワークで利用する活性関数３種の実装
"""

import sys
import os
import numpy as np
from PIL import Image
from ..com.utils import log
from ..dataset.mnist_copy import *


def step_function(x):
    """ステップ関数
    0以上であればすべて
    """
    # xは配列を想定。下記式により、0より大きな要素はすべてTrueとなる。
    tmp = x > 0
    log('Temp Result : {0}'.format(tmp))
    # Trueの要素のみ1に変換
    y = tmp.astype(np.int)
    log('Result : {0}'.format(y))
    return y

def sigmoid(x, enable_log=True):
    """シグモイド関数
    ニューラルネットワークで古くから利用されていた活性化関数
    """
    if enable_log:
        log('X : {0}'.format(x))
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU関数
    最近利用されることの多くなった活性化関数
    """
    log('x : {0}'.format(x))
    return np.maximum(0, x)


def three_layer_nuralnetwork():
    """３層ニューラルネットワーク
    """
    def inilabel_network():
        network = {}
        network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        network['b1'] = np.array([[0.1, 0.2, 0.3]])
        network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        network['b2'] = np.array([0.1, 0.2])
        network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
        network['b3'] = np.array([0.1, 0.2])

        return network
    
    def identity_function(x):
        """恒等関数
        出力層の活性化関数。一旦そのままの値を返却する
        """
        return x
    
    def forward(network, x):
        """伝播"""
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']

        # A = XW + B
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = identity_function(a3)
        
        return y
    
    network = inilabel_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    log('Result : {0}'.format(y))
    return y


def softmax_non_considered_overflow(a, enable_log=True):
    """ソフトマックマックス関数
    出力層の活性化関数
    """
    exp_a = np.exp(a)
    sum_exp = np.sum(exp_a)
    y = exp_a / sum_exp
    if enable_log:
        log('Result : {0} \nSUM(Result) : {1}'.format(y, np.sum(y)))
    return y


def softmax_considered_overflow(a, enable_log=True):
    """ソフトマックス関数（桁あふれ考慮）
    出力層の活性化関数
    """
    c = np.max(a)
    exp_a = np.exp(a - c) # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    if enable_log:
        log('Result : {0} \nSUM(Result) : {1}'.format(y, np.sum(y)))
    return y


def load_image():
    """画像ダウンロード"""
    # (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    (img_train, label_train), (img_test, label_test) = load_mnist(flatten=True, normalize=False)
    log('[DATA]\nimg_train : {0}\nlabel_train : {1}\nimg_test : {2}\nlabel_test : {3}'.format(img_train, label_train, img_test, label_test))
    log('[SHAPE]\nimg_train : {0}\nlabel_train : {1}\nimg_test : {2}\nlabel_test : {3}'.format(img_train.shape, label_train.shape, img_test.shape, label_test.shape))
    return (img_train, label_train), (img_test, label_test)


def show_image():
    """画像解析"""

    def show_image(image):
        """画像表示"""
        pil_image = Image.fromarray(np.uint8(image))
        pil_image.show()

    (img_train, label_train), (img_test, label_test) = load_image();
    image = img_train[0]
    label = label_train[0]
    image_reshape = image.reshape(28, 28) # 1 * 784のデータを28 * 28に整形
    log('Lable : {0}\nShape : {1}\nReSHape : {2}'.format(label, image.shape, image_reshape.shape))
    show_image(image_reshape)


def analyze_image():
    """画像解析"""

    def get_testdata():
        """テストデータ取得"""
        (img_train, label_train), (img_test, label_test) = load_image()
        return (img_test, label_test)


    def init_network():
        """学習済み重みパラメータ取得"""
        with open(os.path.join(os.path.dirname(__file__), 'sample_weight.pkl'), 'rb') as f:
            network = pickle.load(f)
        return network

    def predict(network, x):
        """伝播"""
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']

        # A = XW + B
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1, enable_log=False)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2, enable_log=False)
        a3 = np.dot(z2, W3) + b3
        y = softmax_considered_overflow(a3, enable_log=False)
        #y = softmax_non_considered_overflow(a3, enable_log=False)
        
        return y

    # テストデータ、重み取得
    data, label = get_testdata()
    network = init_network()
    # 解析
    accuracy_cnt = 0
    for i in range(len(data)):
        y = predict(network, data[i])
        p = np.argmax(y)

        if p == label[i]:
            accuracy_cnt += 1
    # 正解率出力
    log('Accuracy : {0}'.format(str(float(accuracy_cnt) / len(data))))

