"""共通fixutureインスタンス
テストに必要な関数をまとめる
"""

import numpy
import pytest
from decimal import Decimal, ROUND_HALF_UP

class TestUtils():

    def __init__(self):
        super().__init__()

    def assert_by_compare_array(self, array1, array2, position_log_message='', num_of_decimal='0.00000001', enable_log=True):
        """配列（numpy配列、リスト）の比較を行います。
        Args:
            array1(ndarray, list) : 比較対象配列１
            array2(ndarray, list) : 比較対象配列２
            position_log_message(str, optional) : 配列ポジション情報（複数層配列の場合に再帰呼び出しする際に利用。指定は不要
            num_of_decimal(str, optional) : 比較時の少数桁数。デフォルト'0.00000001'（少数１桁であれば'0.1'と指定）
            enable_log(boolean, optional) : ログ出力。デフォルトFalse
        """

        for index, (elem_1, elem_2) in enumerate(zip(array1, array2)):
            if isinstance(elem_1, (numpy.ndarray, list)):
                self.assert_by_compare_array(elem_1 , elem_2
                    , '{0}:{1}'.format(position_log_message, index) if position_log_message != '' else str(index)
                    , num_of_decimal=num_of_decimal , enable_log=enable_log)
            else:
                val_1 = Decimal(str(elem_1)).quantize(Decimal(num_of_decimal), rounding=ROUND_HALF_UP)
                val_2 = Decimal(str(elem_2)).quantize(Decimal(num_of_decimal), rounding=ROUND_HALF_UP)
                if enable_log:
                    print('ARRAY1 : ({0}) {1}'.format(
                        '{0}:{1}'.format(position_log_message, index) if position_log_message != '' else str(index)
                        , val_1))
                    print('ARRAY2 : ({0}) {1}'.format(
                        '{0}:{1}'.format(position_log_message, index) if position_log_message != '' else str(index)
                        , val_2))
                assert val_1 == val_2

@pytest.fixture(scope='session')
def test_utils():
    yield TestUtils()