"""ユーティリティ
"""

import inspect

def log(message):
    """ログ出力を行う
    [呼出元関数名] メッセージ
    """
    current_frame = inspect.currentframe()
    call_frame = inspect.getouterframes(current_frame, 2)
    print('[{0}] \r\n{1}\r\n'.format(call_frame[1][3], message))
