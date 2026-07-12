"""テスト用の共通セットアップ。

CI (GPU なし・torch なし) でも純関数とキュー制御ロジックをテストできるよう、
重量級・環境依存の依存ライブラリを import 前にスタブへ差し替える。
テスト対象は音声・推論そのものではなく、その周辺ロジックのみ。
"""

import sys
import types
from unittest.mock import MagicMock

# ollama.ResponseError は except 節で参照されるため、実際の例外クラスが必要
_ollama_stub = types.ModuleType("ollama")
_ollama_stub.ResponseError = type("ResponseError", (Exception,), {})
_ollama_stub.chat = MagicMock()
_ollama_stub.show = MagicMock()

for _name, _module in {
    "whisper": MagicMock(),
    "ollama": _ollama_stub,
    "speech_recognition": MagicMock(),
    "plyer": MagicMock(),
    "winotify": MagicMock(),
}.items():
    sys.modules[_name] = _module
