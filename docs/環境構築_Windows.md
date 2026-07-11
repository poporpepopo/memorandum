# 環境構築（Windows 版）

`memorandum.py` は macOS / Windows 共通のコードで動作します。ここでは Windows での手順をまとめます。

## 前提

- Windows 10 / 11
- Python 3.9 以上（動作確認: 3.10）
- [Ollama for Windows](https://ollama.com/) がインストール済みであること

## 1. セットアップ（初回のみ）

リポジトリのフォルダで PowerShell を開き、以下を実行します。

```powershell
# 仮想環境の作成と依存ライブラリのインストール
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt

# 要約用モデルのダウンロード (約9.6GB)
ollama pull gemma4:e4b
```

メモリ 8GB クラスのマシンでは軽量な `gemma4:e2b`（約7.2GB）を推奨します。
その場合は `ollama pull gemma4:e2b` の上、実行時に `--llm-model gemma4:e2b` を指定してください。

macOS と違い、PyAudio は pip のビルド済みパッケージがそのまま入るため
`portaudio` の別途インストールは不要です。
デスクトップ通知に必要な `pywin32` と、スピーカー音声の録音
（WASAPI ループバック）に必要な `PyAudioWPatch` は、`requirements.txt` の
環境マーカーにより Windows でのみ自動インストールされます。

## 2. 実行

`run.bat` をダブルクリックすると起動します（初回セットアップが済んでいない場合は
venv 作成と依存インストールも自動で行います）。
`Ctrl+C` で終了した際に「バッチ ジョブを終了しますか (Y/N)?」と聞かれたら
`N` を選ぶと、最終要約を画面で確認してからウィンドウを閉じられます。

ターミナルから直接実行する場合:

```powershell
.\venv\Scripts\python memorandum.py
```

### 音源の選択 (--source)

| モード | 用途 | 録音対象 |
| --- | --- | --- |
| `--source system`（既定） | **Web 会議** | スピーカー / イヤホンに再生される音声（WASAPI ループバック） |
| `--source mic` | 対面の会議 | マイク入力 |

既定の `system` モードは、Zoom / Teams / Meet などで**相手の発言**が
再生される音声をそのまま録音します。イヤホン使用時も問題なく録音できます。
ただし自分の声はスピーカーに再生されないため記録されません。
自分の発言も記録したい場合は会議アプリ側で「自分の音声のモニタリング」を
有効にするか、対面用途では `--source mic` を使ってください。

その他のオプションは `--help` で確認できます（使用モデル・録音間隔・保存先などを変更可能）。

```powershell
.\venv\Scripts\python memorandum.py --help
```

`Ctrl+C` で終了すると、最終要約付きの議事録
`meeting_log_YYYYMMDD_HHMMSS.txt` が保存されます。

## 3. 実行時の確認事項

- **Ollama の起動:** Ollama アプリが起動していない場合、スクリプトは
  起動時チェックでエラーメッセージを表示して終了します。
  スタートメニューから Ollama を起動するか `ollama serve` を実行してください。
- **マイクの許可:** 「設定 > プライバシーとセキュリティ > マイク」で
  デスクトップアプリのマイクアクセスが許可されていることを確認してください。
- **Whisper モデルのロード:** 初回実行時のみモデルのダウンロードが走るため
  数分かかることがあります（`%USERPROFILE%\.cache\whisper` に保存されます）。
- **通知:** Windows の「応答不可（集中モード）」が有効だと
  デスクトップ通知が表示されません。
