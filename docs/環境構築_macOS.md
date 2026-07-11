# 環境構築（macOS 版）

`memorandum.py` は macOS / Windows 共通のコードで動作します。ここでは macOS での手順をまとめます。
（Windows は `環境構築_Windows.md` を参照）

## 前提

- macOS (Apple Silicon 推奨)
- Python 3.9 以上
- [Ollama](https://ollama.com/) がインストール済みであること
- Homebrew（PyAudio のビルドに `portaudio` が必要）

## 1. セットアップ（初回のみ）

リポジトリのフォルダでターミナルを開き、以下を実行します。

```bash
# PyAudio のビルドに必要
brew install portaudio

# 仮想環境の作成と依存ライブラリのインストール
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# 要約用モデルのダウンロード (約9.6GB)
ollama pull gemma4:e4b
```

メモリ 8GB クラスのマシンでは軽量な `gemma4:e2b`（約7.2GB）を推奨します。
その場合は `ollama pull gemma4:e2b` の上、実行時に `--llm-model gemma4:e2b` を指定してください。

## 2. 実行

Finder で `run.command` をダブルクリックすると起動します
（初回セットアップが済んでいない場合は venv 作成と依存インストールも自動で行います）。

zip などで取得して実行権限が無い場合は、初回のみ以下を実行してください。

```bash
chmod +x run.command
```

ターミナルから直接実行する場合:

```bash
./venv/bin/python memorandum.py
```

`Ctrl+C` で終了すると、最終要約付きの議事録
`meeting_log_YYYYMMDD_HHMMSS.txt` が保存されます。
オプションは `--help` で確認できます。

### 音源について

macOS では既定の音源は **マイク**（`--source mic` 相当）です。
Web 会議の相手音声（スピーカー出力）を録音したい場合は、
[BlackHole](https://existential.audio/blackhole/) などの仮想オーディオデバイスで
再生音を入力デバイスへルーティングした上で、それを既定の入力に設定してください
（Windows 版は OS 機能のループバックにより `--source system` で直接録音できます）。

## 3. 実行時の確認事項

- **Ollama の起動:** Ollama アプリが起動していない場合、スクリプトは
  起動時チェックでエラーメッセージを表示して終了します。
- **マイクの許可:** 初回実行時にターミナルへの「マイクへのアクセス」許可を
  求められるので「OK」を押してください。
- **Whisper モデルのロード:** 初回実行時のみモデルのダウンロードが走るため
  数分かかることがあります（`~/.cache/whisper` に保存されます）。

## 4. 依存ライブラリ一覧と役割

| ライブラリ | 役割 |
| --- | --- |
| **openai-whisper** | 音声からテキストへの書き起こし（ローカル実行） |
| **ollama** | ローカル LLM（Gemma 4）による要約生成 |
| **SpeechRecognition** | マイクからの音声キャプチャと録音タイミングの制御 |
| **numpy** | 音声データの数値変換・音圧 (RMS) 計算 |
| **plyer** | (macOS) OS 標準のデスクトップ通知の表示 |
| **PyAudio** | マイク入力を Python で扱うためのブリッジ |
| **PyAudioWPatch** | (Windows のみ) スピーカー出力のループバック録音 |
| **winotify** | (Windows のみ) トースト通知の表示 |
