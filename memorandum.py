#!/usr/bin/env python3
"""memorandum: 完全ローカルで動作する会議リアルタイム要約アシスタント。

Web 会議の再生音声 (スピーカー出力のループバック) またはマイク音声を
一定間隔で録音し、Whisper によるローカル文字起こしと
Ollama (ローカル LLM) による要約をパイプラインで実行する。
要約はターミナル表示とデスクトップ通知の両方で確認でき、
終了時 (Ctrl+C) には会議全体の最終要約と全原文をファイルへ保存する。

音声データ・テキストデータは一切外部サーバーへ送信されない。
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Protocol

import numpy as np
import ollama
import speech_recognition as sr
import whisper

logger = logging.getLogger("memorandum")

# Whisper が無音時に出力しがちな定型ハルシネーション。
# 網羅リストではなく、実際の運用で観測したものを追記していくブロックリスト
KNOWN_HALLUCINATIONS = (
    "ご視聴ありがとうございました",
    "チャンネル登録",
)
# これ未満の文字数は無音時の断片ノイズとみなして破棄する。
# トレードオフ: 「賛成」のような短い有効発言も落ちるが、60 秒チャンクの
# 発言全体がこの長さに満たないケースは稀であり、ノイズ除去を優先した
MIN_TRANSCRIPT_CHARS = 5

# 未処理チャンクのキュー上限。60 秒 × 16kHz × float32 ≈ 3.8MB/チャンクのため
# 30 件で約 115MB・遅延 30 分ぶんに相当する。推論が録音に追いつかない環境で
# メモリが際限なく膨らむのを防ぎ、超過時は最も古い未処理チャンクを破棄する
MAX_PENDING_CHUNKS = 30

# Ollama は num_ctx を指定しないと、VRAM 24GiB 未満の環境では 4K トークンで
# モデルをロードし、超過したプロンプトを先頭から黙って切り捨てる (モデル自体が
# 128K 対応でも同じ)。先頭には指示文があるため、長い会議の最終要約は指示ごと
# 前半を失い、終盤だけを見た出力になる。そこで num_ctx を明示したうえで、
# 1 回に渡す原文を SUMMARY_BLOCK_CHARS 以下に分割して段階的に要約する
LLM_NUM_CTX = 8192
# 1 回の要約に渡す原文の上限 (文字数)。日本語 1 文字が 2 トークンになる
# 最悪の見積もりでも、指示文と出力を足して LLM_NUM_CTX に収まる大きさ
SUMMARY_BLOCK_CHARS = 3000
# 段階要約を繰り返す上限。LLM が文字数の指示を守らず縮まらない場合の打ち切り
MAX_SUMMARY_ROUNDS = 3

# Ctrl+C で終了したとき、録音途中のチャンクがこの秒数以上あれば議事録に含める
MIN_PARTIAL_SECONDS = 1.0

# 逐次保存中 (または異常終了した) 議事録の最終要約欄に入る文言
UNFINISHED_SUMMARY = "(最終要約なし: 会議の記録中、または正常に終了しなかった記録です)"


@dataclass(frozen=True)
class Config:
    """実行時設定。CLI 引数から生成する。"""

    llm_model: str = "gemma4:e4b"
    whisper_model: str = "small"
    language: str = "ja"
    # "system" (スピーカー出力) or "mic" (マイク)。
    # ループバック録音は WASAPI 依存のため Windows 以外はマイクを既定とする
    source: str = "system" if sys.platform == "win32" else "mic"
    chunk_seconds: int = 60
    sample_rate: int = 16000
    # 無音判定の音圧しきい値。適正値は入力デバイスのゲインに依存するため、
    # 有効な発言がスキップされる場合はログに出る RMS 実測値を確認して
    # --rms-threshold で調整する
    rms_threshold: float = 0.005
    output_dir: Path = Path(".")
    notify: bool = True


@dataclass(frozen=True)
class AudioChunk:
    """1 サイクル分の録音データ (float32, -1.0〜1.0)。"""

    recorded_at: datetime
    samples: np.ndarray


@dataclass(frozen=True)
class TranscriptChunk:
    """1 サイクル分の文字起こし結果。"""

    recorded_at: datetime
    text: str


def rms(samples: np.ndarray) -> float:
    """音圧 (Root Mean Square) を返す。無音チャンクの足切りに使う。"""
    return float(np.sqrt(np.mean(np.square(samples))))


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """線形補間による簡易リサンプリング (追加依存なしを優先した選択)。

    アンチエイリアシングフィルタを掛けないため、ダウンサンプリング時は
    ナイキスト周波数を超える成分が折り返し雑音になり得る。認識精度への
    影響は未計測。劣化が観測されたら scipy.signal.resample_poly 等への
    置き換えを検討する。
    """
    if src_rate == dst_rate:
        return samples
    n_dst = int(len(samples) * dst_rate / src_rate)
    positions = np.linspace(0, len(samples) - 1, n_dst)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)


def record_chunks(
    read_block: Callable[[], bytes],
    blocks_per_chunk: int,
    min_partial_blocks: int,
    to_waveform: Callable[[bytes], np.ndarray],
    chunk_seconds: int,
) -> Iterator[AudioChunk]:
    """音声ブロックを読み続け、固定長の AudioChunk を無限に yield する。

    Ctrl+C (KeyboardInterrupt) で録音を止めたときは、録音途中のチャンクを
    捨てずに yield してから KeyboardInterrupt を送出し直す。会議の締めくくり
    (決定事項の確認など) は終了操作の直前に話されることが多く、最大
    chunk_seconds 秒ぶんのそれを黙って失わないため。
    """
    while True:
        logger.info("録音中 (%d 秒)...", chunk_seconds)
        recorded_at = datetime.now()
        blocks: list[bytes] = []
        try:
            while len(blocks) < blocks_per_chunk:
                blocks.append(read_block())
        except KeyboardInterrupt:
            if len(blocks) >= min_partial_blocks:
                logger.info("録音途中の音声も議事録に含めます")
                yield AudioChunk(recorded_at, to_waveform(b"".join(blocks)))
            raise
        yield AudioChunk(recorded_at, to_waveform(b"".join(blocks)))


class AudioSource(Protocol):
    """音声チャンクの供給元 (マイク / スピーカーループバック) の共通インターフェース。"""

    def chunks(self) -> Iterator[AudioChunk]: ...


class SystemAudioCapture:
    """スピーカーへ再生中の音声 (システム音声) を録音する。

    Windows の WASAPI ループバックを利用し、Web 会議など
    相手の声がスピーカー側に流れるケースの議事録に使う。
    """

    _FRAMES_PER_BUFFER = 1024
    _MAX_CONSECUTIVE_ERRORS = 5
    _RETRY_WAIT_SECONDS = 2.0

    def __init__(self, config: Config) -> None:
        if sys.platform != "win32":
            raise SystemExit(
                "--source system は Windows (WASAPI ループバック) のみ対応です。"
                "macOS では --source mic と BlackHole 等の仮想デバイスを併用してください。"
            )
        self._config = config

    def chunks(self) -> Iterator[AudioChunk]:
        """録音チャンクを無限に yield する。

        モニタのスリープや再生デバイスの切替でループバックストリームが
        失効した場合は、最新の既定デバイスを取り直して自動再接続する。
        """
        consecutive_errors = 0
        while True:
            try:
                for chunk in self._record_session():
                    consecutive_errors = 0
                    yield chunk
            except OSError as exc:
                consecutive_errors += 1
                if consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                    raise SystemExit(
                        f"録音デバイスのエラーが解消しないため終了します: {exc}\n"
                        "既定の再生デバイスが有効か確認してください。"
                    ) from exc
                logger.warning(
                    "録音デバイスでエラーが発生しました (%s)。%.0f 秒後に再接続します...",
                    exc,
                    self._RETRY_WAIT_SECONDS,
                )
                time.sleep(self._RETRY_WAIT_SECONDS)

    def _record_session(self) -> Iterator[AudioChunk]:
        """ループバックストリームを開き、失効するまでチャンクを yield し続ける。"""
        import pyaudiowpatch as pyaudio  # Windows 専用のため遅延 import

        # デバイス一覧は PyAudio インスタンスに紐づくため、
        # 再接続のたびに作り直して最新の既定デバイスを取得する
        audio = pyaudio.PyAudio()
        stream = None
        try:
            device = self._default_loopback(audio)
            rate = int(device["defaultSampleRate"])
            channels = int(device["maxInputChannels"])
            logger.info(
                "ループバック録音: %s (%d Hz, %d ch)", device["name"], rate, channels
            )
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=int(device["index"]),
                frames_per_buffer=self._FRAMES_PER_BUFFER,
            )
            yield from record_chunks(
                read_block=lambda: stream.read(
                    self._FRAMES_PER_BUFFER, exception_on_overflow=False
                ),
                blocks_per_chunk=max(
                    1,
                    round(rate * self._config.chunk_seconds / self._FRAMES_PER_BUFFER),
                ),
                min_partial_blocks=math.ceil(
                    rate * MIN_PARTIAL_SECONDS / self._FRAMES_PER_BUFFER
                ),
                to_waveform=lambda data: self._to_whisper_waveform(
                    data, rate, channels
                ),
                chunk_seconds=self._config.chunk_seconds,
            )
        finally:
            # ストリームが既に失効していると close も OSError を投げ、
            # 元の例外を覆い隠してしまうため握りつぶす
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
            audio.terminate()

    @staticmethod
    def _default_loopback(audio) -> dict:
        """既定の再生デバイスに対応するループバックデバイスを返す。

        見つからない場合は OSError を送出し、呼び出し側の再接続リトライに委ねる。
        """
        try:
            return audio.get_default_wasapi_loopback()
        except (OSError, LookupError) as exc:
            raise OSError("ループバック録音デバイスが見つかりません") from exc

    def _to_whisper_waveform(
        self, data: bytes, rate: int, channels: int
    ) -> np.ndarray:
        """int16 の生データを Whisper が扱えるモノラル float32 波形へ変換する。"""
        samples = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return resample(samples, rate, self._config.sample_rate)


class MicrophoneCapture:
    """マイクから固定長の音声チャンクを読み続ける (対面会議向け)。

    SpeechRecognition の listen() は発話の開始を待ってから録音し、終わってから
    返すため、Ctrl+C で止めると録音途中の音声が失われ、記録時刻も発話の
    終わりになる。ここではストリームを直接固定長で読み、無音の判定は
    後段の RMS フィルタに任せる。
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def chunks(self) -> Iterator[AudioChunk]:
        """録音した音声チャンクを無限に yield する。"""
        with sr.Microphone(sample_rate=self._config.sample_rate) as source:
            rate, block = source.SAMPLE_RATE, source.CHUNK
            seconds = self._config.chunk_seconds
            yield from record_chunks(
                read_block=lambda: source.stream.read(block),
                blocks_per_chunk=max(1, round(rate * seconds / block)),
                min_partial_blocks=math.ceil(rate * MIN_PARTIAL_SECONDS / block),
                to_waveform=self._to_float_array,
                chunk_seconds=seconds,
            )

    @staticmethod
    def _to_float_array(data: bytes) -> np.ndarray:
        """int16 の生データを Whisper が直接扱える float32 波形へ変換する。"""
        return np.frombuffer(data, np.int16).astype(np.float32) / 32768.0


class Transcriber:
    """Whisper によるローカル文字起こし。"""

    def __init__(self, model_name: str, language: str) -> None:
        logger.info("Whisper モデル (%s) をロード中...", model_name)
        self._model = whisper.load_model(model_name)
        self._language = language

    def transcribe(self, samples: np.ndarray) -> str:
        """音声をテキスト化する。無音・ハルシネーションと判定したら空文字を返す。"""
        result = self._model.transcribe(
            samples,
            language=self._language,
            fp16=False,
            no_speech_threshold=0.6,  # 無音セグメントを破棄 (Whisper の既定値を明示)
            logprob_threshold=-1.0,  # 低確信度の出力を破棄 (Whisper の既定値を明示)
        )
        return self._clean(result["segments"])

    @staticmethod
    def _is_hallucinated_segment(text: str) -> bool:
        """無音時に Whisper が出力しがちな定型文を含むセグメントか。"""
        return any(phrase in text for phrase in KNOWN_HALLUCINATIONS)

    @classmethod
    def _clean(cls, segments: list[dict]) -> str:
        """定型文を含むセグメントだけを除いて連結し、短すぎる結果は空文字にする。

        判定をチャンク全体の文字列に掛けると、60 秒ぶんの正常な発言のどこかに
        定型文が一度現れただけで、その 1 分の原文が丸ごと消える。Whisper の
        定型ハルシネーションはそれ単体で 1 セグメントになるため、セグメント
        単位で落とせば、巻き込む範囲をそのセグメントだけに抑えられる。
        """
        text = "".join(
            str(seg["text"])
            for seg in segments
            if not cls._is_hallucinated_segment(str(seg["text"]))
        ).strip()
        return "" if len(text) < MIN_TRANSCRIPT_CHARS else text


def format_line(chunk: TranscriptChunk) -> str:
    """議事録の原文 1 行の表記。"""
    return f"[{chunk.recorded_at:%H:%M:%S}] {chunk.text}"


@dataclass(frozen=True)
class Excerpt:
    """最終要約の入力単位。原文の 1 行か、複数行をまとめた時間帯メモ。"""

    start: datetime
    end: datetime
    text: str
    is_note: bool = False

    def render(self) -> str:
        if self.is_note:
            return f"【{self.start:%H:%M}〜{self.end:%H:%M}】\n{self.text}"
        return f"[{self.start:%H:%M:%S}] {self.text}"

    @property
    def size(self) -> int:
        return len(self.render()) + 1  # 区切りの改行ぶん


def to_excerpts(chunks: list[TranscriptChunk], budget: int) -> list[Excerpt]:
    """原文を Excerpt へ変換する。1 件で予算を超えるものは文字数で刻む。"""
    # 刻んだ 1 片に時刻の表記 "[HH:MM:SS] " と改行が付いても予算に収まる長さ
    piece = budget - len("[00:00:00] ") - 1
    return [
        Excerpt(c.recorded_at, c.recorded_at, c.text[i : i + piece])
        for c in chunks
        for i in range(0, max(len(c.text), 1), piece)
    ]


def pack(excerpts: list[Excerpt], budget: int) -> list[list[Excerpt]]:
    """時系列の順序を保ったまま、合計が budget 以下になるよう前から詰めて分ける。"""
    groups: list[list[Excerpt]] = []
    current: list[Excerpt] = []
    size = 0
    for excerpt in excerpts:
        if current and size + excerpt.size > budget:
            groups.append(current)
            current, size = [], 0
        current.append(excerpt)
        size += excerpt.size
    if current:
        groups.append(current)
    return groups


def render_excerpts(excerpts: list[Excerpt]) -> str:
    return "\n".join(e.render() for e in excerpts)


# 時間帯メモの見出し "【HH:MM〜HH:MM】" と改行・区切りのぶんの長さ
NOTE_OVERHEAD = len("【00:00〜00:00】\n") + 1


class Summarizer:
    """Ollama (ローカル LLM) による要約生成。"""

    CHUNK_PROMPT = (
        "以下は会議での直近の発言です。記号や見出しを使わず、"
        "日本語30文字以内の一文で要約してください。\n\n{text}"
    )
    FINAL_PROMPT = (
        "あなたはプロの書記です。以下の会議の全発言記録から、"
        "重要な論点・決定事項・ネクストアクションを整理した"
        "議事録要約を日本語で作成してください。\n\n{text}"
    )
    # 原文が 1 回に渡せる量を超えるときの、時間帯ごとの抽出
    SECTION_PROMPT = (
        "以下は会議の一部（{span}）の記録です。この時間帯の重要な論点・"
        "決定事項・ネクストアクションを、400字以内の箇条書きで抽出してください。"
        "\n\n{text}"
    )
    FINAL_FROM_NOTES_PROMPT = (
        "あなたはプロの書記です。以下は会議を時間帯ごとに区切って抽出したメモです。"
        "会議全体を通した重要な論点・決定事項・ネクストアクションを整理した"
        "議事録要約を日本語で作成してください。\n\n{text}"
    )

    def __init__(self, model: str) -> None:
        self._model = model

    def summarize_chunk(self, text: str) -> str:
        summary = self._chat(self.CHUNK_PROMPT.format(text=text))
        # 通知バナーに Markdown 記号が混ざらないよう除去する
        return re.sub(r"[#*`]", "", summary).strip()

    def summarize_meeting(self, chunks: list[TranscriptChunk]) -> str:
        """会議全体の要約を作る。

        全原文が SUMMARY_BLOCK_CHARS に収まれば 1 回で要約する。収まらなければ
        時間帯ごとに区切って箇条書きのメモへ縮め、メモが収まるまで繰り返してから
        全体を要約する。どの段階でも 1 回に渡す量は SUMMARY_BLOCK_CHARS 以下なので、
        会議の長さによらずコンテキスト長を超えて前半が切り捨てられることはない。
        """
        excerpts = to_excerpts(chunks, SUMMARY_BLOCK_CHARS)
        rounds = 0
        while sum(e.size for e in excerpts) > SUMMARY_BLOCK_CHARS:
            if rounds == MAX_SUMMARY_ROUNDS:
                excerpts = self._trim_to_budget(excerpts)
                break
            rounds += 1
            groups = pack(excerpts, SUMMARY_BLOCK_CHARS)
            excerpts = [
                self._summarize_section(group, i, len(groups))
                for i, group in enumerate(groups, 1)
            ]
        prompt = self.FINAL_FROM_NOTES_PROMPT if rounds else self.FINAL_PROMPT
        return self._chat(prompt.format(text=render_excerpts(excerpts)))

    def _summarize_section(
        self, group: list[Excerpt], index: int, total: int
    ) -> Excerpt:
        start, end = group[0].start, group[-1].end
        span = f"{start:%H:%M}〜{end:%H:%M}"
        logger.info("最終要約: %s の区間を要約中 (%d/%d)...", span, index, total)
        note = self._chat(
            self.SECTION_PROMPT.format(span=span, text=render_excerpts(group))
        )
        # LLM が文字数の指示を大きく外れて返すと、そのメモ 1 つで次の段の
        # 入力が上限を超えてしまうため、ここで上限内に収める
        limit = SUMMARY_BLOCK_CHARS - NOTE_OVERHEAD
        if len(note) > limit:
            logger.warning("%s の区間メモが長すぎるため切り詰めます", span)
            note = note[:limit]
        return Excerpt(start, end, note, is_note=True)

    @staticmethod
    def _trim_to_budget(excerpts: list[Excerpt]) -> list[Excerpt]:
        """メモが縮まらないまま上限回数に達したとき、各メモを均等に切り詰める。"""
        logger.warning(
            "区間メモが要約の上限に収まらないため、各メモの末尾を切り詰めます"
        )
        limit = max(1, SUMMARY_BLOCK_CHARS // len(excerpts) - NOTE_OVERHEAD)
        return [
            Excerpt(e.start, e.end, e.text[:limit], e.is_note) for e in excerpts
        ]

    def _chat(self, prompt: str) -> str:
        response = ollama.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": LLM_NUM_CTX},
        )
        return response["message"]["content"].strip()


class Notifier:
    """デスクトップ通知。通知に失敗しても本体の処理は止めない。

    Windows はトースト API を使う winotify、macOS は OS 標準の osascript で送る。
    (plyer の macOS 実装は pyobjus を必要とし、入っていないと毎回失敗する)
    """

    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled and sys.platform in ("win32", "darwin")
        if enabled and not self._enabled:
            logger.info("この OS ではデスクトップ通知に対応していません")
        self._failed = False

    def send(self, title: str, message: str) -> None:
        if not self._enabled:
            return
        try:
            if sys.platform == "win32":
                self._send_windows_toast(title, message)
            else:
                self._send_macos(title, message)
        except Exception:
            # 失敗するたびにトレースバックを出すと、1 分ごとにターミナルの
            # 原文と要約の表示が埋もれるため、詳細を出すのは初回だけにする
            if not self._failed:
                self._failed = True
                logger.warning(
                    "デスクトップ通知に失敗しました (以降の失敗は表示しません)",
                    exc_info=True,
                )

    @staticmethod
    def _send_windows_toast(title: str, message: str) -> None:
        from winotify import Notification  # Windows 専用のため遅延 import

        Notification(
            app_id="memorandum",
            title=title,
            msg=message,
            duration="short",
        ).show()

    @staticmethod
    def _send_macos(title: str, message: str) -> None:
        # 本文を AppleScript の文字列リテラルに埋め込まず引数で渡すので、
        # 要約に引用符やバックスラッシュが含まれても壊れない
        script = "display notification (item 2 of argv) with title (item 1 of argv)"
        subprocess.run(
            ["osascript", "-e", "on run argv", "-e", script, "-e", "end run"]
            + [title, message],
            check=True,
            capture_output=True,
            timeout=10,
        )


def report_path(output_dir: Path, started_at: datetime) -> Path:
    """議事録ファイルのパス。会議の開始時刻で名付ける。"""
    return output_dir / f"meeting_log_{started_at:%Y%m%d_%H%M%S}.txt"


def render_report(
    chunks: list[TranscriptChunk],
    final_summary: str,
    dropped_count: int = 0,
    aborted_count: int = 0,
) -> str:
    """最終要約とタイムスタンプ付き全原文からなる議事録の本文。

    未処理のまま破棄したチャンクがある場合は、欠落の事実を冒頭に明記する。
    """
    lines = ["=== 会議最終要約 ===", final_summary, ""]
    if dropped_count:
        lines += [
            f"⚠ 処理能力不足のため未処理のまま破棄したチャンク: {dropped_count} 件",
            "  (該当時間帯の発言は記録されていません)",
            "",
        ]
    if aborted_count:
        lines += [
            f"⚠ 終了処理の打ち切りにより未処理のまま破棄したチャンク: {aborted_count} 件",
            "  (該当時間帯の発言は記録されていません)",
            "",
        ]
    lines += ["=== 全原文データ ==="]
    lines += [format_line(c) for c in chunks]
    return "\n".join(lines) + "\n"


def save_report(
    chunks: list[TranscriptChunk],
    final_summary: str,
    path: Path,
    dropped_count: int = 0,
    aborted_count: int = 0,
) -> Path:
    """議事録を書き出す。

    一時ファイルに書いてから置き換えるので、書き込み中に中断されても
    逐次保存済みのファイルが壊れた状態で残ることはない。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(render_report(chunks, final_summary, dropped_count, aborted_count))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


class TranscriptLog:
    """文字起こし結果を 1 件ずつ議事録ファイルへ追記する。

    終了時にまとめて書くだけだと、ウィンドウを閉じる・クラッシュする・最終要約の
    生成中に電源が落ちるといった Ctrl+C 以外の終わり方で、会議の原文が全損する。
    原文を逐次ディスクへ残しておき、正常終了時に最終要約付きの完全版で置き換える。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._started = False

    def append(self, chunk: TranscriptChunk) -> None:
        if not self._started:
            save_report([], UNFINISHED_SUMMARY, self.path)
            self._started = True
        with self.path.open("a", encoding="utf-8") as f:
            f.write(format_line(chunk) + "\n")
            f.flush()
            os.fsync(f.fileno())


class MeetingAssistant:
    """録音と AI 処理を並行実行するオーケストレーター。

    録音 (メインスレッド) と文字起こし・要約 (ワーカースレッド) を
    キューで分離し、AI 処理に時間がかかっても録音が途切れないようにする。
    キューは MAX_PENDING_CHUNKS で上限を設け、推論が録音に追いつかない
    環境でもメモリが際限なく増えないようにする (超過時は最古チャンクを破棄)。
    """

    def __init__(
        self,
        config: Config,
        capture: AudioSource,
        transcriber: Transcriber,
        summarizer: Summarizer,
        notifier: Notifier,
    ) -> None:
        self._config = config
        self._capture = capture
        self._transcriber = transcriber
        self._summarizer = summarizer
        self._notifier = notifier
        self._audio_queue: queue.Queue[AudioChunk | None] = queue.Queue(
            maxsize=MAX_PENDING_CHUNKS
        )
        self._chunks: list[TranscriptChunk] = []
        self._log = TranscriptLog(report_path(config.output_dir, datetime.now()))
        self._abort = threading.Event()
        self._dropped_count = 0  # キュー溢れで捨てた件数
        self._aborted_count = 0  # 2 度目の Ctrl+C で捨てた件数
        # _aborted_count は打ち切り時に録音側とワーカー側の両方から数える
        self._count_lock = threading.Lock()

    def run(self) -> None:
        worker = threading.Thread(
            target=self._process_loop, name="processor", daemon=True
        )
        worker.start()
        aborted = False
        try:
            self._capture_loop()
        except KeyboardInterrupt:
            print("\n⏹ 会議を終了します。未処理の音声を処理中...")
            print("   (もう一度 Ctrl+C で残りを打ち切り、最終要約を省略して保存します)")
        finally:
            # 終了処理のどこで再度 Ctrl+C されても、文字起こし済みの内容は
            # 必ず _finalize で保存する (議事録の全損を防ぐ)
            try:
                self._drain_and_join(worker)
            except KeyboardInterrupt:
                aborted = True
                self._abort_and_join(worker)
            finally:
                self._finalize(skip_summary=aborted)

    def _capture_loop(self) -> None:
        for chunk in self._capture.chunks():
            level = rms(chunk.samples)
            if level < self._config.rms_threshold:
                logger.info(
                    "無音のためスキップ (RMS=%.4f < しきい値 %.4f)",
                    level,
                    self._config.rms_threshold,
                )
                continue
            logger.info("音声チャンクをキューへ投入 (RMS=%.4f)", level)
            self._enqueue(chunk)

    def _enqueue(self, chunk: AudioChunk) -> None:
        """キューへ追加する。満杯なら最も古い未処理チャンクを警告付きで破棄する。

        録音 (メイン) スレッドをブロックしないことを最優先とし、議事録の
        欠落はログで明示する。破棄は最古側: 会議の結論が出やすい直近の
        発言を優先して残すための選択。
        """
        try:
            self._audio_queue.put_nowait(chunk)
            return
        except queue.Full:
            pass
        try:
            dropped = self._audio_queue.get_nowait()
        except queue.Empty:  # ワーカーが直前に消化した場合
            dropped = None
        if isinstance(dropped, AudioChunk):
            self._dropped_count += 1
            logger.warning(
                "推論が録音に追いついていません。最古の未処理チャンク"
                " [%s] を破棄しました (累計 %d 件)。--whisper-model base"
                " など軽いモデルの利用を検討してください。",
                f"{dropped.recorded_at:%H:%M:%S}",
                self._dropped_count,
            )
        self._audio_queue.put(chunk)

    def _process_loop(self) -> None:
        while True:
            chunk = self._audio_queue.get()
            if chunk is None:
                return
            if self._abort.is_set():
                # 打ち切りの直前にキューから取っていた 1 件。録音側の破棄では
                # 数えられないため、ここで数える
                with self._count_lock:
                    self._aborted_count += 1
                return
            try:
                self._process_chunk(chunk)
            except Exception:
                logger.exception("音声チャンクの処理に失敗しました")

    def _drain_and_join(self, worker: threading.Thread) -> None:
        """未処理キューを処理し切るまで待ち、残チャンク数を定期表示する。"""
        self._audio_queue.put(None)  # ワーカーへの終了シグナル
        while worker.is_alive():
            worker.join(timeout=5.0)
            pending = self._audio_queue.qsize()
            if worker.is_alive() and pending:
                print(
                    f"   残り約 {pending} チャンクを処理中..."
                    " (もう一度 Ctrl+C で打ち切って保存します)"
                )

    def _abort_and_join(self, worker: threading.Thread) -> None:
        """2 度目の Ctrl+C: 未処理チャンクを破棄し、処理中の 1 件だけ完了を待つ。

        破棄した件数は議事録に明記する (黙って欠落させない)。
        """
        print(
            "\n⏹ 残りの処理を打ち切ります。"
            "最終要約は省略し、文字起こし済みの内容で議事録を保存します。"
        )
        self._abort.set()
        try:
            while True:
                if isinstance(self._audio_queue.get_nowait(), AudioChunk):
                    with self._count_lock:
                        self._aborted_count += 1
        except queue.Empty:
            pass
        self._audio_queue.put(None)
        # Whisper / LLM の推論は途中で安全に中断できないため、
        # 現在処理中のチャンクの完了だけは待つ (最大でも 1 チャンクぶん)
        if worker.is_alive():
            print("   処理中のチャンクの完了を待っています...")
        worker.join()

    def _process_chunk(self, chunk: AudioChunk) -> None:
        text = self._transcriber.transcribe(chunk.samples)
        if not text:
            return
        print(f"\n--- 原文 [{chunk.recorded_at:%H:%M:%S}] ---\n{text}")
        transcript = TranscriptChunk(chunk.recorded_at, text)
        self._chunks.append(transcript)
        try:
            self._log.append(transcript)
        except OSError as exc:
            # 逐次保存に失敗しても原文はメモリにあり、終了時の保存は引き続き試みる
            logger.error("原文の逐次保存に失敗しました: %s", exc)

        summary = self._summarizer.summarize_chunk(text)
        print(f"💡 要約: {summary}")
        self._notifier.send("✨ 直近の要約", summary)

    def _finalize(self, skip_summary: bool = False) -> None:
        # 3 度目の Ctrl+C でワーカーが止まりきらないまま来ることがあるため、
        # この時点の原文で確定させる
        chunks = list(self._chunks)
        discarded = self._dropped_count + self._aborted_count
        if not chunks:
            print("文字起こし済みの発言がないため保存をスキップしました。")
            if discarded:
                print(f"⚠ 未処理のまま破棄したチャンクが {discarded} 件あります。")
            return
        if skip_summary:
            final_summary = "(終了処理を打ち切ったため、最終要約は省略しました)"
        else:
            print("\n📝 最終要約を生成中... (Ctrl+C で省略して原文だけ保存します)")
            try:
                final_summary = self._summarizer.summarize_meeting(chunks)
            except KeyboardInterrupt:
                # 最終要約の生成中に Ctrl+C されても原文の議事録は必ず残す
                final_summary = "(最終要約は Ctrl+C により中断されました)"
            except Exception:
                logger.exception("最終要約の生成に失敗しました。原文のみ保存します。")
                final_summary = "(最終要約の生成に失敗しました)"

        print("\n" + "=" * 30)
        print("📝 【最終要約】")
        print(final_summary)
        print("=" * 30)

        if discarded:
            print(
                f"⚠ 未処理のまま {discarded} チャンクを破棄しました。"
                "議事録に欠落があります。"
            )
        path = save_report(
            chunks,
            final_summary,
            self._log.path,
            dropped_count=self._dropped_count,
            aborted_count=self._aborted_count,
        )
        print(f"📄 議事録を保存しました: {path}")
        self._notifier.send("✅ 保存完了", f"議事録を保存しました: {path.name}")


def ensure_ollama_ready(model: str) -> None:
    """起動時に Ollama サーバーへの接続とモデルの存在を確認する。"""
    try:
        ollama.show(model)
    except ollama.ResponseError as exc:
        raise SystemExit(
            f"Ollama にモデル '{model}' が見つかりません。"
            f"`ollama pull {model}` を実行してください。({exc.error})"
        ) from exc
    except Exception as exc:
        raise SystemExit(
            "Ollama サーバーに接続できません。"
            "Ollama を起動してから再実行してください。"
        ) from exc


def ensure_output_dir_writable(output_dir: Path) -> None:
    """議事録の保存先に書き込めることを起動時に確認する。

    会議が終わってから保存に失敗すると取り返しがつかないため、先に確かめる。
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=output_dir):
            pass
    except OSError as exc:
        raise SystemExit(
            f"議事録の保存先 '{output_dir}' に書き込めません: {exc}"
        ) from exc


def parse_args(argv: list[str] | None = None) -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--llm-model",
        default=defaults.llm_model,
        help=f"要約に使う Ollama モデル (default: {defaults.llm_model})",
    )
    parser.add_argument(
        "--whisper-model",
        default=defaults.whisper_model,
        help=f"Whisper モデルサイズ (default: {defaults.whisper_model})",
    )
    parser.add_argument(
        "--language",
        default=defaults.language,
        help=f"文字起こしの言語コード (default: {defaults.language})",
    )
    parser.add_argument(
        "--source",
        choices=("system", "mic"),
        default=defaults.source,
        help=(
            "録音する音源。system: スピーカー出力のループバック (Web会議向け) / "
            f"mic: マイク (対面向け) (default: {defaults.source})"
        ),
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=defaults.chunk_seconds,
        help=f"1 サイクルの録音秒数 (default: {defaults.chunk_seconds})",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=defaults.rms_threshold,
        help=f"無音とみなす音圧しきい値 (default: {defaults.rms_threshold})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults.output_dir,
        help="議事録の保存先ディレクトリ (default: カレントディレクトリ)",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="デスクトップ通知を無効にする",
    )
    args = parser.parse_args(argv)
    return Config(
        llm_model=args.llm_model,
        whisper_model=args.whisper_model,
        language=args.language,
        source=args.source,
        chunk_seconds=args.chunk_seconds,
        rms_threshold=args.rms_threshold,
        output_dir=args.output_dir,
        notify=not args.no_notify,
    )


def configure_output_encoding() -> None:
    """Windows でリダイレクト時に stdout が cp932 になり、絵文字や日本語の
    出力が UnicodeEncodeError で落ちるのを防ぐ。"""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> None:
    configure_output_encoding()
    config = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ensure_ollama_ready(config.llm_model)
    ensure_output_dir_writable(config.output_dir)
    capture: AudioSource = (
        SystemAudioCapture(config)
        if config.source == "system"
        else MicrophoneCapture(config)
    )
    assistant = MeetingAssistant(
        config,
        capture=capture,
        transcriber=Transcriber(config.whisper_model, config.language),
        summarizer=Summarizer(config.llm_model),
        notifier=Notifier(config.notify),
    )
    source_label = (
        "スピーカー出力 (Web会議)" if config.source == "system" else "マイク"
    )
    print("🚀 完全ローカル・会議アシスタント稼働中")
    print(f"   入力: {source_label}")
    print(f"   STT: Whisper ({config.whisper_model}) / LLM: {config.llm_model}")
    print("   【Ctrl+C】で終了し、最終レポートを作成します。")
    try:
        assistant.run()
    except KeyboardInterrupt:
        # 終了処理中の 3 度目以降の Ctrl+C。原文は逐次保存済みなので
        # トレースバックを出さずに終える
        sys.exit(130)


if __name__ == "__main__":
    main()
