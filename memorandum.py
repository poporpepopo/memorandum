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
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np
import ollama
import speech_recognition as sr
import whisper

logger = logging.getLogger("memorandum")

# Whisper が無音時に出力しがちな定型ハルシネーション。
# 網羅リストではなく、実際の運用で観測したものを追記していくブロックリスト
KNOWN_HALLUCINATIONS = (
    "スタッフの方が",
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
            reads_per_chunk = max(
                1,
                round(rate * self._config.chunk_seconds / self._FRAMES_PER_BUFFER),
            )
            while True:
                logger.info("録音中 (%d 秒)...", self._config.chunk_seconds)
                recorded_at = datetime.now()
                frames = [
                    stream.read(self._FRAMES_PER_BUFFER, exception_on_overflow=False)
                    for _ in range(reads_per_chunk)
                ]
                yield AudioChunk(
                    recorded_at,
                    self._to_whisper_waveform(b"".join(frames), rate, channels),
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
    """マイクから固定長の音声チャンクを読み続ける (対面会議向け)。"""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._recognizer = sr.Recognizer()
        # 沈黙で録音を打ち切らず、chunk_seconds の固定長バッチとして扱う
        self._recognizer.pause_threshold = config.chunk_seconds

    def chunks(self) -> Iterator[AudioChunk]:
        """録音した音声チャンクを無限に yield する。"""
        with sr.Microphone(sample_rate=self._config.sample_rate) as source:
            logger.info("環境ノイズを計測中...")
            self._recognizer.adjust_for_ambient_noise(source, duration=1)
            while True:
                logger.info("録音中 (最大 %d 秒)...", self._config.chunk_seconds)
                audio = self._recognizer.listen(
                    source,
                    timeout=None,
                    phrase_time_limit=self._config.chunk_seconds,
                )
                yield AudioChunk(
                    recorded_at=datetime.now(),
                    samples=self._to_float_array(audio),
                )

    @staticmethod
    def _to_float_array(audio: sr.AudioData) -> np.ndarray:
        """Whisper が直接扱える float32 波形へ変換する。"""
        samples = np.frombuffer(audio.get_raw_data(), np.int16)
        return samples.astype(np.float32) / 32768.0


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
        text = str(result["text"]).strip()
        return "" if self._is_hallucination(text) else text

    @staticmethod
    def _is_hallucination(text: str) -> bool:
        """無音時に Whisper が出力しがちな定型文・短い断片を除外する。"""
        if len(text) < MIN_TRANSCRIPT_CHARS:
            return True
        return any(phrase in text for phrase in KNOWN_HALLUCINATIONS)


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

    def __init__(self, model: str) -> None:
        self._model = model

    def summarize_chunk(self, text: str) -> str:
        summary = self._chat(self.CHUNK_PROMPT.format(text=text))
        # 通知バナーに Markdown 記号が混ざらないよう除去する
        return re.sub(r"[#*`]", "", summary).strip()

    def summarize_meeting(self, text: str) -> str:
        return self._chat(self.FINAL_PROMPT.format(text=text))

    def _chat(self, prompt: str) -> str:
        response = ollama.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()


class Notifier:
    """デスクトップ通知。通知に失敗しても本体の処理は止めない。

    Windows では plyer が使う旧来のバルーン通知が表示されないことがあるため、
    モダンなトースト API を使う winotify を優先する。
    """

    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled

    def send(self, title: str, message: str) -> None:
        if not self._enabled:
            return
        try:
            if sys.platform == "win32":
                self._send_windows_toast(title, message)
            else:
                self._send_plyer(title, message)
        except Exception:
            logger.warning("デスクトップ通知に失敗しました", exc_info=True)

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
    def _send_plyer(title: str, message: str) -> None:
        from plyer import notification

        notification.notify(
            title=title,
            message=message,
            app_name="memorandum",
            timeout=6,
        )


def save_report(
    chunks: list[TranscriptChunk],
    final_summary: str,
    output_dir: Path,
    dropped_count: int = 0,
) -> Path:
    """最終要約とタイムスタンプ付き全原文をテキストファイルへ保存する。

    処理しきれず破棄したチャンクがある場合は、欠落の事実をファイル冒頭に明記する。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"meeting_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    lines = ["=== 会議最終要約 ===", final_summary, ""]
    if dropped_count:
        lines += [
            f"⚠ 処理能力不足のため未処理のまま破棄したチャンク: {dropped_count} 件",
            "  (該当時間帯の発言は記録されていません)",
            "",
        ]
    lines += ["=== 全原文データ ==="]
    lines += [f"[{c.recorded_at:%H:%M:%S}] {c.text}" for c in chunks]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
        self._abort = threading.Event()
        self._dropped_count = 0

    def run(self) -> None:
        worker = threading.Thread(
            target=self._process_loop, name="processor", daemon=True
        )
        worker.start()
        try:
            self._capture_loop()
        except KeyboardInterrupt:
            print("\n⏹ 会議を終了します。未処理の音声を処理中...")
            print("   (もう一度 Ctrl+C で残りの処理を打ち切り、保存へ進みます)")
        finally:
            # 終了処理のどこで再度 Ctrl+C されても、文字起こし済みの内容は
            # 必ず _finalize で保存する (議事録の全損を防ぐ)
            try:
                self._drain_and_join(worker)
            except KeyboardInterrupt:
                self._abort_and_join(worker)
            finally:
                self._finalize()

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
            if chunk is None or self._abort.is_set():
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
                    " (もう一度 Ctrl+C で打ち切って保存へ進みます)"
                )

    def _abort_and_join(self, worker: threading.Thread) -> None:
        """2 度目の Ctrl+C: 未処理チャンクを破棄し、処理中の 1 件だけ完了を待つ。"""
        print("\n⏹ 残りの処理を打ち切ります。文字起こし済みの内容で議事録を保存します。")
        self._abort.set()
        try:
            while True:
                self._audio_queue.get_nowait()
        except queue.Empty:
            pass
        self._audio_queue.put(None)
        # Whisper / LLM の推論は途中で安全に中断できないため、
        # 現在処理中のチャンクの完了だけは待つ (最大でも 1 チャンクぶん)
        worker.join()

    def _process_chunk(self, chunk: AudioChunk) -> None:
        text = self._transcriber.transcribe(chunk.samples)
        if not text:
            return
        print(f"\n--- 原文 [{chunk.recorded_at:%H:%M:%S}] ---\n{text}")
        self._chunks.append(TranscriptChunk(chunk.recorded_at, text))

        summary = self._summarizer.summarize_chunk(text)
        print(f"💡 要約: {summary}")
        self._notifier.send("✨ 直近の要約", summary)

    def _finalize(self) -> None:
        if not self._chunks:
            print("録音データがないため保存をスキップしました。")
            return
        full_text = "\n".join(c.text for c in self._chunks)
        try:
            final_summary = self._summarizer.summarize_meeting(full_text)
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

        if self._dropped_count:
            print(
                f"⚠ 処理能力不足により {self._dropped_count} チャンクを破棄しました。"
                "議事録に欠落があります。"
            )
        path = save_report(
            self._chunks,
            final_summary,
            self._config.output_dir,
            dropped_count=self._dropped_count,
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
    assistant.run()


if __name__ == "__main__":
    main()
