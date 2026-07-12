"""memorandum の純関数・キュー制御・保存処理のテスト。

音声デバイスや Whisper / Ollama の実体は使わない (conftest.py でスタブ)。
"""

import queue
from datetime import datetime

import numpy as np
import pytest

from memorandum import (
    MAX_PENDING_CHUNKS,
    AudioChunk,
    Config,
    MeetingAssistant,
    Transcriber,
    TranscriptChunk,
    rms,
    resample,
    save_report,
)


class TestRms:
    def test_silence_is_zero(self):
        assert rms(np.zeros(1000, dtype=np.float32)) == 0.0

    def test_full_scale_dc_is_one(self):
        assert rms(np.ones(1000, dtype=np.float32)) == pytest.approx(1.0)

    def test_sine_wave_is_amplitude_over_sqrt2(self):
        t = np.linspace(0, 1, 16000, endpoint=False)
        wave = 0.5 * np.sin(2 * np.pi * 440 * t)
        assert rms(wave) == pytest.approx(0.5 / np.sqrt(2), rel=1e-3)

    def test_threshold_boundary(self):
        """既定しきい値 0.005 の前後で無音判定が分かれること。"""
        config = Config()
        quiet = np.full(1000, 0.004, dtype=np.float32)
        loud = np.full(1000, 0.006, dtype=np.float32)
        assert rms(quiet) < config.rms_threshold
        assert rms(loud) > config.rms_threshold


class TestResample:
    def test_same_rate_is_passthrough(self):
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        assert resample(samples, 16000, 16000) is samples

    def test_downsample_length(self):
        """48kHz の 1 秒は 16kHz でちょうど 1/3 のサンプル数になること。"""
        samples = np.zeros(48000, dtype=np.float32)
        assert len(resample(samples, 48000, 16000)) == 16000

    def test_constant_signal_is_preserved(self):
        samples = np.full(48000, 0.5, dtype=np.float32)
        result = resample(samples, 48000, 16000)
        np.testing.assert_allclose(result, 0.5, rtol=1e-6)

    def test_linear_ramp_is_preserved(self):
        """線形補間なので、直線的な信号は形を保つこと。"""
        samples = np.linspace(0.0, 1.0, 48000).astype(np.float32)
        result = resample(samples, 48000, 16000)
        expected = np.linspace(0.0, 1.0, 16000)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_returns_float32(self):
        samples = np.zeros(48000, dtype=np.float64)
        assert resample(samples, 48000, 16000).dtype == np.float32


class TestHallucinationFilter:
    def test_short_fragment_is_rejected(self):
        assert Transcriber._is_hallucination("はい")

    def test_known_phrase_is_rejected(self):
        assert Transcriber._is_hallucination("ご視聴ありがとうございました")

    def test_known_phrase_embedded_is_rejected(self):
        assert Transcriber._is_hallucination("それではチャンネル登録をお願いします")

    def test_valid_speech_is_kept(self):
        assert not Transcriber._is_hallucination(
            "次回の会議は金曜日の午後3時からです。"
        )

    def test_exactly_min_length_is_kept(self):
        assert not Transcriber._is_hallucination("承知しました")


class TestSaveReport:
    CHUNKS = [
        TranscriptChunk(datetime(2026, 7, 11, 14, 3, 12), "最初の発言です。"),
        TranscriptChunk(datetime(2026, 7, 11, 14, 4, 12), "次の発言です。"),
    ]

    def test_report_contains_summary_and_transcript(self, tmp_path):
        path = save_report(self.CHUNKS, "最終要約テキスト", tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "最終要約テキスト" in content
        assert "[14:03:12] 最初の発言です。" in content
        assert "[14:04:12] 次の発言です。" in content

    def test_creates_missing_output_dir(self, tmp_path):
        target = tmp_path / "nested" / "dir"
        path = save_report(self.CHUNKS, "要約", target)
        assert path.exists()

    def test_dropped_chunks_are_disclosed(self, tmp_path):
        path = save_report(self.CHUNKS, "要約", tmp_path, dropped_count=3)
        content = path.read_text(encoding="utf-8")
        assert "破棄したチャンク: 3 件" in content

    def test_no_drop_notice_when_nothing_dropped(self, tmp_path):
        path = save_report(self.CHUNKS, "要約", tmp_path)
        assert "破棄" not in path.read_text(encoding="utf-8")


def _make_assistant() -> MeetingAssistant:
    """キュー制御のテスト用に、外部依存を持たないインスタンスを作る。"""
    return MeetingAssistant(
        Config(),
        capture=None,
        transcriber=None,
        summarizer=None,
        notifier=None,
    )


def _make_chunk(second: int) -> AudioChunk:
    return AudioChunk(datetime(2026, 7, 11, 14, 0, second), np.zeros(4))


class TestQueueBackpressure:
    def test_queue_is_bounded(self):
        assistant = _make_assistant()
        assert assistant._audio_queue.maxsize == MAX_PENDING_CHUNKS

    def test_enqueue_below_limit_keeps_everything(self):
        assistant = _make_assistant()
        for i in range(MAX_PENDING_CHUNKS):
            assistant._enqueue(_make_chunk(i % 60))
        assert assistant._audio_queue.qsize() == MAX_PENDING_CHUNKS
        assert assistant._dropped_count == 0

    def test_enqueue_over_limit_drops_oldest(self):
        assistant = _make_assistant()
        for i in range(MAX_PENDING_CHUNKS + 2):
            assistant._enqueue(_make_chunk(i % 60))

        assert assistant._audio_queue.qsize() == MAX_PENDING_CHUNKS
        assert assistant._dropped_count == 2
        # 最古の 2 件 (second=0, 1) が破棄され、先頭は second=2 になる
        head = assistant._audio_queue.get_nowait()
        assert head.recorded_at.second == 2

    def test_abort_discards_pending_and_stops_worker(self):
        """2 度目の Ctrl+C 相当: 未処理チャンクを捨ててもワーカーが停止すること。"""
        assistant = _make_assistant()
        for i in range(5):
            assistant._enqueue(_make_chunk(i))

        assistant._abort.set()
        try:
            while True:
                assistant._audio_queue.get_nowait()
        except queue.Empty:
            pass
        assistant._audio_queue.put(None)

        # _process_loop は番兵 (None) か abort フラグで即座に return する
        assistant._process_loop()
        assert assistant._audio_queue.qsize() == 0
