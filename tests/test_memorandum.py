"""memorandum の純関数・キュー制御・保存処理・終了処理のテスト。

音声デバイスや Whisper / Ollama の実体は使わない (conftest.py でスタブ)。
"""

import logging
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pytest

import memorandum
from memorandum import (
    MAX_PENDING_CHUNKS,
    SUMMARY_BLOCK_CHARS,
    UNFINISHED_SUMMARY,
    ensure_output_dir_writable,
    AudioChunk,
    Config,
    MeetingAssistant,
    Notifier,
    Summarizer,
    Transcriber,
    TranscriptChunk,
    TranscriptLog,
    record_chunks,
    rms,
    resample,
    save_report,
    to_excerpts,
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


def _segments(*texts):
    return [{"text": t} for t in texts]


class TestHallucinationFilter:
    def test_short_fragment_is_rejected(self):
        assert Transcriber._clean(_segments("はい")) == ""

    def test_known_phrase_is_rejected(self):
        assert Transcriber._clean(_segments("ご視聴ありがとうございました")) == ""

    def test_known_phrase_embedded_is_rejected(self):
        assert (
            Transcriber._clean(_segments("それではチャンネル登録をお願いします")) == ""
        )

    def test_valid_speech_is_kept(self):
        text = "次回の会議は金曜日の午後3時からです。"
        assert Transcriber._clean(_segments(text)) == text

    def test_exactly_min_length_is_kept(self):
        assert Transcriber._clean(_segments("承知しました")) == "承知しました"

    def test_hallucination_does_not_take_down_the_whole_chunk(self):
        """定型文のセグメントだけを落とし、同じ 1 分の正常な発言は残すこと。"""
        cleaned = Transcriber._clean(
            _segments(
                "次回の会議は金曜日の午後3時からです。",
                "資料は山田さんが準備します。",
                "ご視聴ありがとうございました",
            )
        )
        assert (
            cleaned
            == "次回の会議は金曜日の午後3時からです。資料は山田さんが準備します。"
        )

    def test_phrase_in_real_speech_only_drops_that_segment(self):
        """通常の発言に定型文が含まれても、欠けるのはそのセグメントだけであること。"""
        cleaned = Transcriber._clean(
            _segments(
                "チャンネル登録者数は先月から一割伸びています。",
                "来週火曜までに見積もりを出します。",
            )
        )
        assert cleaned == "来週火曜までに見積もりを出します。"


def _at(minute: int, second: int = 0) -> datetime:
    return datetime(2026, 7, 11, 14, 0, 0) + timedelta(minutes=minute, seconds=second)


class TestSaveReport:
    CHUNKS = [
        TranscriptChunk(datetime(2026, 7, 11, 14, 3, 12), "最初の発言です。"),
        TranscriptChunk(datetime(2026, 7, 11, 14, 4, 12), "次の発言です。"),
    ]

    def test_report_contains_summary_and_transcript(self, tmp_path):
        path = save_report(self.CHUNKS, "最終要約テキスト", tmp_path / "log.txt")
        content = path.read_text(encoding="utf-8")
        assert "最終要約テキスト" in content
        assert "[14:03:12] 最初の発言です。" in content
        assert "[14:04:12] 次の発言です。" in content

    def test_creates_missing_output_dir(self, tmp_path):
        path = save_report(self.CHUNKS, "要約", tmp_path / "nested" / "dir" / "log.txt")
        assert path.exists()

    def test_dropped_chunks_are_disclosed(self, tmp_path):
        path = save_report(self.CHUNKS, "要約", tmp_path / "log.txt", dropped_count=3)
        assert "処理能力不足のため未処理のまま破棄したチャンク: 3 件" in path.read_text(
            encoding="utf-8"
        )

    def test_aborted_chunks_are_disclosed(self, tmp_path):
        path = save_report(self.CHUNKS, "要約", tmp_path / "log.txt", aborted_count=2)
        assert "打ち切りにより未処理のまま破棄したチャンク: 2 件" in path.read_text(
            encoding="utf-8"
        )

    def test_no_drop_notice_when_nothing_dropped(self, tmp_path):
        path = save_report(self.CHUNKS, "要約", tmp_path / "log.txt")
        assert "破棄" not in path.read_text(encoding="utf-8")

    def test_leaves_no_temporary_file(self, tmp_path):
        save_report(self.CHUNKS, "要約", tmp_path / "log.txt")
        assert [p.name for p in tmp_path.iterdir()] == ["log.txt"]


class TestOutputDirCheck:
    def test_creates_missing_dir(self, tmp_path):
        ensure_output_dir_writable(tmp_path / "new" / "dir")
        assert (tmp_path / "new" / "dir").is_dir()

    def test_unusable_dir_fails_at_startup(self, tmp_path):
        """会議の後ではなく起動時に、保存できないことが分かること。"""
        blocker = tmp_path / "not_a_dir.txt"
        blocker.write_text("")
        with pytest.raises(SystemExit):
            ensure_output_dir_writable(blocker / "sub")


class TestTranscriptLog:
    def test_transcript_survives_without_finalize(self, tmp_path):
        """終了処理を経ずにプロセスが消えても、それまでの原文がファイルに残ること。"""
        log = TranscriptLog(tmp_path / "log.txt")
        log.append(TranscriptChunk(_at(0), "最初の発言です。"))
        log.append(TranscriptChunk(_at(1), "次の発言です。"))

        content = log.path.read_text(encoding="utf-8")
        assert UNFINISHED_SUMMARY in content
        assert "[14:00:00] 最初の発言です。" in content
        assert "[14:01:00] 次の発言です。" in content

    def test_final_report_replaces_the_log(self, tmp_path):
        log = TranscriptLog(tmp_path / "log.txt")
        chunk = TranscriptChunk(_at(0), "最初の発言です。")
        log.append(chunk)
        save_report([chunk], "最終要約テキスト", log.path)

        content = log.path.read_text(encoding="utf-8")
        assert UNFINISHED_SUMMARY not in content
        assert "最終要約テキスト" in content
        assert content.count("最初の発言です。") == 1

    def test_no_file_until_first_transcript(self, tmp_path):
        TranscriptLog(tmp_path / "log.txt")
        assert not (tmp_path / "log.txt").exists()


def _reader(n_blocks, interrupt_after=None):
    """n_blocks 個まで 1 バイトのブロックを返し、interrupt_after 個目の後で Ctrl+C を送出する。"""
    count = 0

    def read():
        nonlocal count
        if interrupt_after is not None and count == interrupt_after:
            raise KeyboardInterrupt
        count += 1
        return b"x"

    return read


def _no_interrupt(fn, *args):
    """fn を呼び、KeyboardInterrupt が漏れたらテストの失敗にする。

    テスト中に KeyboardInterrupt が漏れると、pytest は失敗ではなく実行全体の
    中断として扱い、FAILED が出ないまま終わる。Ctrl+C を模したテストでは
    予定外の KeyboardInterrupt を必ず失敗として拾う。
    """
    try:
        return fn(*args)
    except KeyboardInterrupt:
        pytest.fail("予定外の KeyboardInterrupt が送出された")


def _record(read, blocks_per_chunk=4, min_partial_blocks=2):
    return record_chunks(
        read_block=read,
        blocks_per_chunk=blocks_per_chunk,
        min_partial_blocks=min_partial_blocks,
        to_waveform=lambda data: np.frombuffer(data, np.uint8).astype(np.float32),
        chunk_seconds=60,
    )


class TestRecordChunks:
    def test_yields_fixed_length_chunks(self):
        gen = _record(_reader(100))
        assert [len(next(gen).samples) for _ in range(3)] == [4, 4, 4]

    def test_partial_chunk_is_kept_on_ctrl_c(self):
        """録音途中で Ctrl+C されても、そこまでの音声を渡してから終了すること。"""
        gen = _record(_reader(100, interrupt_after=4 + 3))
        assert len(_no_interrupt(next, gen).samples) == 4
        assert len(_no_interrupt(next, gen).samples) == 3  # 録音途中の 3 ブロック
        with pytest.raises(KeyboardInterrupt):
            next(gen)

    def test_too_short_partial_is_dropped(self):
        gen = _record(_reader(100, interrupt_after=4 + 1))
        assert len(_no_interrupt(next, gen).samples) == 4
        with pytest.raises(KeyboardInterrupt):
            next(gen)

    def test_recorded_at_is_start_of_chunk(self):
        def slow_read():
            time.sleep(0.02)
            return b"x"

        before = datetime.now()
        chunk = next(_record(slow_read, blocks_per_chunk=5))
        assert before <= chunk.recorded_at < before + timedelta(seconds=0.05)


class _FakeChat:
    """ollama.chat の代わりに呼ばれ、渡されたプロンプトを記録する。"""

    def __init__(self, reply=lambda prompt: "要約"):
        self.prompts = []
        self.options = []
        self._reply = reply

    def __call__(self, model, messages, options=None):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        self.options.append(options)
        return {"message": {"content": self._reply(prompt)}}


@pytest.fixture
def fake_chat(monkeypatch):
    def install(reply=lambda prompt: "要約"):
        chat = _FakeChat(reply)
        monkeypatch.setattr(memorandum.ollama, "chat", chat)
        return chat

    return install


def _meeting(minutes: int, chars_per_minute: int = 300):
    return [
        TranscriptChunk(_at(m), f"発言{m:03d}" + "あ" * (chars_per_minute - 5))
        for m in range(minutes)
    ]


# SECTION_PROMPT / FINAL_PROMPT の指示文と時間帯表記のぶんの余裕
PROMPT_OVERHEAD = 200


class TestSummarizeMeeting:
    def test_short_meeting_is_summarized_in_one_call(self, fake_chat):
        chat = fake_chat()
        Summarizer("m").summarize_meeting(_meeting(5))
        assert len(chat.prompts) == 1
        assert chat.prompts[0].startswith(
            "あなたはプロの書記です。以下の会議の全発言記録"
        )
        assert all(f"発言{m:03d}" in chat.prompts[0] for m in range(5))

    def test_one_hour_meeting_is_summarized_in_one_call(self, fake_chat):
        """60 分 (約 18,000 文字) の会議は分割せず、全文を 1 回で渡すこと。"""
        chat = fake_chat()
        Summarizer("m").summarize_meeting(_meeting(60))
        assert len(chat.prompts) == 1
        assert all(f"発言{m:03d}" in chat.prompts[0] for m in range(60))

    def test_every_meeting_prompt_asks_to_keep_specifics(self, fake_chat):
        """日付・金額などを落とさない指示が、全段の要約プロンプトに入っていること。"""
        chat = fake_chat()
        Summarizer("m").summarize_meeting(_meeting(180))
        # 定数そのものではなく文言で確かめる (定数を空にされても気づけるように)
        assert all(
            "日付・金額・数量・担当者" in p and "省略せず" in p for p in chat.prompts
        )

    def test_context_length_is_always_explicit(self, fake_chat):
        """num_ctx を渡さないと Ollama の既定 (4K) で切り捨てられるため、毎回明示すること。"""
        chat = fake_chat()
        summarizer = Summarizer("m")
        summarizer.summarize_chunk("直近の発言です。")
        summarizer.summarize_meeting(_meeting(180))
        assert chat.options and all(
            o
            == {
                "num_ctx": memorandum.LLM_NUM_CTX,
                "num_predict": memorandum.LLM_MAX_OUTPUT_TOKENS,
            }
            for o in chat.options
        )

    def test_long_meeting_never_exceeds_budget(self, fake_chat):
        """3 時間の会議でも、1 回に渡す量が予算を超えないこと (= 前半が切り捨てられない)。"""
        chat = fake_chat()
        Summarizer("m").summarize_meeting(_meeting(180))
        assert len(chat.prompts) > 1
        assert all(
            len(p) <= SUMMARY_BLOCK_CHARS + PROMPT_OVERHEAD for p in chat.prompts
        )

    def test_long_meeting_covers_every_minute_exactly_once(self, fake_chat):
        """段階要約の最初の段で、全ての発言がちょうど 1 回ずつ LLM に渡ること。"""
        chat = fake_chat()
        Summarizer("m").summarize_meeting(_meeting(180))
        sections = [p for p in chat.prompts if p.startswith("以下は会議の一部")]
        for m in range(180):
            assert sum(f"発言{m:03d}" in p for p in sections) == 1
        assert chat.prompts[-1].startswith(
            "あなたはプロの書記です。以下は会議を時間帯ごとに"
        )

    def test_sections_are_labelled_with_their_time_span(self, fake_chat):
        chat = fake_chat()
        Summarizer("m").summarize_meeting(_meeting(180))
        assert "（14:00〜" in chat.prompts[0]
        assert "〜16:59】" in chat.prompts[-1]

    def test_oversized_single_chunk_is_split_without_loss(self):
        """1 チャンクだけで予算を超えても、切り詰めずに刻むこと。"""
        text = "".join(f"{i:05d}" for i in range(12000))  # 60000 文字
        excerpts = to_excerpts([TranscriptChunk(_at(0), text)], SUMMARY_BLOCK_CHARS)
        assert "".join(e.text for e in excerpts) == text
        assert all(e.size <= SUMMARY_BLOCK_CHARS for e in excerpts)

    def test_oversized_single_chunk_stays_within_budget(self, fake_chat):
        chat = fake_chat()
        Summarizer("m").summarize_meeting([TranscriptChunk(_at(0), "あ" * 60000)])
        sections = [p for p in chat.prompts if p.startswith("以下は会議の一部")]
        assert len(sections) == 3
        assert all(
            len(p) <= SUMMARY_BLOCK_CHARS + PROMPT_OVERHEAD for p in chat.prompts
        )

    def test_oversized_note_does_not_break_budget(self, fake_chat):
        """LLM が区間メモを予算より長く返しても、次の段の入力が予算を超えないこと。"""
        chat = fake_chat(reply=lambda prompt: "長" * 30000)
        Summarizer("m").summarize_meeting(_meeting(180))
        assert all(
            len(p) <= SUMMARY_BLOCK_CHARS + PROMPT_OVERHEAD for p in chat.prompts
        )

    def test_terminates_when_llm_ignores_length_limit(self, fake_chat):
        """LLM が文字数の指示を無視して長く返しても、有限回で終わり予算内に収まること。"""
        chat = fake_chat(
            reply=lambda prompt: "長" * 13000
        )  # 2 つ並ぶと予算を超える長さ
        Summarizer("m").summarize_meeting(_meeting(180))
        assert len(chat.prompts) < 200
        assert len(chat.prompts[-1]) <= SUMMARY_BLOCK_CHARS + PROMPT_OVERHEAD


class TestNotifier:
    def test_macos_passes_text_as_arguments(self, monkeypatch):
        """要約を AppleScript に埋め込まず引数で渡すこと (引用符で壊れない)。"""
        calls = []
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(subprocess, "run", lambda args, **kw: calls.append(args))
        Notifier(enabled=True).send("タイトル", '引用符 " と \\ を含む要約')
        assert calls[0][0] == "osascript"
        assert calls[0][-2:] == ["タイトル", '引用符 " と \\ を含む要約']

    def test_failure_is_reported_only_once(self, monkeypatch, caplog):
        def fail(*args, **kwargs):
            raise subprocess.CalledProcessError(1, "osascript")

        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(subprocess, "run", fail)
        notifier = Notifier(enabled=True)
        with caplog.at_level(logging.WARNING, logger="memorandum"):
            for _ in range(3):
                notifier.send("タイトル", "本文")
        assert len(caplog.records) == 1

    def test_unsupported_os_does_nothing(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("呼ばれた"))
        Notifier(enabled=True).send("タイトル", "本文")


def _make_assistant(config=None, **deps) -> MeetingAssistant:
    """外部依存を持たないインスタンスを作る。必要な依存だけ差し込む。"""
    base = dict(
        capture=None, transcriber=None, summarizer=None, notifier=Notifier(False)
    )
    base.update(deps)
    return MeetingAssistant(config or Config(), **base)


def _make_chunk(second: int) -> AudioChunk:
    return AudioChunk(datetime(2026, 7, 11, 14, 0, second), np.ones(4))


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


class _Capture:
    """n 個のチャンクを渡したあと、Ctrl+C を押されたかのように終わる音源。"""

    def __init__(self, n):
        self._n = n

    def chunks(self):
        for i in range(self._n):
            yield _make_chunk(i)
        raise KeyboardInterrupt


class _Transcriber:
    def __init__(self, gate=None):
        self._gate = gate  # 与えられたら、最初の 1 件をその条件が満たされるまで止める
        self.calls = 0

    def transcribe(self, samples):
        self.calls += 1
        if self._gate is not None and self.calls == 1:
            while not self._gate():
                time.sleep(0.005)
        return f"{self.calls}件目の発言です。"


class _Summarizer:
    def __init__(self, meeting=lambda chunks: "全体の要約"):
        self._meeting = meeting
        self.meeting_calls = 0

    def summarize_chunk(self, text):
        return "要約"

    def summarize_meeting(self, chunks):
        self.meeting_calls += 1
        return self._meeting(chunks)


def _report(tmp_path) -> str:
    (path,) = tmp_path.glob("meeting_log_*.txt")
    return path.read_text(encoding="utf-8")


class TestShutdown:
    def test_ctrl_c_saves_summary_and_all_transcripts(self, tmp_path):
        assistant = _make_assistant(
            Config(output_dir=tmp_path),
            capture=_Capture(3),
            transcriber=_Transcriber(),
            summarizer=_Summarizer(),
        )
        _no_interrupt(assistant.run)

        content = _report(tmp_path)
        assert "全体の要約" in content
        assert all(f"{n}件目の発言です。" in content for n in (1, 2, 3))
        assert UNFINISHED_SUMMARY not in content

    def test_ctrl_c_during_final_summary_keeps_transcript(self, tmp_path):
        def interrupted(chunks):
            raise KeyboardInterrupt

        assistant = _make_assistant(
            Config(output_dir=tmp_path),
            capture=_Capture(2),
            transcriber=_Transcriber(),
            summarizer=_Summarizer(meeting=interrupted),
        )
        _no_interrupt(assistant.run)

        content = _report(tmp_path)
        assert "Ctrl+C により中断" in content
        assert "1件目の発言です。" in content and "2件目の発言です。" in content

    def test_second_ctrl_c_skips_summary_and_discloses_discarded(self, tmp_path):
        """2 度目の Ctrl+C: 未処理を捨て、最終要約を待たずに保存し、捨てた件数を明記する。"""
        summarizer = _Summarizer()
        assistant = _make_assistant(
            Config(output_dir=tmp_path),
            capture=_Capture(3),
            # 1 件目の処理中に打ち切られた状況を作る (2・3 件目はキューに残る)
            transcriber=_Transcriber(gate=lambda: assistant._abort.is_set()),
            summarizer=summarizer,
        )

        def second_ctrl_c(worker):
            raise KeyboardInterrupt

        assistant._drain_and_join = second_ctrl_c
        _no_interrupt(assistant.run)

        content = _report(tmp_path)
        assert summarizer.meeting_calls == 0
        assert "最終要約は省略しました" in content
        assert "1件目の発言です。" in content
        assert "打ち切りにより未処理のまま破棄したチャンク: 2 件" in content

    def test_abort_stops_worker_and_empties_queue(self):
        """_abort_and_join 自体を呼び、ワーカーが止まってキューが空になること。"""
        assistant = _make_assistant(
            transcriber=_Transcriber(gate=lambda: assistant._abort.is_set()),
            summarizer=_Summarizer(),
        )
        assistant._log.append = lambda chunk: None  # カレントディレクトリに書かない
        for i in range(5):
            assistant._enqueue(_make_chunk(i))
        worker = threading.Thread(target=assistant._process_loop, daemon=True)
        worker.start()

        assistant._abort_and_join(worker)

        assert not worker.is_alive()
        assert assistant._audio_queue.qsize() == 0
        assert assistant._aborted_count == 4  # 処理中だった 1 件以外

    def test_transcript_is_on_disk_before_finalize(self, tmp_path):
        """終了処理に入る前 (= ウィンドウを閉じられた時点) で原文がファイルにあること。"""
        assistant = _make_assistant(
            Config(output_dir=tmp_path),
            transcriber=_Transcriber(),
            summarizer=_Summarizer(),
        )
        assistant._process_chunk(_make_chunk(0))
        assistant._process_chunk(_make_chunk(1))

        content = _report(tmp_path)
        assert UNFINISHED_SUMMARY in content
        assert "1件目の発言です。" in content and "2件目の発言です。" in content

    def test_chunk_taken_by_worker_at_abort_is_counted(self):
        """打ち切りの直前にワーカーがキューから取った 1 件も、破棄件数に数えること。"""
        assistant = _make_assistant()
        assistant._enqueue(_make_chunk(0))
        assistant._abort.set()
        assistant._process_loop()
        assert assistant._aborted_count == 1

    def test_no_file_when_nothing_was_said(self, tmp_path):
        class _Silent(_Transcriber):
            def transcribe(self, samples):
                return ""

        assistant = _make_assistant(
            Config(output_dir=tmp_path),
            capture=_Capture(2),
            transcriber=_Silent(),
            summarizer=_Summarizer(),
        )
        _no_interrupt(assistant.run)
        assert list(tmp_path.iterdir()) == []
