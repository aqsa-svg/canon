"""The demo page renders, and renders *without* contacting a provider.

Both halves matter. Streamlit's ``AppTest`` executes the script body, which a health check
does not — the script only runs when a session connects, so "the server started" says
nothing about whether the page works.

The second half is a real constraint rather than a nicety: this suite makes zero LLM
calls, and the demo now talks to a provider. So these tests exercise the load state, which
must stop before any call, and assert that it does. Driving the answered and clarify paths
would need a live call and free-tier quota, so those are covered by the harness instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.data

TIMEOUT_SECONDS = 120

# AppTest.from_file resolves a relative path against the calling file's directory, not the
# working directory, so this has to be absolute.
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _run() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT_SECONDS)
    app.run()
    return app


def test_the_page_loads_without_touching_a_provider() -> None:
    """The load state must be free.

    If pressing nothing made a call, opening the demo would spend free-tier quota the
    sweep needs, and this suite's zero-LLM-calls promise would be false.
    """
    app = _run()
    assert not app.exception, [error.value for error in app.exception]
    assert app.title[0].value == "Canon"

    # It stops at the prompt rather than answering, which is what makes the load free.
    assert app.info, "the page should prompt for a question before doing anything"
    assert "press" in app.info[0].value.lower()

    # Nothing was answered, so none of the result sections exist yet.
    headings = {heading.value for heading in app.markdown}
    assert not any("How it got there" in heading for heading in headings)


def test_the_four_examples_are_offered() -> None:
    """Each example demonstrates a different mechanism; losing one loses the point."""
    app = _run()
    labels = [button.label for button in app.button]

    assert "Ask" in labels
    examples = [label for label in labels if label != "Ask"]
    assert len(examples) == 4, f"expected four examples, got {examples}"

    joined = " ".join(examples).lower()
    # One ambiguous, one hitting the definition change, one tripping the gate.
    assert "ambiguous" in joined
    assert "definition changes" in joined
    assert "gate" in joined


def test_the_sidebar_shows_the_free_tier_budget() -> None:
    """On a free tier, remaining quota is information the viewer needs."""
    app = _run()
    labels = [metric.label for metric in app.sidebar.metric]
    assert any("free tokens" in label.lower() for label in labels), labels


def test_clicking_an_example_only_changes_the_question() -> None:
    """Selecting an example must not itself trigger a call.

    The button sets the text; the separate Ask button spends quota. Collapsing those into
    one click would make browsing the examples cost money.
    """
    app = _run()
    ambiguous = next(button for button in app.button if "ambiguous" in button.label.lower())
    ambiguous.click().run()

    assert not app.exception, [error.value for error in app.exception]
    assert app.info, "still waiting to be asked"
    assert "revenue" in app.text_input[0].value.lower()


class TestPublicExposure:
    """The page is deployed publicly against a shared free-tier quota.

    Every guard below is exercised by forcing its path, not by reading the code and
    concluding it would work. The partial-run refusal message in the variance report was
    wrong until it was actually triggered, which is the precedent for this class.
    """

    def test_an_example_replays_from_cache_and_spends_nothing(self) -> None:
        """The headline safety property: the four examples are free.

        This drives the real cache-only client, so it fails if the recorded responses are
        missing or the cache key stops matching the benchmark's.
        """
        app = _run()
        app.button[0].click().run()  # first example
        ask = next(button for button in app.button if button.label == "Ask")
        ask.click().run()

        assert not app.exception, [error.value for error in app.exception]
        captions = " ".join(caption.value for caption in app.caption)
        assert "Replay of a recorded run" in captions, captions[:400]
        assert "No model call was made" in captions

    def test_a_replay_is_never_dressed_up_as_a_live_call(self) -> None:
        """A cached answer and a live one are identical downstream.

        So the difference has to be stated. An interface that cannot tell the viewer
        which one they are looking at is the same defect as a latency mean that averages
        cache hits.
        """
        app = _run()
        app.button[0].click().run()
        next(button for button in app.button if button.label == "Ask").click().run()

        captions = " ".join(caption.value for caption in app.caption)
        assert "Replay" in captions
        assert "Live call" not in captions

    def test_the_session_cap_refuses_a_fourth_live_question(self) -> None:
        """A stranger cannot spend the day's quota three tokens at a time."""
        app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT_SECONDS)
        app.session_state["live_calls"] = 3
        app.run()
        app.text_input[0].set_value("how many sellers are there in Bahia").run()
        next(button for button in app.button if button.label == "Ask").click().run()

        assert not app.exception, [error.value for error in app.exception]
        warnings = " ".join(warning.value for warning in app.warning)
        assert "live questions" in warnings, warnings[:400]
        # It must point somewhere useful rather than just refusing.
        body = warnings + " ".join(block.value for block in app.markdown)
        assert "cost nothing" in body

    def test_a_question_matching_an_example_is_replayed_not_charged(self) -> None:
        """Typing an example's text verbatim takes the free path too.

        Routing on the button rather than on the question would charge a visitor for a
        question that is already on disk.
        """
        app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT_SECONDS)
        app.session_state["live_calls"] = 99  # live calls are exhausted
        app.run()
        app.text_input[0].set_value("What was net revenue in 2018?").run()
        next(button for button in app.button if button.label == "Ask").click().run()

        assert not app.exception, [error.value for error in app.exception]
        captions = " ".join(caption.value for caption in app.caption)
        assert "Replay of a recorded run" in captions, captions[:300]

    def test_the_daily_ceiling_refuses_and_points_at_the_examples(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Forced rather than reasoned about: the demo's own budget is spent.

        Metered under a separate ledger key from the benchmark. An earlier version read
        the provider's total, so a sweep that spent most of the day's tokens silently
        disabled the public demo -- found by this test expecting the session cap and
        getting the daily one.
        """
        import canon.agent.providers as providers

        class _SpentLedger:
            def usage(self, provider: str, day: str | None = None) -> object:
                return type("U", (), {"tokens": 10**9, "requests": 0})()

            def record(self, provider: str, *, tokens: int, requests: int = 1) -> None:
                return None

        monkeypatch.setattr(providers, "QuotaLedger", _SpentLedger)

        app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT_SECONDS)
        app.run()
        app.text_input[0].set_value("how many sellers are there in Bahia").run()
        next(button for button in app.button if button.label == "Ask").click().run()

        assert not app.exception, [error.value for error in app.exception]
        errors = " ".join(error.value for error in app.error)
        assert "daily budget" in errors.lower(), errors[:400]
        assert "00:00 UTC" in errors
        body = " ".join(block.value for block in app.markdown)
        assert "cost nothing" in body, "a refusal must point back at the free examples"

    def test_a_provider_failure_is_a_message_not_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error path a visitor is most likely to hit, driven for real."""
        from canon.agent.llm import LlmError
        from canon.agent.pipeline import CanonPipeline

        def _boom(self: CanonPipeline, question: str, context: object = None) -> None:
            raise LlmError("429 rate_limit_exceeded: slow down")

        monkeypatch.setattr(CanonPipeline, "answer", _boom)

        app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT_SECONDS)
        app.run()
        app.text_input[0].set_value("how many sellers are there in Bahia").run()
        next(button for button in app.button if button.label == "Ask").click().run()

        assert not app.exception, "a provider error must not surface as a stack trace"
        errors = " ".join(error.value for error in app.error)
        assert "could not answer" in errors.lower(), errors[:400]
        captions = " ".join(caption.value for caption in app.caption)
        assert "LlmError" in captions, "the real cause is still shown, just not as a crash"

    def test_every_example_has_a_committed_recording(self) -> None:
        """All four, not just the one the click test happens to drive.

        ``demo_cache/`` is keyed by a hash over the system prompt, the schema and the
        question. Any change to the prompt or to ``AgentDecision`` silently invalidates
        every key, and the first symptom would be a deployed demo that charges visitors
        for questions it advertises as free. This fails loudly instead.
        """
        import demo_config as demo
        from canon.agent.llm import CacheMiss, CallContext, LlmClient
        from canon.agent.pipeline import CanonPipeline

        pipeline = CanonPipeline(
            db_path=demo.DB_PATH,
            client=LlmClient(cache_only=True, cache_dir=demo.DEMO_CACHE),
            decision_log=None,
        )
        try:
            for example in demo.EXAMPLES:
                try:
                    result = pipeline.answer(
                        example.question,
                        context=CallContext(
                            system="canon", question_id=example.question_id, run_index=0
                        ),
                    )
                except CacheMiss:  # pragma: no cover - the failure this test exists for
                    pytest.fail(
                        f"no recording for {example.question_id} in demo_cache/. A prompt or "
                        "schema change invalidated the key; re-record before deploying."
                    )
                assert result.usage.total_tokens == 0, f"{example.question_id} spent tokens"
                assert result.answer is not None, f"{example.question_id} gave no answer"
        finally:
            pipeline.close()

    def test_the_demo_cache_is_tracked_by_git(self) -> None:
        """A gitignored recording is no recording at all on Streamlit Cloud."""
        import subprocess

        import demo_config as demo

        tracked = subprocess.run(
            ["git", "ls-files", str(demo.DEMO_CACHE.name)],
            capture_output=True,
            text=True,
            cwd=demo.REPO_ROOT,
            check=False,
        ).stdout.split()
        assert len(tracked) == len(demo.EXAMPLES), (
            f"expected {len(demo.EXAMPLES)} tracked recordings, found {len(tracked)}"
        )

    def test_without_a_key_the_examples_still_work(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A deployment with no key configured must still demo.

        This is what a visitor saw on the first Streamlit Cloud deploy: the sidebar
        reported no API key. The examples replay from ``demo_cache/`` and need no
        provider at all, so a missing key is a warning about one feature rather than a
        failure of the page.
        """
        for name in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(name, raising=False)

        app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT_SECONDS)
        app.run()
        assert not app.exception, [error.value for error in app.exception]

        warnings = " ".join(warning.value for warning in app.sidebar.warning)
        assert "No API key" in warnings, warnings[:300]
        assert "examples still work" in warnings, "it must say what still works"
        # Not fatal: the page carries on to offer the examples.
        assert len([b for b in app.button if b.label != "Ask"]) == 4

        app.button[0].click().run()
        next(button for button in app.button if button.label == "Ask").click().run()
        assert not app.exception, [error.value for error in app.exception]
        captions = " ".join(caption.value for caption in app.caption)
        assert "Replay of a recorded run" in captions, captions[:300]

    def test_the_missing_key_hint_covers_both_deployments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The original message told a Cloud user to create a file they cannot create."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT_SECONDS)
        app.run()
        hint = " ".join(caption.value for caption in app.sidebar.caption)
        assert "Secrets" in hint, "must name the Streamlit Cloud path"
        assert ".env" in hint, "must still name the local path"

    def test_a_broken_replay_is_reported_not_redacted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The replay path caught only CacheMiss, so everything else escaped.

        On Streamlit Cloud an escaped exception renders as a generic page with the
        message redacted and a traceback, which is the least useful possible output for
        a fault that is entirely inside this repository -- no provider is involved in a
        replay.
        """
        from canon.agent.pipeline import CanonPipeline

        def _boom(self: CanonPipeline, question: str, context: object = None) -> None:
            raise ValueError("1 validation error for QueryPlan")

        monkeypatch.setattr(CanonPipeline, "answer", _boom)

        app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT_SECONDS)
        app.run()
        app.button[0].click().run()
        next(button for button in app.button if button.label == "Ask").click().run()

        assert not app.exception, "a replay fault must not surface as a stack trace"
        errors = " ".join(error.value for error in app.error)
        assert "could not be replayed" in errors, errors[:300]
        assert "fault in Canon" in errors, "it must not blame the provider"
        captions = " ".join(caption.value for caption in app.caption)
        assert "validation error for QueryPlan" in captions, "the real cause must show"
