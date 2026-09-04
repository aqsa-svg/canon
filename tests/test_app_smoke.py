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
