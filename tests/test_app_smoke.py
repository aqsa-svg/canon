"""The demo page actually renders, and its refusal path actually refuses.

Streamlit's ``AppTest`` executes the script body, which a health check does not: the
script only runs when a session connects, so "the server started" says nothing about
whether the page works. No LLM is involved -- the Day 1 page is the compiler only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.data

TIMEOUT_SECONDS = 120

# AppTest.from_file resolves a relative path against the calling file's directory,
# not the working directory, so this has to be absolute.
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_the_page_renders_sql_a_definition_and_caveats() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT_SECONDS)
    app.run()

    assert not app.exception, [error.value for error in app.exception]
    assert app.title[0].value == "Canon"

    headings = {heading.value for heading in app.subheader}
    assert "SQL" in headings
    assert "Metric definition used" in headings
    assert any(heading.startswith("Caveats") for heading in headings)

    # The first code block is the compiled SQL, and it must be real SQL over a real table.
    compiled_sql = app.code[0].value
    assert compiled_sql.startswith("SELECT")
    assert "FROM" in compiled_sql


def test_an_illegal_pair_refuses_in_the_ui_without_producing_sql() -> None:
    """Choosing the wrong-grain trap in the UI must surface a refusal, not a number."""
    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT_SECONDS)
    app.run()
    app.selectbox[0].select("avg_order_value").run()
    app.multiselect[0].select("product_category").run()

    assert not app.exception, [error.value for error in app.exception]
    assert app.warning, "an illegal metric/dimension pair must warn"
    assert "dimension_not_valid_for_metric" in app.warning[0].value

    # And it must offer a way forward rather than a dead end.
    suggestions = [item.value for item in app.markdown if "instead" in item.value.lower()]
    assert suggestions
    assert "avg_item_price" in suggestions[0]
