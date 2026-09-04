"""The deterministic SQL compiler. No LLM, no network, no clock.

``compile_intent`` is the entry point: it validates a ``QueryIntent`` against the
canonical layer and returns SQL, the metric card, and the caveats that must travel with
the number. It raises ``InvalidIntent`` before SQL exists when the request is illegal.
"""

from canon.compiler.build import CompiledQuery, compile_intent, compile_plan
from canon.compiler.plan import PERIOD_ALIAS, QueryPlan
from canon.compiler.validate import validate_intent

__all__ = [
    "PERIOD_ALIAS",
    "CompiledQuery",
    "QueryPlan",
    "compile_intent",
    "compile_plan",
    "validate_intent",
]
