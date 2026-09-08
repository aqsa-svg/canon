"""The only place the LLM lives. It fills a QueryIntent and nothing else.

``CanonPipeline`` is the entry point: question in, verified answer or clarifying
question out. The refusal decision is made by ``policy`` (deterministic, over the
model's metric scores), and every executed result passes through ``verify`` before a
number is reported.
"""

from canon.agent.llm import MODEL_ID, CacheMiss, LlmClient, LlmError, Usage
from canon.agent.pipeline import CanonPipeline, PipelineResult
from canon.agent.policy import PolicyOutcome, PolicyRule, PolicyThresholds, RefusalDecision
from canon.agent.prompt import DataWindow, build_system_prompt
from canon.agent.verify import VerificationFailure, VerificationReport, verify

__all__ = [
    "MODEL_ID",
    "CacheMiss",
    "CanonPipeline",
    "DataWindow",
    "LlmClient",
    "LlmError",
    "PipelineResult",
    "PolicyOutcome",
    "PolicyRule",
    "PolicyThresholds",
    "RefusalDecision",
    "Usage",
    "VerificationFailure",
    "VerificationReport",
    "build_system_prompt",
    "verify",
]
