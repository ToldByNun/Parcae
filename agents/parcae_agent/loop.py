"""Agent loop: model turn ↔ ToolBridge until stop / success / budget."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

from parcae_agent.config import AgentConfig
from parcae_agent.llm import ChatMessage, LlmClient, LlmError, ToolCall
from parcae_agent.prompts import liber_primus_system_prompt
from parcae_agent.tool_bridge import ToolBridge, ToolInvocationResult
from parcae_agent.tool_schemas import openai_tools
from parcae_agent.transcript import TranscriptWriter


class AgentStopReason(str, Enum):
    Completed = "completed"  # model stopped without tool calls
    Succeeded = "succeeded"  # success criterion met
    BudgetSteps = "budget_steps"
    BudgetToolCalls = "budget_tool_calls"
    BudgetWall = "budget_wall"
    Error = "error"


@dataclass(frozen=True, slots=True)
class AgentStep:
    index: int
    assistant: ChatMessage
    tool_results: tuple[ToolInvocationResult, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    stop_reason: AgentStopReason
    messages: tuple[ChatMessage, ...]
    steps: tuple[AgentStep, ...]
    tool_calls: int
    wall_seconds: float
    final_text: str | None = None
    error: str | None = None
    run_id: str | None = None
    transcript_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.stop_reason in {
            AgentStopReason.Completed,
            AgentStopReason.Succeeded,
        }

    @property
    def exit_code(self) -> int:
        if self.stop_reason == AgentStopReason.Succeeded:
            return 0
        if self.stop_reason == AgentStopReason.Completed:
            return 0
        if self.stop_reason == AgentStopReason.Error:
            return 2
        return 1  # budget exhaustion


StepHook = Callable[[AgentStep], None]


class AgentLoop:
    """CMD agent loop over an OpenAI-compatible LLM + ToolBridge."""

    def __init__(
        self,
        config: AgentConfig,
        llm: LlmClient,
        bridge: ToolBridge,
        *,
        temperature: float = 0.0,
        on_step: StepHook | None = None,
        clock: Callable[[], float] | None = None,
        transcript: TranscriptWriter | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._bridge = bridge
        self._temperature = temperature
        self._on_step = on_step
        self._clock = clock or time.monotonic
        self._transcript = transcript

    @property
    def config(self) -> AgentConfig:
        return self._config

    def run(self, user_prompt: str) -> AgentRunResult:
        """Run until the model stops, a success criterion hits, or a budget ends."""
        started = self._clock()
        budgets = self._config.budgets
        tools = openai_tools()
        system_text = liber_primus_system_prompt(
            workspace=self._config.workspace,
            allow_cuda=self._config.allow_cuda,
        )
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_text),
            ChatMessage(role="user", content=user_prompt),
        ]
        if self._transcript is not None:
            self._transcript.append("system", summary="liber primus system prompt")
            self._transcript.append("user", summary=user_prompt)

        steps: list[AgentStep] = []
        tool_calls = 0

        for step_index in range(budgets.max_steps):
            if self._clock() - started >= budgets.max_wall_seconds:
                return self._finish(
                    AgentStopReason.BudgetWall,
                    messages,
                    steps,
                    tool_calls,
                    started,
                )

            try:
                completion = self._llm.chat_completions(
                    messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=self._temperature,
                )
            except LlmError as exc:
                return self._finish(
                    AgentStopReason.Error,
                    messages,
                    steps,
                    tool_calls,
                    started,
                    error=str(exc),
                )

            assistant = completion.message
            messages.append(assistant)

            if not assistant.tool_calls:
                step = AgentStep(index=step_index, assistant=assistant)
                steps.append(step)
                if self._transcript is not None:
                    self._transcript.append(
                        "assistant",
                        summary=assistant.content or "(empty)",
                    )
                if self._on_step is not None:
                    self._on_step(step)
                return self._finish(
                    AgentStopReason.Completed,
                    messages,
                    steps,
                    tool_calls,
                    started,
                    final_text=assistant.content,
                )

            # Budget check before executing tools (hard stop).
            pending = len(assistant.tool_calls)
            if tool_calls + pending > budgets.max_tool_calls:
                step = AgentStep(index=step_index, assistant=assistant)
                steps.append(step)
                if self._transcript is not None:
                    self._transcript.append(
                        "assistant",
                        summary=assistant.content or f"{pending} tool call(s) blocked by budget",
                    )
                if self._on_step is not None:
                    self._on_step(step)
                return self._finish(
                    AgentStopReason.BudgetToolCalls,
                    messages,
                    steps,
                    tool_calls,
                    started,
                    final_text=assistant.content,
                )

            if self._clock() - started >= budgets.max_wall_seconds:
                return self._finish(
                    AgentStopReason.BudgetWall,
                    messages,
                    steps,
                    tool_calls,
                    started,
                    final_text=assistant.content,
                )

            if self._transcript is not None:
                names = ", ".join(tc.name for tc in assistant.tool_calls)
                self._transcript.append(
                    "assistant",
                    summary=assistant.content or f"tool_calls: {names}",
                )

            invocations: list[ToolInvocationResult] = []
            succeeded = False
            for call in assistant.tool_calls:
                invocation = self._bridge.invoke_tool_call(call)
                invocations.append(invocation)
                tool_calls += 1
                messages.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=call.id,
                        name=call.name,
                        content=json.dumps(invocation.envelope, ensure_ascii=False),
                    )
                )
                if self._transcript is not None:
                    self._transcript.append_tool(invocation, arguments=call.arguments)
                if _is_success_criterion(call, invocation):
                    succeeded = True

            step = AgentStep(
                index=step_index,
                assistant=assistant,
                tool_results=tuple(invocations),
            )
            steps.append(step)
            if self._on_step is not None:
                self._on_step(step)

            if succeeded:
                return self._finish(
                    AgentStopReason.Succeeded,
                    messages,
                    steps,
                    tool_calls,
                    started,
                    final_text=assistant.content,
                )

        return self._finish(
            AgentStopReason.BudgetSteps,
            messages,
            steps,
            tool_calls,
            started,
        )

    def _finish(
        self,
        reason: AgentStopReason,
        messages: Sequence[ChatMessage],
        steps: Sequence[AgentStep],
        tool_calls: int,
        started: float,
        *,
        final_text: str | None = None,
        error: str | None = None,
    ) -> AgentRunResult:
        if self._transcript is not None:
            summary = reason.value
            if error:
                summary = f"{reason.value}: {error}"
            elif final_text:
                summary = f"{reason.value}: {final_text}"
            self._transcript.append(
                "finish",
                summary=summary,
                ok=reason in {AgentStopReason.Completed, AgentStopReason.Succeeded},
            )
        return AgentRunResult(
            stop_reason=reason,
            messages=tuple(messages),
            steps=tuple(steps),
            tool_calls=tool_calls,
            wall_seconds=max(0.0, self._clock() - started),
            final_text=final_text,
            error=error,
            run_id=self._transcript.run_id if self._transcript else None,
            transcript_path=self._transcript.path if self._transcript else None,
        )


def _is_success_criterion(
    call: ToolCall, invocation: ToolInvocationResult
) -> bool:
    if not invocation.ok:
        return False
    if call.name == "validate":
        return True
    if call.name == "hypothesis_set_status":
        try:
            args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            return False
        status = args.get("status") if isinstance(args, dict) else None
        return isinstance(status, str) and status.strip().lower() == "promoted"
    return False
