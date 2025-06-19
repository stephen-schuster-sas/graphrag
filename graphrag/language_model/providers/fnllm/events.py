# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""FNLLM llm events provider."""

from typing import Any

from fnllm.events import LLMEvents

from graphrag.index.typing.error_handler import ErrorHandlerFn
from fnllm.types.io import LLMOutput
import logging

# Set up a logger
log = logging.getLogger(__name__)


class FNLLMEvents(LLMEvents):
    """FNLLM events handler that calls the error handler."""

    def __init__(self, on_error: ErrorHandlerFn):
        self._on_error = on_error

    async def on_error(
        self,
        error: BaseException | None,
        traceback: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Handle an fnllm error."""
        self._on_error(error, traceback, arguments)

    async def on_llm_response(self, response: LLMOutput, **kwargs) -> None:
        """Handle LLM responses."""
        if response.usage:
            log.info(f"LLM Token Usage: {response.usage}") # Or use any other logging mechanism
        await self.callbacks.on_llm_response(response, **kwargs)
