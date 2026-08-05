"""Portable LLM-provider and prompt subsystem.

Self-contained: importing this package (or any of its submodules other than
concrete adapters) never requires an optional provider dependency such as
``openai``, ``requests``, or ``transformers`` to be installed, and never
imports game, agent, experiment, or CLI code. See ``PORTING_MANIFEST.md`` in
this directory for what a destination repository needs to copy alongside it.

``providers`` and ``prompts`` are independent siblings — neither imports the
other — so this top-level package intentionally re-exports nothing itself.
Import ``mas_cc.llm_runtime.providers`` and ``mas_cc.llm_runtime.prompts``
directly.
"""
