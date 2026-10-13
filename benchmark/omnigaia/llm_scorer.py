"""LLM-based semantic similarity scorer for OmniGAIA."""
from __future__ import annotations

import asyncio
import os
from typing import Optional, Tuple

from openai import AsyncOpenAI

from base.engine.async_llm import LLMsConfig, llm_retry
from base.engine.logs import logger

DEFAULT_JUDGE_MODEL = "qwen3.7-plus"
JUDGE_PROMPT = """You are a precise answer evaluator. Given the original question, determine if the predicted answer is semantically equivalent to the expected answer.

IMPORTANT RULES:
1. Focus on SEMANTIC MEANING, not exact string match.
2. Date format differences are EQUIVALENT: "January 2018" ~ "2018-01" ~ "Jan 2018".
3. Currency format differences are EQUIVALENT: "$500 billion" ~ "500 billion dollars".
4. Percentage format differences are EQUIVALENT: "90%" ~ "90 percent".
5. Approximate values are EQUIVALENT: "nearly 90%" ~ "90%", "about 500" ~ "500".
6. A numeric value WITH or WITHOUT a unit/label word is EQUIVALENT as long as the number is the same.
7. Thousand separators and formatting differences are EQUIVALENT.
8. Numbers must actually match in value.
9. If the predicted answer contains the expected answer with correct context, it's EQUIVALENT.
10. If a list or multi-part response contains the expected answer, treat it as EQUIVALENT.
11. Order matters only when the question explicitly asks for ordering.
12. Minor wording differences that preserve meaning are EQUIVALENT.
13. Use the original question as context.
14. Extra correct context is OK when the key answer matches.

Original Question: {question}
Expected Answer: {expected}
Predicted Answer: {predicted}

Respond with ONLY one word: "EQUIVALENT" or "DIFFERENT"
"""


def _get_llm_config(model_name: Optional[str] = None) -> Tuple[str, str, str]:
    model_name = model_name or DEFAULT_JUDGE_MODEL
    try:
        llms_config = LLMsConfig.default()
        model_config = llms_config.get(model_name)
        if model_config:
            return (model_config.key, model_config.base_url, model_config.model)
    except Exception:
        pass
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("LLM_API_BASE")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return (api_key, base_url, model_name)


async def llm_semantic_score(
    predicted: str,
    expected: str,
    model: Optional[str] = None,
    question: Optional[str] = None,
) -> float:
    if not predicted or not expected:
        return 0.0
    if predicted.strip().lower() == expected.strip().lower():
        return 1.0
    api_key, base_url, model_name = _get_llm_config(model)
    if not api_key:
        logger.warning("[LLM Scorer] No API key available, falling back to 0.0")
        return 0.0
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    prompt = JUDGE_PROMPT.format(
        question=question or "(not provided)",
        expected=expected,
        predicted=predicted,
    )
    try:
        completion = await llm_retry(
            client.chat.completions.create,
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
            label=f"LLMScorer({model_name})",
        )
        response = completion.choices[0].message.content.strip().upper()
        if "EQUIVALENT" in response:
            logger.info(f"[LLM Scorer] EQUIVALENT: '{predicted}' ~ '{expected}'")
            return 1.0
        logger.info(f"[LLM Scorer] DIFFERENT: '{predicted}' ≠ '{expected}'")
        return 0.0
    except Exception as e:
        logger.error(f"[LLM Scorer] Error: {e}")
        return 0.0


def llm_semantic_score_sync(
    predicted: str,
    expected: str,
    model: Optional[str] = None,
    question: Optional[str] = None,
) -> float:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    llm_semantic_score(predicted, expected, model, question),
                )
                return future.result(timeout=30)
        return loop.run_until_complete(
            llm_semantic_score(predicted, expected, model, question)
        )
    except Exception as e:
        logger.error(f"[LLM Scorer Sync] Error: {e}")
        return 0.0
