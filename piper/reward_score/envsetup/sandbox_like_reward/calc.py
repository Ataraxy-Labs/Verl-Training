#!/usr/bin/env python3
"""
Calculation utilities for LLM judge evaluation.
"""

import asyncio
import os
from typing import Annotated, List, Optional, Tuple

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState, create_react_agent
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    after_log
)
import logging
from openai import (
    APIError,
    APITimeoutError,
    InternalServerError,
    RateLimitError
)

from piper.reward_score.envsetup.sandbox_like_reward.prompts import create_evaluation_prompt
from piper.reward_score.envsetup.sandbox_like_reward.utils import default_execute_bash_command, gather_repo_exploration, truncate

# Set up logging for retry attempts
logger = logging.getLogger(__name__)

# Set up logging

max_turns = 10


def limit_turns(state: dict) -> dict:
    """Post-model hook to limit the number of turns for the ReAct agent."""
    ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    if len(ai_messages) > max_turns:
        stop_message = AIMessage(content="Sorry, need more steps to process this request.")
        return state | {"messages": [RemoveMessage(id=state["messages"][-1].id), stop_message]}
    return state


def create_tool(repository: str, max_length: int = 8000):
    @tool
    async def bash_tool(command: str, state: Annotated[dict, InjectedState]) -> str:
        """Execute a bash command in the repository."""
        result = await default_execute_bash_command(command, repository)
        ai_messages = [ai_msg for ai_msg in state["messages"] if isinstance(ai_msg, AIMessage)]
        len_ai_messages = len(ai_messages)
        turns_info = f"You have only {max_turns - len_ai_messages} turns left for tool execution. Please deliver your final answer before you run out of turns."
        return f"{turns_info}\nExit code: {result['exit_code']}\nstdout: {truncate(result['stdout'], max_length)}\nstderr: {truncate(result['stderr'], max_length)}"
    return bash_tool


class LLMJudgeBatch:
    """LLM Judge with batch processing capabilities."""

    def __init__(self,
                 model: str = "gpt-4o",
                 temperature: float = 0.0,
                 debug: bool = False,
                 is_agent: bool = False,
                 use_exploration: bool = True,
                 version: str = "v1",
                 max_concurrent_requests: int = 5):
        """Initialize the LLM judge.

        Args:
            model: OpenAI model to use
            temperature: Temperature for generation
            debug: Enable debug logging
            is_agent: Use agent mode with tools
            use_exploration: Enable repository exploration
            version: Prompt version
            max_concurrent_requests: Maximum number of concurrent API requests (default: 5)
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_retries=5,  # Retry up to 5 times for transient errors
            timeout=120,    # 2 minute timeout for requests
            request_timeout=120  # Request-level timeout
        )
        self.debug = debug
        self.is_agent = is_agent
        self.use_exploration = use_exploration
        self.version = version
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def _rate_limited_call_with_retry(self, task_callable):
        """Execute a task with rate limiting and retry logic for transient errors.

        Args:
            task_callable: A callable that returns a coroutine (not a coroutine itself).
                          This allows retries to work properly.
        """
        async with self.semaphore:
            # Retry configuration:
            # - Retry on transient errors (500, 429, timeout, generic API errors)
            # - Stop after 5 attempts
            # - Wait with exponential backoff: 2s, 4s, 8s, 16s, 32s
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type((
                    InternalServerError,      # 500 errors
                    RateLimitError,           # 429 rate limit errors
                    APITimeoutError,          # Timeout errors
                    APIError                  # Generic API errors
                )),
                stop=stop_after_attempt(5),
                wait=wait_exponential(multiplier=2, min=2, max=60),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True
            ):
                with attempt:
                    # Call the callable to get a fresh coroutine for each attempt
                    result = await task_callable()
                    return result

    async def evaluate_batch_async(self, scripts: List[str],
                                   repo_names: Optional[List[str]] = None) -> List[dict]:
        """Evaluate a batch of scripts asynchronously with rate limiting."""
        if repo_names is None:
            repo_names = ["unknown"] * len(scripts)

        import time
        start_time = time.time()
        print(f"[LLMJudgeBatch] Evaluating {len(scripts)} scripts with max {self.max_concurrent_requests} concurrent requests")
        print(f"[LLMJudgeBatch] Starting evaluation at {time.strftime('%H:%M:%S')}")

        # Create async tasks for each prompt
        tasks = []
        prompts = []
        for script, repo_name in zip(scripts, repo_names):
            if self.version == "v1":
                if self.use_exploration:
                    repo_exploration = await gather_repo_exploration(repo_name)
                else:
                    repo_exploration = None
                prompt = create_evaluation_prompt(script, repo_name, is_agent=self.is_agent, exploration_commands=repo_exploration)
            else:
                raise ValueError(f"Invalid version: {self.version}")
            prompts.append(prompt)
            if self.is_agent:
                bash_tool = create_tool(repo_name)
                agent = create_react_agent(
                    model=self.llm,
                    tools=[bash_tool],
                    post_model_hook=limit_turns
                )
                chain = {"messages": lambda x: [HumanMessage(content=x)]} | agent
                # Wrap with rate limiting and retry logic
                # Pass a lambda that creates a fresh coroutine on each retry attempt
                tasks.append(self._rate_limited_call_with_retry(lambda p=prompt: chain.ainvoke(p)))
            else:
                # Wrap with rate limiting and retry logic
                # Pass a lambda that creates a fresh coroutine on each retry attempt
                tasks.append(self._rate_limited_call_with_retry(lambda p=prompt: self.llm.ainvoke(p)))

        # Execute all tasks concurrently with exception handling and rate limiting
        print(f"[LLMJudgeBatch] Waiting for {len(tasks)} evaluation tasks to complete...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time
        print(f"[LLMJudgeBatch] Completed all evaluations in {elapsed:.1f}s ({elapsed/len(scripts):.2f}s per script)")
        print(f"[LLMJudgeBatch] Parsing {len(results)} results...")
        
        
        # Parse the raw outputs manually
        predictions = []
        for i, result in enumerate(results):
            try:
                if isinstance(result, Exception):
                    # Handle exception case
                    print(f"[LLMJudgeBatch] Error in LLM call for script {i+1}: {result}")
                    predictions.append({
                        'exit_code': -999,
                        'issues_count': -999,
                        'reasoning': f"Error in LLM call: {result}"
                    })
                    continue
                
                # Extract content from result
                if hasattr(result, 'content'):
                    content = str(result.content)
                elif isinstance(result, dict) and 'messages' in result:
                    # Handle case where result is dict with messages key
                    messages = result['messages']
                    if messages:
                        content = str(messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1]))
                    else:
                        content = ""
                else:
                    content = str(result)
                
                # Manual parsing of JSON content
                import json
                import re
                
                # Try to extract JSON from the content using regex
                json_match = re.search(r'\{[^{}]*"exit_code"[^{}]*"issues_count"[^{}]*\}', content)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        parsed = json.loads(json_str)
                        exit_code = parsed.get('exit_code', -999)
                        issues_count = parsed.get('issues_count', -999)
                    except (json.JSONDecodeError, AttributeError):
                        # Fallback to number extraction
                        exit_code = -999
                        issues_count = -999
                        numbers = re.findall(r'\d+', content)
                        if len(numbers) >= 2:
                            exit_code = int(numbers[0])
                            issues_count = int(numbers[1])
                else:
                    # No JSON found, try to extract numbers from text
                    exit_code = -999
                    issues_count = -999
                    numbers = re.findall(r'\d+', content)
                    if len(numbers) >= 2:
                        exit_code = int(numbers[0])
                        issues_count = int(numbers[1])
                
                predictions.append({
                    'exit_code': exit_code,
                    'issues_count': issues_count,
                    'reasoning': content
                })
                
                    
            except Exception as e:
                print(f"[LLMJudgeBatch] Error parsing individual result: {e}")
                predictions.append({
                    'exit_code': -999,
                    'issues_count': -999,
                    'reasoning': f"Error parsing result: {e}"
                })

        # Summary statistics
        success_count = sum(1 for p in predictions if p['exit_code'] != -999)
        error_count = len(predictions) - success_count
        print(f"[LLMJudgeBatch] Parsing complete: {success_count} successful, {error_count} errors")
        print(f"[LLMJudgeBatch] Returning {len(predictions)} predictions")

        return predictions


async def evaluate_scripts_with_llm_async(df: pd.DataFrame, max_samples: Optional[int] = None, debug: bool = False,
                                        model: str = "gpt-4o", temperature: float = 0.0) -> pd.DataFrame:
    """
    Evaluate all scripts in the DataFrame using the LLM judge with async batch processing.
    
    Args:
        df: DataFrame with scripts and ground truth
        max_samples: Maximum number of samples to evaluate (for testing)
        debug: Whether to show debug information including raw LLM outputs
        model: LLM model to use for evaluation
        temperature: Temperature for LLM generation
        
    Returns:
        DataFrame with LLM predictions added
    """
    
    # Limit samples if specified
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"[evaluate_scripts_with_llm_async] Limited evaluation to {max_samples} samples")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print(f"[evaluate_scripts_with_llm_async] OPENAI_API_KEY not set. Cannot run LLM evaluation.")
        return df
    
    # Initialize LLM judge
    judge = LLMJudgeBatch(model=model, temperature=temperature, debug=debug)
    
    # Add prediction columns
    df['llm_exit_code'] = None
    df['llm_issues_count'] = None
    df['llm_reward'] = None
    df['llm_reasoning'] = None
    
    # Get all scripts and repo names
    scripts = df['script'].tolist()
    repo_names = df['repository'].tolist()
    
    print(f"[evaluate_scripts_with_llm_async] Evaluating {len(df)} scripts in a single batch")
    
    # Evaluate all scripts asynchronously in one batch
    predictions = await judge.evaluate_batch_async(scripts, repo_names)
    
    # Store predictions with progress tracking
    for i, prediction in enumerate(predictions):
        df.at[i, 'llm_exit_code'] = prediction['exit_code']
        df.at[i, 'llm_issues_count'] = prediction['issues_count']
        df.at[i, 'llm_reward'] = max(1 - prediction['issues_count']/100, 0) if prediction['exit_code'] == 0 else 0
        df.at[i, 'llm_reasoning'] = prediction['reasoning']
    
    print("[evaluate_scripts_with_llm_async] LLM evaluation completed")
    return df
