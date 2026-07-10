import asyncio
import sys
import os

from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ.get("NVIDIA_API_KEY", "")

from nemoguardrails import LLMRails, RailsConfig
from guardrails.actions import (
    is_guardrails_block,
    check_content_safety,
    check_for_safety_violations,
    check_logical_consistency,
    detect_hallucinations,
    check_output_completeness
)

async def test_nemo():
    print("Loading config...")
    config = RailsConfig.from_path("guardrails")
    rails = LLMRails(config)
    
    print("Registering actions...")
    rails.register_action(is_guardrails_block, "is_guardrails_block")
    rails.register_action(check_content_safety, "check_content_safety")
    rails.register_action(check_for_safety_violations, "check_for_safety_violations")
    rails.register_action(check_logical_consistency, "check_logical_consistency")
    rails.register_action(detect_hallucinations, "detect_hallucinations")
    rails.register_action(check_output_completeness, "check_output_completeness")
    
    print("Running generate_async for a safe message...")
    resp_safe = await rails.generate_async(messages=[{"role": "user", "content": "This is a completely safe report. Everything is functioning normally and there are absolutely no issues with the motor or the sensor data."}])
    print("Safe Response:", resp_safe)

    print("Running generate_async for a hallucinated output...")
    # Because we evaluate everything on the input flow, we pass it as a user message
    resp_bad = await rails.generate_async(messages=[
        {"role": "user", "content": "I am 100% certainty that the motor is broken."}
    ])
    print("Bad Response:", resp_bad)

if __name__ == "__main__":
    asyncio.run(test_nemo())
