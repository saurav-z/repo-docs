<p align="center">
  <img src="https://raw.githubusercontent.com/AgentOps-AI/tokencost/main/tokencost.png" height="300" alt="Tokencost" />
</p>

<p align="center">
  <em>Clientside token counting + price estimation for LLM apps and AI agents.</em>
</p>
<p align="center">
    <a href="https://pypi.org/project/tokencost/" target="_blank">
        <img alt="Python" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
        <img alt="Version" src="https://img.shields.io/pypi/v/tokencost?style=for-the-badge&color=3670A0">
    </a>
</p>
<p align="center">
<a href="https://twitter.com/agentopsai/">🐦 Twitter</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://discord.com/invite/FagdcwwXRR">📢 Discord</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://agentops.ai/?tokencost">🖇️ AgentOps</a>
</p>

# tokencost: Effortlessly Calculate LLM Costs

**Quickly and accurately estimate the cost of your Large Language Model (LLM) interactions with `tokencost`.**  Calculate the USD cost of using major LLM providers' APIs by calculating the estimated cost of prompts and completions.

🔗 [View the Code on GitHub](https://github.com/AgentOps-AI/tokencost)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Version](https://img.shields.io/pypi/v/tokencost)](https://pypi.org/project/tokencost/)
[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/AgentOpsAI)](https://x.com/agentopsai)

## Key Features

*   **Accurate Token Counting:** Precisely count prompt and completion tokens using OpenAI's Tiktoken and the Anthropic token counting API.
*   **Real-Time LLM Price Tracking:**  Stay updated on the latest pricing for major LLM providers, including OpenAI, Anthropic, and more.
*   **Easy Integration:** Estimate costs with a single function call.

## Installation

Install `tokencost` using pip:

```bash
pip install tokencost
```

## Usage

Calculate prompt and completion costs and count tokens with ease.

### Cost Estimation

```python
from tokencost import calculate_prompt_cost, calculate_completion_cost

model = "gpt-3.5-turbo"
prompt = [{ "role": "user", "content": "Hello world"}]
completion = "How may I assist you today?"

prompt_cost = calculate_prompt_cost(prompt, model)
completion_cost = calculate_completion_cost(completion, model)

print(f"{prompt_cost} + {completion_cost} = {prompt_cost + completion_cost}")
# 0.0000135 + 0.000014 = 0.0000275
```

### Using string prompts instead of messages:

```python
from tokencost import calculate_prompt_cost

prompt_string = "Hello world"
response = "How may I assist you today?"
model= "gpt-3.5-turbo"

prompt_cost = calculate_prompt_cost(prompt_string, model)
print(f"Cost: ${prompt_cost}")
# Cost: $3e-06
```

### Token Counting

```python
from tokencost import count_message_tokens, count_string_tokens

message_prompt = [{ "role": "user", "content": "Hello world"}]
# Counting tokens in prompts formatted as message lists
print(count_message_tokens(message_prompt, model="gpt-3.5-turbo"))
# 9

# Alternatively, counting tokens in string prompts
print(count_string_tokens(prompt="Hello world", model="gpt-3.5-turbo"))
# 2
```

## How Tokens are Counted

`tokencost` utilizes [Tiktoken](https://github.com/openai/tiktoken), OpenAI's official tokenizer, to count tokens for most models. For Anthropic models above version 3 (i.e. Sonnet 3.5, Haiku 3.5, and Opus 3), we use the [Anthropic beta token counting API](https://docs.anthropic.com/claude/docs/beta-api-for-counting-tokens) for more accurate token counts.

## Current Pricing Information

Refer to the [pricing_table.md](pricing_table.md) file for the most up-to-date pricing information.