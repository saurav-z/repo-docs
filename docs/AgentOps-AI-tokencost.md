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

# Tokencost: Accurate LLM Cost Estimation & Token Counting

**Calculate the costs of your LLM prompts and completions with precision using Tokencost, and optimize your AI applications.** [Check out the original repo](https://github.com/AgentOps-AI/tokencost).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Version](https://img.shields.io/pypi/v/tokencost)](https://pypi.org/project/tokencost/)
[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/AgentOpsAI)](https://x.com/agentopsai)

Tokencost provides client-side token counting and cost estimation for various Large Language Model (LLM) APIs, giving you the power to accurately track and manage your LLM usage expenses.

### Key Features

*   **Precise Token Counting:** Accurately count tokens for prompts and completions using Tiktoken, OpenAI's official tokenizer, and Anthropic beta token counting API.
*   **LLM Price Tracking:** Stay up-to-date with the latest pricing for major LLM providers.
*   **Easy Integration:** Calculate prompt and completion costs with a single function call.
*   **Comprehensive Model Support:** Supports a wide array of LLM models and their costs.

### Example Usage

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

## Installation

#### Recommended: [PyPI](https://pypi.org/project/tokencost/):

```bash
pip install tokencost
```

## Usage

### Cost Estimates

```python
from openai import OpenAI

client = OpenAI()
model = "gpt-3.5-turbo"
prompt = [{ "role": "user", "content": "Say this is a test"}]

chat_completion = client.chat.completions.create(
    messages=prompt, model=model
)

completion = chat_completion.choices[0].message.content
# "This is a test."

prompt_cost = calculate_prompt_cost(prompt, model)
completion_cost = calculate_completion_cost(completion, model)
print(f"{prompt_cost} + {completion_cost} = {prompt_cost + completion_cost}")
# 0.0000180 + 0.000010 = 0.0000280
```

**Calculating cost using string prompts:**

```python
from tokencost import calculate_prompt_cost

prompt_string = "Hello world"
response = "How may I assist you today?"
model= "gpt-3.5-turbo"

prompt_cost = calculate_prompt_cost(prompt_string, model)
print(f"Cost: ${prompt_cost}")
# Cost: $3e-06
```

### Counting Tokens

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

Tokencost uses [Tiktoken](https://github.com/openai/tiktoken), OpenAI's official tokenizer, for tokenization. Tiktoken splits text into tokens and handles both raw strings and message formats, incorporating tokens for message formatting and roles. For Anthropic models above version 3 (i.e. Sonnet 3.5, Haiku 3.5, and Opus 3), it uses the [Anthropic beta token counting API](https://docs.anthropic.com/claude/docs/beta-api-for-counting-tokens) for accurate token counts. Older Claude models are approximated using Tiktoken.

## Cost Table

Units are denominated in USD. All prices can be located [here](pricing_table.md).

```
<!-- PRICING_TABLE_START -->

| Model Name                             | Prompt Cost (USD) per 1M tokens | Completion Cost (USD) per 1M tokens | Max Prompt Tokens |   Max Output Tokens |
|:---------------------------------------|:--------------------------------|:------------------------------------|:------------------|--------------------:|
| gpt-4                                  | $30                             | $60                                 | 8192              |      4096           |
| gpt-4o                                 | $2.5                            | $10                                 | 128,000           |     16384           |
| ... (Rest of the table is truncated for brevity) ...
```