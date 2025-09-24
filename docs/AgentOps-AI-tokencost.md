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

# Tokencost: Effortlessly Calculate LLM Costs and Manage Token Usage

**Tokencost** is a Python library that simplifies the process of estimating the cost of using Large Language Models (LLMs) by accurately calculating token counts and providing cost estimations.  [Explore the original repository](https://github.com/AgentOps-AI/tokencost) for detailed information.

**Key Features:**

*   **Precise Token Counting:** Accurately count prompt and completion tokens using OpenAI's official Tiktoken tokenizer, and Anthropic's API for Claude models, ensuring reliable cost calculations.
*   **LLM Cost Tracking:**  Stay up-to-date with the latest pricing from major LLM providers.  Tokencost helps you manage your budget by providing real-time cost estimates.
*   **Easy Integration:**  Calculate the cost of your LLM prompts and completions with a single function call, simplifying integration into your AI applications.
*   **Flexible Usage:** Supports both message lists (ChatML) and string-based prompts for maximum flexibility.
*   **Comprehensive Model Support:**  Supports a wide array of LLM models from various providers like OpenAI, Anthropic, and more.

## Installation

Install Tokencost using pip:

```bash
pip install tokencost
```

## Usage

### Calculating Cost Estimates

Here's how to estimate the cost of prompts and completions:

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

### String-Based Prompts

Calculate costs using string prompts instead of message lists:

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

Count tokens in prompts formatted as message lists or strings:

```python
from tokencost import count_message_tokens, count_string_tokens

message_prompt = [{ "role": "user", "content": "Hello world"}]
print(count_message_tokens(message_prompt, model="gpt-3.5-turbo"))
# 9

print(count_string_tokens(prompt="Hello world", model="gpt-3.5-turbo"))
# 2
```

## How Tokens are Counted

Tokencost uses [Tiktoken](https://github.com/openai/tiktoken), OpenAI's official tokenizer, to accurately count tokens for most models.  For Anthropic models above version 3 (i.e. Sonnet 3.5, Haiku 3.5, and Opus 3), the [Anthropic beta token counting API](https://docs.anthropic.com/claude/docs/beta-api-for-counting-tokens) is used for accuracy.  For older Claude models, it approximates using Tiktoken with the cl100k_base encoding.

## Pricing Table

View the latest LLM pricing information in the [pricing\_table.md](pricing_table.md) file.

```