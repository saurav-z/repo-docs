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

**Tokencost** is your go-to Python library for precisely calculating the cost of using various Large Language Model (LLM) APIs and efficiently counting tokens, enabling you to manage your AI application expenses effectively.  Check out the original repo [here](https://github.com/AgentOps-AI/tokencost).

## Key Features

*   **Real-time LLM Price Tracking:** Stay updated with the latest pricing from major LLM providers.
*   **Precise Token Counting:** Accurately count tokens for prompts and completions using OpenAI's Tiktoken tokenizer and the Anthropic token counting API, for accurate cost estimations.
*   **Easy Integration:** Calculate prompt and completion costs with a single, simple function call.
*   **Supports a wide range of models** including but not limited to OpenAI, Anthropic, Groq, and more.

## Installation

Install `tokencost` easily using pip:

```bash
pip install tokencost
```

## Usage

### Calculating Cost

Quickly estimate the cost of your prompts and completions.

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

### Usage with OpenAI API

Integrate directly with your OpenAI API calls.

```python
from openai import OpenAI
from tokencost import calculate_prompt_cost, calculate_completion_cost

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

### Cost Estimation with String Prompts

Calculate prompt costs using string-based prompts.

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

Use the provided functions to count tokens in prompts.

```python
from tokencost import count_message_tokens, count_string_tokens

model = "gpt-3.5-turbo"
message_prompt = [{ "role": "user", "content": "Hello world"}]
# Counting tokens in prompts formatted as message lists
print(count_message_tokens(message_prompt, model=model))
# 9

# Alternatively, counting tokens in string prompts
print(count_string_tokens(prompt="Hello world", model=model))
# 2
```

## How Tokens are Counted

`Tokencost` utilizes [Tiktoken](https://github.com/openai/tiktoken), OpenAI's official tokenizer, for precise tokenization of strings and ChatML messages. For Anthropic models above version 3 (i.e. Sonnet 3.5, Haiku 3.5, and Opus 3), we use the [Anthropic beta token counting API](https://docs.anthropic.com/claude/docs/beta-api-for-counting-tokens) to ensure accurate token counts. For older Claude models, we approximate using Tiktoken with the cl100k_base encoding.

## Updated Pricing Table

For the most up-to-date pricing details and model specifications, please refer to the comprehensive pricing table in the `pricing_table.md` file.