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

# Tokencost: Accurate LLM Cost Calculation for Your AI Applications

**Easily estimate the cost of your Large Language Model (LLM) API calls with Tokencost, ensuring budget optimization for your AI projects.**  Check out the original repository [here](https://github.com/AgentOps-AI/tokencost).

**Key Features:**

*   **Precise Token Counting:** Accurately calculate prompt and completion tokens using the official Tiktoken tokenizer and the Anthropic beta token counting API, ensuring the proper cost of your requests.
*   **LLM Price Tracking:** Stay up-to-date with the latest pricing for major LLM providers, including OpenAI, Anthropic, and more.  Includes pricing for LLMs from Google, Meta, Mistral AI, and many more providers.
*   **Simple Integration:** Get cost estimates for prompts and completions with a single function call, simplifying your AI development workflow.

##  Installation

Install Tokencost using pip:

```bash
pip install tokencost
```

##  Usage

Calculate the cost of LLM prompts and completions easily in your Python code:

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

**Calculate cost using string prompts instead of messages:**

```python
from tokencost import calculate_prompt_cost

prompt_string = "Hello world"
response = "How may I assist you today?"
model= "gpt-3.5-turbo"

prompt_cost = calculate_prompt_cost(prompt_string, model)
print(f"Cost: ${prompt_cost}")
# Cost: $3e-06
```

**Counting tokens**

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

## How Tokens Are Counted

Tokencost utilizes OpenAI's [Tiktoken](https://github.com/openai/tiktoken) library for tokenization, providing accurate token counts for most models. For Anthropic models above version 3 (i.e. Sonnet 3.5, Haiku 3.5, and Opus 3), the [Anthropic beta token counting API](https://docs.anthropic.com/claude/docs/beta-api-for-counting-tokens) is used for the most accurate count.

## Pricing Table

Find a comprehensive table outlining the cost of various models:

*   See [pricing_table.md](pricing_table.md)