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

# Accurately Calculate LLM Costs with TokenCost 

**Effortlessly estimate the cost of your Large Language Model (LLM) applications with Tokencost, the Python library for precise token counting and pricing.**

[Check out the original repo](https://github.com/AgentOps-AI/tokencost)

**Key Features:**

*   **Precise Token Counting:** Utilize Tiktoken, OpenAI's official tokenizer, for accurate token calculations of prompts and completions.
*   **LLM Price Tracking:** Stay updated with the latest pricing for various LLM providers.
*   **Easy Integration:** Calculate prompt and completion costs with a single function call.

## Installation

Install via PyPI:

```bash
pip install tokencost
```

## Core Functionality

*   **Calculate Cost:** Easily estimate the cost of prompts and completions.
    ```python
    from tokencost import calculate_prompt_cost, calculate_completion_cost

    model = "gpt-3.5-turbo"
    prompt = [{"role": "user", "content": "Hello world"}]
    completion = "How may I assist you today?"

    prompt_cost = calculate_prompt_cost(prompt, model)
    completion_cost = calculate_completion_cost(completion, model)

    print(f"{prompt_cost} + {completion_cost} = {prompt_cost + completion_cost}")
    # 0.0000135 + 0.000014 = 0.0000275
    ```

*   **Calculate Cost for String Prompts:**
    ```python
    from tokencost import calculate_prompt_cost

    prompt_string = "Hello world" 
    response = "How may I assist you today?"
    model= "gpt-3.5-turbo"

    prompt_cost = calculate_prompt_cost(prompt_string, model)
    print(f"Cost: ${prompt_cost}")
    # Cost: $3e-06
    ```

*   **Token Counting:** Easily count tokens in messages and strings.
    ```python
    from tokencost import count_message_tokens, count_string_tokens

    message_prompt = [{"role": "user", "content": "Hello world"}]
    print(count_message_tokens(message_prompt, model="gpt-3.5-turbo"))
    # 9

    print(count_string_tokens(prompt="Hello world", model="gpt-3.5-turbo"))
    # 2
    ```

## How Tokens are Counted

TokenCost uses [Tiktoken](https://github.com/openai/tiktoken), OpenAI's official tokenizer, to tokenize strings and ChatML messages. For Anthropic models above version 3, the [Anthropic beta token counting API](https://docs.anthropic.com/claude/docs/beta-api-for-counting-tokens) is used.

## Pricing Table

Find the latest pricing information [here](pricing_table.md).
```