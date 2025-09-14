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

# Tokencost: Calculate LLM Costs and Token Counts for AI Applications

**Accurately estimate the cost of your Large Language Model (LLM) API calls with Tokencost, your go-to Python library for token counting and pricing.**  [Check out the original repo](https://github.com/AgentOps-AI/tokencost)

**Key Features:**

*   ✅ **Accurate Token Counting:** Utilizes Tiktoken, OpenAI's official tokenizer, and Anthropic's token counting API for precise token calculations.
*   ✅ **Real-Time Pricing:** Stay up-to-date with the latest pricing from major LLM providers.
*   ✅ **Simplified Integration:** Easily calculate the cost of prompts and completions with simple functions.
*   ✅ **Supports Various Models:** Works seamlessly with OpenAI and Anthropic models, with comprehensive pricing data for many other models.

## Core Functionality

*   **Calculate Prompt and Completion Costs:**
    ```python
    from tokencost import calculate_prompt_cost, calculate_completion_cost

    model = "gpt-3.5-turbo"
    prompt = [{"role": "user", "content": "Hello world"}]
    completion = "How may I assist you today?"

    prompt_cost = calculate_prompt_cost(prompt, model)
    completion_cost = calculate_completion_cost(completion, model)

    print(f"{prompt_cost} + {completion_cost} = {prompt_cost + completion_cost}")
    # Example Output: 0.0000135 + 0.000014 = 0.0000275
    ```
*   **Calculate Cost using string prompts instead of message lists:**
    ```python
    from tokencost import calculate_prompt_cost

    prompt_string = "Hello world" 
    response = "How may I assist you today?"
    model= "gpt-3.5-turbo"

    prompt_cost = calculate_prompt_cost(prompt_string, model)
    print(f"Cost: ${prompt_cost}")
    # Cost: $3e-06
    ```
*   **Count Tokens:**
    ```python
    from tokencost import count_message_tokens, count_string_tokens

    message_prompt = [{"role": "user", "content": "Hello world"}]
    # Counting tokens in prompts formatted as message lists
    print(count_message_tokens(message_prompt, model="gpt-3.5-turbo"))
    # 9

    # Alternatively, counting tokens in string prompts
    print(count_string_tokens(prompt="Hello world", model="gpt-3.5-turbo"))
    # 2
    ```

## Installation

Install Tokencost using pip:

```bash
pip install tokencost
```

## Token Counting Logic

Tokencost employs Tiktoken for tokenization of strings and ChatML messages, accurately reflecting the token counts used by OpenAI. For Anthropic models above version 3 (e.g., Sonnet 3.5, Haiku 3.5, and Opus 3), the library utilizes the Anthropic beta token counting API to guarantee precise token counts. For older Claude models, an approximation is calculated using Tiktoken with cl100k\_base encoding.

## LLM Pricing Table

Detailed pricing information is available [here](pricing_table.md). This table includes:

*   Model Name
*   Prompt Cost (USD) per 1M tokens
*   Completion Cost (USD) per 1M tokens
*   Maximum Prompt Tokens
*   Maximum Output Tokens

*(Pricing table will be here - included a placeholder)*

## Further exploration
  Building AI agents? Check out [AgentOps](https://agentops.ai/?tokencost)