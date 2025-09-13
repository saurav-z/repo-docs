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

# Tokencost: Effortlessly Calculate LLM Costs in Your AI Applications

**Tokencost** is a Python library that simplifies the process of estimating the cost of using Large Language Models (LLMs) like GPT-4, Claude, and others by accurately calculating prompt and completion costs.  [Check out the original repo](https://github.com/AgentOps-AI/tokencost) for the latest updates.

**Key Features:**

*   **Real-time LLM Price Tracking:**  Keeps you up-to-date with the latest pricing from major LLM providers, which often change.
*   **Precise Token Counting:** Accurately counts tokens in your prompts and completions using OpenAI's Tiktoken and other methods, allowing for accurate cost estimation.
*   **Simple Integration:** Easily determine the cost of your prompts and completions with a single function call.
*   **Supports Many Models:** Compatible with a wide range of models from OpenAI, Anthropic, and more.

### **Getting Started with Tokencost**

**Installation**

Install Tokencost using pip:

```bash
pip install tokencost
```

**Usage**

*   **Calculate Prompt and Completion Costs:**

```python
from tokencost import calculate_prompt_cost, calculate_completion_cost

model = "gpt-3.5-turbo"
prompt = [{ "role": "user", "content": "Hello world"}]
completion = "How may I assist you today?"

prompt_cost = calculate_prompt_cost(prompt, model)
completion_cost = calculate_completion_cost(completion, model)

print(f"Prompt Cost: {prompt_cost} + Completion Cost: {completion_cost} = Total Cost: {prompt_cost + completion_cost}")
# Example Output: Prompt Cost: 0.0000135 + Completion Cost: 0.000014 = Total Cost: 0.0000275
```

*   **Using string prompts:**

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

message_prompt = [{ "role": "user", "content": "Hello world"}]
model = "gpt-3.5-turbo"

print(count_message_tokens(message_prompt, model))
# 9

print(count_string_tokens(prompt="Hello world", model="gpt-3.5-turbo"))
# 2
```

### How Tokens are Counted

This library leverages [Tiktoken](https://github.com/openai/tiktoken), OpenAI's official tokenizer, to break down text into tokens, taking into account message formatting and roles for more accurate calculations.

For Anthropic models above version 3 (i.e. Sonnet 3.5, Haiku 3.5, and Opus 3), the [Anthropic beta token counting API](https://docs.anthropic.com/claude/docs/beta-api-for-counting-tokens) is used for accurate token counts. For older Claude models, Tokencost approximates using Tiktoken with the cl100k_base encoding.

###  LLM Pricing Table

Find the latest LLM pricing information in USD below.

*Note: Pricing data sourced from model provider documentation and subject to change.*

| Model Name                                                            | Prompt Cost (USD) per 1M tokens   | Completion Cost (USD) per 1M tokens   | Max Prompt Tokens   |   Max Output Tokens |
|:----------------------------------------------------------------------|:----------------------------------|:--------------------------------------|:--------------------|--------------------:|
| gpt-4                                                                 | $30                               | $60                                   | 8192                |      4096           |
| gpt-4o                                                                | $2.5                              | $10                                   | 128,000             |     16384           |
| gpt-4o-audio-preview                                                  | $2.5                              | $10                                   | 128,000             |     16384           |
| gpt-4o-audio-preview-2024-10-01                                       | $2.5                              | $10                                   | 128,000             |     16384           |
| gpt-4o-mini                                                           | $0.15                             | $0.6                                  | 128,000             |     16384           |
| gpt-4o-mini-2024-07-18                                                | $0.15                             | $0.6                                  | 128,000             |     16384           |
| o1-mini                                                               | $1.1                              | $4.4                                  | 128,000             |     65536           |
| o1-mini-2024-09-12                                                    | $3                                | $12                                   | 128,000             |     65536           |
| o1-preview                                                            | $15                               | $60                                   | 128,000             |     32768           |
| o1-preview-2024-09-12                                                 | $15                               | $60                                   | 128,000             |     32768           |
| chatgpt-4o-latest                                                     | $5                                | $15                                   | 128,000             |      4096           |
| gpt-4o-2024-05-13                                                     | $5                                | $15                                   | 128,000             |      4096           |
| gpt-4o-2024-08-06                                                     | $2.5                              | $10                                   | 128,000             |     16384           |
| gpt-4-turbo-preview                                                   | $10                               | $30                                   | 128,000             |      4096           |
| gpt-4-0314                                                            | $30                               | $60                                   | 8,192               |      4096           |
| gpt-4-0613                                                            | $30                               | $60                                   | 8,192               |      4096           |
| gpt-4-32k                                                             | $60                               | $120                                  | 32,768              |      4096           |
| gpt-4-32k-0314                                                        | $60                               | $120                                  | 32,768              |      4096           |
| gpt-4-32k-0613                                                        | $60                               | $120                                  | 32,768              |      4096           |
| gpt-4-turbo                                                           | $10                               | $30                                   | 128,000             |      4096           |
| gpt-4-turbo-2024-04-09                                                | $10                               | $30                                   | 128,000             |      4096           |
| gpt-4-1106-preview                                                    | $10                               | $30                                   | 128,000             |      4096           |
| gpt-4-0125-preview                                                    | $10                               | $30                                   | 128,000             |      4096           |
| gpt-4-vision-preview                                                  | $10                               | $30                                   | 128,000             |      4096           |
| gpt-4-1106-vision-preview                                             | $10                               | $30                                   | 128,000             |      4096           |
| gpt-3.5-turbo                                                         | $1.5                              | $2                                    | 16,385              |      4096           |
| gpt-3.5-turbo-0301                                                    | $1.5                              | $2                                    | 4,097               |      4096           |
| gpt-3.5-turbo-0613                                                    | $1.5                              | $2                                    | 4,097               |      4096           |
| gpt-3.5-turbo-1106                                                    | $1                                | $2                                    | 16,385              |      4096           |
| gpt-3.5-turbo-0125                                                    | $0.5                              | $1.5                                  | 16,385              |      4096           |
| gpt-3.5-turbo-16k                                                     | $3                                | $4                                    | 16,385              |      4096           |
| gpt-3.5-turbo-16k-0613                                                | $3                                | $4                                    | 16,385              |      4096           |
| ft:gpt-3.5-turbo                                                      | $3                                | $6                                    | 16,385              |      4096           |
| ft:gpt-3.5-turbo-0125                                                 | $3                                | $6                                    | 16,385              |      4096           |
| ft:gpt-3.5-turbo-1106                                                 | $3                                | $6                                    | 16,385              |      4096           |
| ft:gpt-3.5-turbo-0613                                                 | $3                                | $6                                    | 4,096               |      4096           |
| ft:gpt-4-0613                                                         | $30                               | $60                                   | 8,192               |      4096           |
| ft:gpt-4o-2024-08-06                                                  | $3.75                             | $15                                   | 128,000             |     16384           |
| ft:gpt-4o-mini-2024-07-18                                             | $0.3                              | $1.2                                  | 128,000             |     16384           |
| ft:davinci-002                                                        | $2                                | $2                                    | 16,384              |      4096           |
| ft:babbage-002                                                        | $0.4                              | $0.4                                  | 16,384              |      4096           |
| text-embedding-3-large                                                | $0.13                             | $0                                    | 8,191               |       nan           |
| text-embedding-3-small                                                | $0.02                             | $0                                    | 8,191               |       nan           |
| text-embedding-ada-002                                                | $0.1                              | $0                                    | 8,191               |       nan           |
| text-embedding-ada-002-v2                                             | $0.1                              | $0                                    | 8,191               |       nan           |
| text-moderation-stable                                                | $0                                | $0                                    | 32,768              |         0           |
| text-moderation-007                                                   | $0                                | $0                                    | 32,768              |         0           |
| text-moderation-latest                                                | $0                                | $0                                    | 32,768              |         0           |
| 256-x-256/dall-e-2                                                    | --                                | --                                    | nan                 |       nan           |
| 512-x-512/dall-e-2                                                    | --                                | --                                    | nan                 |       nan           |
| 1024-x-1024/dall-e-2                                                  | --                                | --                                    | nan                 |       nan           |
| hd/1024-x-1792/dall-e-3                                               | --                                | --                                    | nan                 |       nan           |
| hd/1792-x-1024/dall-e-3                                               | --                                | --                                    | nan                 |       nan           |
| hd/1024-x-1024/dall-e-3                                               | --                                | --                                    | nan                 |       nan           |
| standard/1024-x-1792/dall-e-3                                         | --                                | --                                    | nan                 |       nan           |
| standard/1792-x-1024/dall-e-3                                         | --                                | --                                    | nan                 |       nan           |
| standard/1024-x-1024/dall-e-3                                         | --                                | --                                    | nan                 |       nan           |
| whisper-1                                                             | --                                | --                                    | nan                 |       nan           |
| tts-1                                                                 | --                                | --                                    | nan                 |       nan           |
| tts-1-hd                                                              | --                                | --                                    | nan                 |       nan           |
| azure/tts-1                                                           | --                                | --                                    | nan                 |       nan           |
| azure/tts-1-hd                                                        | --                                | --                                    | nan                 |       nan           |
| azure/whisper-1                                                       | --                                | --                                    | nan                 |       nan           |
| azure/o1-mini                                                         | $1.21                             | $4.84                                 | 128,000             |     65536           |
| azure/o1-mini-2024-09-12                                              | $1.1                              | $4.4                                  | 128,000             |     65536           |
| azure/o1-preview                                                      | $15                               | $60                                   | 128,000             |     32768           |
| azure/o1-preview-2024-09-12                                           | $15                               | $60                                   | 128,000             |     32768           |
| azure/gpt-4o                                                          | $2.5                              | $10                                   | 128,000             |     16384           |
| azure/gpt-4o-2024-08-06                                               | $2.5                              | $10                                   | 128,000             |     16384           |
| azure/gpt-4o-2024-05-13                                               | $5                                | $15                                   | 128,000             |      4096           |
| azure/global-standard/gpt-4o-2024-08-06                               | $2.5                              | $10                                   | 128,000             |     16384           |
| azure/global-standard/gpt-4o-mini                                     | $0.15                             | $0.6                                  | 128,000             |     16384           |
| azure/gpt-4o-mini                                                     | $0.16                             | $0.66                                 | 128,000             |     16384           |
| azure/gpt-4-turbo-2024-04-09                                          | $10                               | $30                                   | 128,000             |      4096           |
| azure/gpt-4-0125-preview                                              | $10                               | $30                                   | 128,000             |      4096           |
| azure/gpt-4-1106-preview                                              | $10                               | $30                                   | 128,000             |      4096           |
| azure/gpt-4-0613                                                      | $30                               | $60                                   | 8,192               |      4096           |
| azure/gpt-4-32k-0613                                                  | $60                               | $120                                  | 32,768              |      4096           |
| azure/gpt-4-32k                                                       | $60                               | $120                                  | 32,768              |      4096           |
| azure/gpt-4                                                           | $30                               | $60                                   | 8,192               |      4096           |
| azure/gpt-4-turbo                                                     | $10                               | $30                                   | 128,000             |      4096           |
| azure/gpt-4-turbo-vision-preview                                      | $10                               | $30                                   | 128,000             |      4096           |
| azure/gpt-35-turbo-16k-0613                                           | $3                                | $4                                    | 16,385              |      4096           |
| azure/gpt-35-turbo-1106                                               | $1                                | $2                                    | 16,384              |      4096           |
| azure/gpt-35-turbo-0613                                               | $1.5                              | $2                                    | 4,097               |      4096           |
| azure/gpt-35-turbo-0301                                               | $0.2                              | $2                                    | 4,097               |      4096           |
| azure/gpt-35-turbo-0125                                               | $0.5                              | $1.5                                  | 16,384              |      4096           |
| azure/gpt-35-turbo-16k                                                | $3                                | $4                                    | 16,385              |      4096           |
| azure/gpt-35-turbo                                                    | $0.5                              | $1.5                                  | 4,097               |      4096           |
| azure/gpt-3.5-turbo-instruct-0914                                     | $1.5                              | $2                                    | 4,097               |       nan           |
| azure/gpt-35-turbo-instruct                                           | $1.5                              | $2                                    | 4,097               |       nan           |
| azure/gpt-35-turbo-instruct-0914                                      | $1.5                              | $2                                    | 4,097               |       nan           |
| azure/mistral-large-latest                                            | $8                                | $24                                   | 32,000              |       nan           |
| azure/mistral-large-2402                                              | $8                                | $24                                   | 32,000              |       nan           |
| azure/command-r-plus                                                  | $3                                | $15                                   | 128,000             |      4096           |
| azure/ada                                                             | $0.1                              | $0                                    | 8,191               |       nan           |
| azure/text-embedding-ada-002                                          | $0.1                              | $0                                    | 8,191               |       nan           |
| azure/text-embedding-3-large                                          | $0.13                             | $0                                    | 8,191               |       nan           |
| azure/text-embedding-3-small                                          | $0.02                             | $0                                    | 8,191               |       nan           |
| azure/standard/1024-x-1024/dall-e-3                                   | --                                | $0                                    | nan                 |       nan           |
| azure/hd/1024-x-1024/dall-e-3                                         | --                                | $0                                    | nan                 |       nan           |
| azure/standard/1024-x-1792/dall-e-3                                   | --                                | $0                                    | nan                 |       nan           |
| azure/standard/1792-x-1024/dall-e-3                                   | --                                | $0                                    | nan                 |       nan           |
| azure/hd/1024-x-1792/dall-e-3                                         | --                                | $0                                    | nan                 |       nan           |
| azure/hd/1792-x-1024/dall-e-3                                         | --                                | $0                                    | nan                 |       nan           |
| azure/standard/1024-x-1024/dall-e-2                                   | --                                | $0                                    | nan                 |       nan           |
| azure_ai/jamba-instruct                                               | $0.5                              | $0.7                                  | 70,000              |      4096           |
| azure_ai/mistral-large                                                | $4                                | $12                                   | 32,000              |      8191           |
| azure_ai/mistral-small                                                | $1                                | $3                                    | 32,000              |      8191           |
| azure_ai/Meta-Llama-3-70B-Instruct                                    | $1.1                              | $0.37                                 | 8,192               |      2048           |
| azure_ai/Meta-Llama-3.1-8B-Instruct                                   | $0.3                              | $0.61                                 | 128,000             |      2048           |
| azure_ai/Meta-Llama-3.1-70B-Instruct                                  | $2.68                             | $3.54                                 | 128,000             |      2048           |
| azure_ai/Meta-Llama-3.1-405B-Instruct                                 | $5.33                             | $16                                   | 128,000             |      2048           |
| azure_ai/cohere-rerank-v3-multilingual                                | $0                                | $0                                    | 4,096               |      4096           |
| azure_ai/cohere-rerank-v3-english                                     | $0                                | $0                                    | 4,096               |      4096           |
| azure_ai/Cohere-embed-v3-english                                      | $0.1                              | $0                                    | 512                 |       nan           |
| azure_ai/Cohere-embed-v3-multilingual                                 | $0.1                              | $0                                    | 512                 |       nan           |
| babbage-002                                                           | $0.4                              | $0.4                                  | 16,384              |      4096           |
| davinci-002                                                           | $2                                | $2                                    | 16,384              |      4096           |
| gpt-3.5-turbo-instruct                                                | $1.5                              | $2                                    | 8,192               |      4096           |
| gpt-3.5-turbo-instruct-0914                                           | $1.5                              | $2                                    | 8,192               |      4097           |
| claude-instant-1                                                      | $1.63                             | $5.51                                 | 100,000             |      8191           |
| mistral/mistral-tiny                                                  | $0.25                             | $0.25                                 | 32,000              |      8191           |
| mistral/mistral-small                                                 | $0.1                              | $0.3                                  | 32,000              |      8191           |
| mistral/mistral-small-latest                                          | $0.1                              | $0.3                                  | 32,000              |      8191           |
| mistral/mistral-medium                                                | $2.7                              | $8.1                                  | 32,000              |      8191           |
| mistral/mistral-medium-latest                                         | $0.4                              | $2                                    | 131,072             |      8191           |
| mistral/mistral-medium-2312                                           | $2.7                              | $8.1                                  | 32,000              |      8191           |
| mistral/mistral-large-latest                                          | $2                                | $6                                    | 128,000             |    128000           |
| mistral/mistral-large-2402                                            | $4                                | $12                                   | 32,000              |      8191           |
| mistral/mistral-large-2407                                            | $3                                | $9                                    | 128,000             |    128000           |
| mistral/pixtral-12b-2409                                              | $0.15                             | $0.15                                 | 128,000             |    128000           |
| mistral/open-mistral-7b                                               | $0.25                             | $0.25                                 | 32,000              |      8191           |
| mistral/open-mixtral-8x7b                                             | $0.7                              | $0.7                                  | 32,000              |      8191           |
| mistral/open-mixtral-8x22b                                            | $2                                | $6                                    | 65,336              |      8191           |
| mistral/codestral-latest                                              | $1                                | $3                                    | 32,000              |      8191           |
| mistral/codestral-2405                                                | $1                                | $3                                    | 32,000              |      8191           |
| mistral/open-mistral-nemo                                             | $0.3                              | $0.3                                  | 128,000             |    128000           |
| mistral/open-mistral-nemo-2407                                        | $0.3                              | $0.3                                  | 128,000             |    128000           |
| mistral/open-codestral-mamba                                          | $0.25                             | $0.25                                 | 256,000             |    256000           |
| mistral/codestral-mamba-latest                                        | $0.25                             | $0.25                                 | 256,000             |    256000           |
| mistral/mistral-embed                                                 | $0.1                              | --                                    | 8,192               |       nan           |
| deepseek-chat                                                         | $0.14                             | $0.28                                 | 128,000             |      4096           |
| codestral/codestral-latest                                            | $0                                | $0                                    | 32,000              |      8191           |
| codestral/codestral-2405                                              | $0                                | $0                                    | 32,000              |      8191           |
| text-completion-codestral/codestral-latest                            | $0                                | $0                                    | 32,000              |      8191           |
| text-completion-codestral/codestral-2405                              | $0                                | $0                                    | 32,000              |      8191           |
| deepseek-coder                                                        | $0.14                             | $0.28                                 | 128,000             |      4096           |
| groq/llama2-70b-4096                                                  | $0.7                              | $0.8                                  | 4,096               |      4096           |
| groq/llama3-8b-8192                                                   | $0.05                             | $0.08                                 | 8,192               |      8192           |
| groq/llama3-70b-8192                                                  | $0.59                             | $0.79                                 | 8,192               |      8192           |
| groq/llama-3.1-8b-instant                                             | $0.05                             | $0.08                                 | 128,000             |      8192           |
| groq/llama-3.1-70b-versatile                                          | $0.59                             | $0.79                                 | 8,192               |      8192           |
| groq/llama-3.1-405b-reasoning                                         | $0.59                             | $0.79                                 | 8,192               |      8192           |
| groq/mixtral-8x7b-32768                                               | $0.24                             | $0.24                                 | 32,768              |     32768           |
| groq/gemma-7b-it                                                      | $0.07                             | $0.07                                 | 8,192               |      8192           |
| groq/gemma2-9b-it                                                     | $0.2                              | $0.2                                  | 8,192               |      8192           |
| groq/llama3-groq-70b-8192-tool-use-preview                            | $0.89                             | $0.89                                 | 8,192               |      8192           |
| groq/llama3-groq-8b-8192-tool-use-preview                             | $0.19                             | $0.19                                 | 8,192               |      8192           |
| cerebras/llama3.1-8b                                                  | $0.1                              | $0.1                                  | 128,000             |    128000           |
| cerebras/llama3.1-70b                                                 | $0.6                              | $0.6                                  | 128,000             |    128000           |
| friendliai/mixtral-8x7b-instruct-v0-1                                 | $0.4                              | $0.4                                  | 32,768              |     32768           |
| friendliai/meta-llama-3-8b-instruct                                   | $0.1                              | $0.1                                  | 8,192               |