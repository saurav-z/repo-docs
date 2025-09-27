# Gorilla: Unleash the Power of LLMs with Massive API Access

**Gorilla empowers Large Language Models (LLMs) to interact seamlessly with a vast array of APIs, unlocking unprecedented capabilities.**  [Explore the original repository](https://github.com/ShishirPatil/gorilla).

<div align="center">
  <img src="https://github.com/ShishirPatil/gorilla/blob/gh-pages/assets/img/logo.png" width="50%" height="50%">
</div>

<div align="center">

[![Arxiv](https://img.shields.io/badge/Gorilla_Paper-2305.15334-<COLOR>.svg?style=flat-square)](https://arxiv.org/abs/2305.15334) [![Discord](https://img.shields.io/discord/1111172801899012102?label=Discord&logo=discord&logoColor=green&style=flat-square)](https://discord.gg/grXXvj9Whz) [![Gorilla Website](https://img.shields.io/badge/Website-gorilla.cs.berkeley.edu-blue?style=flat-square)](https://gorilla.cs.berkeley.edu/) [![Gorilla Blog](https://img.shields.io/badge/Blog-gorilla.cs.berkeley.edu/blog.html-blue?style=flat-square)](https://gorilla.cs.berkeley.edu/blog.html) [![Hugging Face](https://img.shields.io/badge/🤗-gorilla--llm-yellow.svg?style=flat-square)](https://huggingface.co/gorilla-llm)

</div>

## Key Features

*   **API Invocation:** Enables LLMs to accurately call and utilize over 1,600 APIs, expanding their functionality.
*   **Reduced Hallucination:** Designed to minimize errors and ensure reliable API interactions.
*   **OpenFunctions:**  Offers a drop-in alternative for function calling, supporting multiple complex data types, parallel execution, and enhanced RESTful API formatting capabilities.
*   **Berkeley Function Calling Leaderboard (BFCL):** Comprehensive evaluation platform for assessing and comparing function-calling capabilities.
*   **Agent Arena:** Compare LLM agents across various models, tools, and frameworks.
*   **GoEx:** A runtime for executing LLM-generated actions with built-in safety measures.
*   **RAFT (Retrieval-Augmented Fine-tuning):** Fine-tuning recipe for domain-specific RAG.
*   **Gorilla CLI:** User-friendly command-line tool for interacting with APIs via natural language.
*   **API Zoo:** A community-maintained repository of API documentation for improved model training.

## Latest Updates

*   **[07/17/2025]** Announcing BFCL V4 Agentic! Focuses on tool-calling in real-world agentic settings, featuring web search with multi-hop reasoning and error recovery, agent memory management, and format sensitivity evaluation.
*   **[10/04/2024]** Introducing the Agent Arena by Gorilla X LMSYS Chatbot Arena! Compare different agents in tasks like search, finance, RAG, and beyond.
*   **[09/21/2024]** Announcing BFCL V3 - Evaluating multi-turn and multi-step function calling capabilities! New state-based evaluation system tests models on handling complex workflows, sequential functions, and service states.
*   **[08/20/2024]** Released BFCL V2 • Live! The Berkeley Function-Calling Leaderboard now features enterprise-contributed data and real-world scenarios.
*   **[04/12/2024]** Excited to release GoEx - a runtime for LLM-generated actions like code, API calls, and more.

## About

Gorilla revolutionizes how LLMs utilize tools by accurately invoking APIs based on natural language queries. This repository provides the tools to run Gorilla models, evaluate results, and contribute to the API ecosystem.  Since the initial release, Gorilla has been embraced by developers worldwide and the project continues to evolve with new tools, evaluations, leaderboards, and community contributions.

## Getting Started

### Quick Start

*   🚀 [Gorilla Colab Demo](https://colab.research.google.com/drive/1y78Zj7xHysX0xMpr9S468HYs12Mj6X1F?usp=sharing): Try the base Gorilla model
*   🌐 [Gorilla Gradio Demo](https://huggingface.co/spaces/gorilla-llm/gorilla-demo/): Interactive web interface
*   🔥 [OpenFunctions Colab Demo](https://colab.research.google.com/drive/1Td3_R5vPael9PnKYHcl-PxmZkZzA9TCo?usp=sharing): Try the latest OpenFunctions model
*   🎯 [OpenFunctions Website Demo](https://gorilla.cs.berkeley.edu/leaderboard.html#api-explorer): Experiment with function calling
*   📊 [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard): Compare function calling capabilities

### Installation Options

1.  **Gorilla CLI**
```bash
pip install gorilla-cli
gorilla generate 100 random characters into a file called test.txt
```
[Learn more about Gorilla CLI →](https://github.com/gorilla-llm/gorilla-cli)

2.  **Run Gorilla Locally**
```bash
git clone https://github.com/ShishirPatil/gorilla.git
cd gorilla/inference
```
[Detailed local setup instructions →](/gorilla/inference/README.md)

3.  **Use OpenFunctions**
```python
import openai

openai.api_key = "EMPTY"
openai.api_base = "http://luigi.millennium.berkeley.edu:8000/v1"

# Define your functions
functions = [{
    "name": "get_current_weather",
    "description": "Get weather in a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
}]

# Make API call
completion = openai.ChatCompletion.create(
    model="gorilla-openfunctions-v2",
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    functions=functions
)
```
[OpenFunctions documentation →](/openfunctions/README.md)

### Development Tools

*   [GoEx](/goex/README.md): Safe execution of LLM-generated actions
*   [RAFT](/raft/README.md): Fine-tune models for domain-specific tasks
*   [API Store](/data/README.md): Contribute and use APIs

## Frequently Asked Questions

1.  **Can I use Gorilla commercially?**

    Yes! Apache 2.0 licensed models are available for commercial use without obligations.

2.  **Can I use Gorilla with tools like Langchain?**

    Absolutely! Gorilla is designed to work with agentic frameworks and tools like Langchain.

## Project Roadmap

*   [ ] Multimodal function-calling leaderboard
*   [ ] Agentic function-calling leaderboard
*   [ ] New batch of user contributed live function calling evals.
*   [ ] BFCL metrics to evaluate contamination
*   [ ] Openfunctions-v3 model to support more languages and multi-turn capability

## License

Gorilla is Apache 2.0 licensed, permitting both academic and commercial applications.

## Contact

*   💬 Join our [Discord Community](https://discord.gg/grXXvj9Whz)
*   🐦 Follow us on [X](https://x.com/shishirpatil_)

## Citation

```text
@article{patil2023gorilla,
  title={Gorilla: Large Language Model Connected with Massive APIs},
  author={Shishir G. Patil and Tianjun Zhang and Xin Wang and Joseph E. Gonzalez},
  year={2023},
  journal={arXiv preprint arXiv:2305.15334},
}
```