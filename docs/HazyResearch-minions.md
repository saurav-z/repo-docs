![Minions Logo](assets/Ollama_minionS_background.png)

# Minions: Unleash the Power of On-Device and Cloud LLMs

**Collaborate with on-device and cloud LLMs efficiently using the Minions protocol, reducing cloud costs and maintaining high quality!**

[View the Minions GitHub Repository](https://github.com/HazyResearch/minions) | [Join our Discord](https://discord.gg/jfJyxXwFVa)

Minions is a cutting-edge communication protocol that seamlessly connects small, efficient on-device language models with powerful cloud-based models, allowing for intelligent collaboration.  By intelligently offloading long context processing to local models, Minions minimizes cloud costs while maintaining the performance of leading LLMs. This repository provides a comprehensive demonstration of the Minions protocol, along with tools and applications to get you started.

**Key Features:**

*   **Cost-Effective LLM Collaboration:**  Leverage on-device models to reduce reliance on expensive cloud resources.
*   **Seamless Integration:** Easy-to-use protocol for combining local and cloud models.
*   **Flexible Deployment:**  Supports diverse local model servers (Ollama, Lemonade, Tokasaurus) and cloud providers (OpenAI, Together AI, Azure OpenAI, DeepSeek, Anthropic, Mistral AI).
*   **Secure Communication:**  Includes a secure protocol with end-to-end encryption, attestation, and replay protection.
*   **WebGPU App:** Run the Minions protocol entirely in your browser.
*   **Comprehensive Examples:** Ready-to-use code examples for various applications.
*   **Versatile Applications:**  Includes agent-to-agent integration, character chat, document search, story telling and more.

**Explore Minions:**

*   **Paper:** [Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models](https://arxiv.org/pdf/2502.15964)
*   **Blog Post:** [Minions Introduction](https://hazyresearch.stanford.edu/blog/2025-02-24-minions)
*   **Secure Minions Chat:** [Secure Minions Chat](https://hazyresearch.stanford.edu/blog/2025-05-12-security)
*   **Demo Application:** [Streamlit Demo](#minions-demo-application)
*   **WebGPU App:** [WebGPU App](#minions-webgpu-app)

## Table of Contents

-   [Setup](#setup)
    -   [Step 1: Install the Minions Package](#step-1-clone-and-install)
    -   [Step 2: Choose and Install a Local Model Server](#step-2-install-a-server-for-running-the-local-model)
    -   [Step 3: Configure Cloud LLM API Keys](#step-3-set-your-api-key-for-at-least-one-of-the-following-cloud-llm-providers)
-   [Minions Demo Application](#minions-demo-application)
-   [Minions WebGPU App](#minions-webgpu-app)
-   [Example Code](#example-code-minion-singular)
    -   [Minion (Singular)](#example-code-minion-singular)
    -   [Minions (Plural)](#example-code-minions-plural)
-   [Python Notebook](#python-notebook)
-   [Docker Support](#docker-support)
-   [Command Line Interface (CLI)](#cli)
-   [Secure Minions Local-Remote Protocol](#secure-minions-local-remote-protocol)
-   [Secure Minions Chat](#secure-minions-chat)
-   [Apps](#apps)
-   [Inference Estimator](#inference-estimator)
    -   [Command Line Usage](#command-line-usage)
    -   [Python API Usage](#python-api-usage)
-   [Miscellaneous Setup](#miscellaneous-setup)
    -   [Using Azure OpenAI](#using-azure-openai-with-minions)
-   [Maintainers](#maintainers)

## Setup

*Tested on Mac and Ubuntu with Python 3.10-3.11 (Python 3.13 is not supported)*

**Step 1: Clone and Install**

1.  Clone the repository:

    ```bash
    git clone https://github.com/HazyResearch/minions.git
    cd minions
    ```

2.  Install the Python package in editable mode:

    ```bash
    pip install -e .
    ```

    *   For MLX-LM support: `pip install -e ".[mlx]"`
    *   For Secure Minions Chat: `pip install -e ".[secure]"`

**Step 2: Choose and Install a Local Model Server**

Select and install *one* of the following local model servers:

*   **Ollama:** Recommended if you *do not* have NVIDIA/AMD GPUs. Install from [here](https://ollama.com/download).  To enable Flash Attention, run `launchctl setenv OLLAMA_FLASH_ATTENTION 1` and restart the Ollama app on macOS.
*   **Lemonade:** Use if you have AMD CPUs/GPUs/NPUs.  Install from [here](https://lemonade-server.ai/). *Note: Lemonade does not support the Minion-CUA protocol at this time.*
*   **Tokasaurus:** Designed for NVIDIA GPUs, and benefits from the high-throughput for Minions protocol. Install with:  `pip install tokasaurus`

    *   Optional: Install Cartesia-MLX on Apple Silicon following the instructions in the original README.
    *   Optional: Install llama-cpp-python, following the instructions in the original README.

**Step 3: Configure Cloud LLM API Keys**

Set your API key for *at least one* of the following cloud LLM providers. Create an API key if you don't have one (e.g., [OpenAI](https://platform.openai.com/docs/overview)).

```bash
export OPENAI_API_KEY=<your-openai-api-key>
export OPENAI_BASE_URL=<your-openai-base-url>  # Optional: Use a different OpenAI API endpoint

export TOGETHER_API_KEY=<your-together-api-key>
export OPENROUTER_API_KEY=<your-openrouter-api-key>
export OPENROUTER_BASE_URL=<your-openrouter-base-url>
export PERPLEXITY_API_KEY=<your-perplexity-api-key>
export PERPLEXITY_BASE_URL=<your-perplexity-base-url>
export TOKASAURUS_BASE_URL=<your-tokasaurus-base-url>
export DEEPSEEK_API_KEY=<your-deepseek-api-key>
export ANTHROPIC_API_KEY=<your-anthropic-api-key>
export MISTRAL_API_KEY=<your-mistral-api-key>
```

## Minions Demo Application

Run the interactive demo to see Minions in action:

```bash
pip install torch transformers
streamlit run app.py
```

*   If you encounter an Ollama connection error, try: `OLLAMA_FLASH_ATTENTION=1 ollama serve`

## Minions WebGPU App

Experience Minions in your browser with this WebGPU app:

1.  `cd apps/minions-webgpu`
2.  `npm install`
3.  `npm start`
4.  Open your browser and navigate to the provided URL.

## Example Code: Minion (Singular)

This example uses `ollama` locally and `openai` remotely:

```python
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minion import Minion

local_client = OllamaClient(
        model_name="llama3.2",
    )

remote_client = OpenAIClient(
        model_name="gpt-4o",
    )

minion = Minion(local_client, remote_client)

context = """
Patient John Doe is a 60-year-old male with a history of hypertension...
"""

task = "Based on the patient's blood pressure and LDL cholesterol readings..."

output = minion(task=task, context=[context], max_rounds=2)
```

## Example Code: Minions (Plural)

This example uses `ollama` locally and `openai` remotely with structured output:

```python
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minions import Minions
from pydantic import BaseModel

class StructuredLocalOutput(BaseModel):
    explanation: str
    citation: str | None
    answer: str | None

local_client = OllamaClient(
        model_name="llama3.2",
        temperature=0.0,
        structured_output_schema=StructuredLocalOutput
)

remote_client = OpenAIClient(
        model_name="gpt-4o",
)

minion = Minions(local_client, remote_client)

context = """
Patient John Doe is a 60-year-old male with a history of hypertension...
"""

task = "Based on the patient's blood pressure and LDL cholesterol readings..."

output = minion(
    task=task,
    doc_metadata="Medical Report",
    context=[context],
    max_rounds=2
)
```

## Python Notebook

Explore Minions interactively in a Jupyter Notebook: `minions.ipynb`.

## Docker Support

Build, run, and use the Minions protocol within a Docker container.  The container includes an Ollama service for local inference.

### Build the Docker Image

```bash
docker build -t minions-docker .
```

### Run the Container

```bash
# Basic usage (includes Ollama service)
docker run -i minions-docker

# With Docker socket mounted (for Docker Model Runner)
docker run -i -v /var/run/docker.sock:/var/run/docker.sock minions-docker

# With API keys for remote models
docker run -i -e OPENAI_API_KEY=your_key -e ANTHROPIC_API_KEY=your_key minions-docker

# With custom Ollama host
docker run -i -e OLLAMA_HOST=0.0.0.0:11434 minions-docker

# For Streamlit app (legacy usage)
docker run -p 8501:8501 --env OPENAI_API_KEY=<your-openai-api-key> --env DEEPSEEK_API_KEY=<your-deepseek-api-key> minions-docker
```

### Docker Minion Protocol Usage

Use JSON input via stdin/stdout:

```json
{
  "local_client": {
    "type": "ollama",
    "model_name": "llama3.2:3b",
    "port": 11434,
    "kwargs": {}
  },
  "remote_client": {
    "type": "openai",
    "model_name": "gpt-4o",
    "kwargs": {
      "api_key": "your_openai_key"
    }
  },
  "protocol": {
    "type": "minion",
    "max_rounds": 3,
    "log_dir": "minion_logs",
    "kwargs": {}
  },
  "call_params": {
    "task": "Your task here",
    "context": ["Context string 1", "Context string 2"],
    "max_rounds": 2
  }
}
```

**Examples:**

```bash
echo '{...}' | docker run -i -e OPENAI_API_KEY=$OPENAI_API_KEY minions-docker
```

## CLI

Use the command line interface to run Minion/Minions.

1.  Set your local and remote model choices:

    ```bash
    export MINIONS_LOCAL=ollama/llama3.2
    export MINIONS_REMOTE=openai/gpt-4o
    ```

2.  Run:

    ```bash
    minions --help
    minions --context <path_to_context> --protocol <minion|minions>
    ```

## Secure Minions Local-Remote Protocol

Implement end-to-end encrypted communication between local and remote LLMs.  Includes attestation, perfect forward secrecy, and replay protection.

1.  Install Secure Dependencies: `pip install -e ".[secure]"`
2.  Use the Python API or CLI:

    ```python
    from minions.clients import OllamaClient
    from secure.minions_secure import SecureMinionProtocol

    local_client = OllamaClient(model_name="llama3.2")
    protocol = SecureMinionProtocol(...)
    result = protocol(task="...", context=["..."], max_rounds=2)
    ```

    ```bash
    python secure/minions_secure.py ...
    ```

## Secure Minions Chat

For secure chat functionality, install:

```bash
pip install -e ".[secure]"
```

See the [Secure Minions Chat README](secure/README.md) for setup and usage.

## Apps

Explore specialized applications within the `apps/` directory:

*   [A2A-Minions](apps/minions-a2a/) - Agent-to-Agent integration
*   [Character Chat](apps/minions-character-chat/) - Role-playing with AI personas
*   [Document Search](apps/minions-doc-search/) - Multi-method document retrieval
*   [Story Teller](apps/minions-story-teller/) - Creative storytelling with illustrations
*   [Tools Comparison](apps/minions-tools/) - MCP tools performance comparison
*   [WebGPU App](apps/minions-webgpu/) - Browser-based Minions protocol

## Inference Estimator

Get insights into LLM inference speed on your hardware.

### Command Line Usage

```bash
python -m minions.utils.inference_estimator --model llama3.2 --tokens 1000 --describe
```

### Python API Usage

```python
from minions.utils.inference_estimator import InferenceEstimator

estimator = InferenceEstimator(model_name="llama3.2", is_quant=True, quant_bits=4)
tokens_per_second, estimated_time = estimator.estimate(1000)
detailed_info = estimator.describe(1000)
```

## Miscellaneous Setup

### Using Azure OpenAI with Minions

1.  Set environment variables:

    ```bash
    export AZURE_OPENAI_API_KEY=your-api-key
    export AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
    export AZURE_OPENAI_API_VERSION=2024-02-15-preview
    ```

2.  Use the Azure OpenAI client in your code:

    ```python
    from minions.clients.ollama import OllamaClient
    from minions.clients.azure_openai import AzureOpenAIClient
    from minions.minion import Minion

    remote_client = AzureOpenAIClient(...)
    ```

## Maintainers

*   Avanika Narayan (avanika@cs.stanford.edu)
*   Dan Biderman (biderman@stanford.edu)
*   Sabri Eyuboglu (eyuboglu@cs.stanford.edu)