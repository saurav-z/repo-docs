[![Minions Logo](assets/Ollama_minionS_background.png)](https://github.com/HazyResearch/minions)

# Minions: Unleashing Collaboration Between On-Device and Cloud LLMs

**Minions empowers efficient LLM collaboration, enabling small on-device models to work seamlessly with powerful cloud models to reduce costs and improve performance.**

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/jfJyxXwFVa)

Explore the power of collaborative language models with Minions, a revolutionary communication protocol. This repository provides the tools and resources to get started.  **[Get started with Minions on GitHub!](https://github.com/HazyResearch/minions)** For a deeper dive, check out our [paper](https://arxiv.org/pdf/2502.15964) and [blogpost](https://hazyresearch.stanford.edu/blog/2025-02-24-minions).

**Key Features:**

*   **Cost-Effective LLM Collaboration:** Offloads long contexts to local, on-device models for reduced cloud costs.
*   **On-Device Inference:** Leverage the power of local models (e.g., Ollama, Lemonade, tokasaurus) for faster context processing.
*   **Cloud Integration:** Seamlessly integrates with leading cloud LLM providers (e.g., OpenAI, Together AI, DeepSeek) for superior performance.
*   **Secure Communication:** Supports an end-to-end encrypted protocol for private and secure interactions.
*   **WebGPU App:** Experience the Minions protocol directly in your browser with a WebGPU-powered interface.

**Jump to a Section:**

*   [Setup](#setup)
*   [Minions Demo Application](#minions-demo-application)
*   [Minions WebGPU App](#minions-webgpu-app)
*   [Example Code](#example-code)
*   [Docker Support](#docker-support)
*   [CLI](#cli)
*   [Secure Minions Protocol](#secure-minions-local-remote-protocol)
*   [Inference Estimator](#inference-estimator)
*   [Miscellaneous Setup](#miscellaneous-setup)
*   [Maintainers](#maintainers)

## Setup

Follow these steps to get started with Minions:

**Prerequisites:** Python 3.10 - 3.11 (Python 3.13 is not supported)

**Step 1: Clone and Install**

```bash
git clone https://github.com/HazyResearch/minions.git
cd minions
pip install -e .  # Installs the minions package in editable mode
```

*   Install MLX-LM: `pip install -e ".[mlx]"`
*   Install Secure Minions Chat: `pip install -e ".[secure]"`

**Step 2: Local Model Server Installation**

Choose and install at least one of the following local model servers:

*   **Ollama:** Recommended if you do not have access to NVIDIA/AMD GPUs. Install from [here](https://ollama.com/download). For Flash Attention, run `launchctl setenv OLLAMA_FLASH_ATTENTION 1` and restart the Ollama app (Mac).
*   **Lemonade:** Use if you have AMD CPUs/GPUs/NPUs. Install from [here](https://lemonade-server.ai/). Launch the server after installation. _Note: Lemonade does not support the Minion-CUA protocol at this time._
*   **Tokasaurus:** If you have NVIDIA GPUs and want to use the Minions protocol. `pip install tokasaurus`

<details>
  <summary>Optional: Install Cartesia-MLX (Apple Silicon)</summary>
   ... (follow installation instructions from the original README)
</details>

<details>
  <summary>Optional: Install llama-cpp-python</summary>
   ... (follow installation instructions from the original README)
</details>

**Step 3: Cloud LLM API Keys**

Set API keys for your preferred cloud LLM providers.

```bash
# OpenAI
export OPENAI_API_KEY=<your-openai-api-key>
export OPENAI_BASE_URL=<your-openai-base-url>  # Optional

# Together AI
export TOGETHER_API_KEY=<your-together-api-key>

# OpenRouter
export OPENROUTER_API_KEY=<your-openrouter-api-key>
export OPENROUTER_BASE_URL=<your-openrouter-base-url>  # Optional

# Perplexity
export PERPLEXITY_API_KEY=<your-perplexity-api-key>
export PERPLEXITY_BASE_URL=<your-perplexity-base-url>  # Optional

# Tokasaurus
export TOKASAURUS_BASE_URL=<your-tokasaurus-base-url>  # Optional

# DeepSeek
export DEEPSEEK_API_KEY=<your-deepseek-api-key>

# Anthropic
export ANTHROPIC_API_KEY=<your-anthropic-api-key>

# Mistral AI
export MISTRAL_API_KEY=<your-mistral-api-key>
```

## Minions Demo Application

[![Watch the video](https://img.youtube.com/vi/70Kot0_DFNs/0.jpg)](https://www.youtube.com/watch?v=70Kot0_DFNs)

Run the following commands to experience the Minion or Minions protocol:

```bash
pip install torch transformers
streamlit run app.py
```

If you encounter `ollama` client connection issues, try:

```bash
OLLAMA_FLASH_ATTENTION=1 ollama serve
```

## Minions WebGPU App

The WebGPU app provides a browser-based interface to the Minions protocol.

### Features

*   Browser-based: No local server setup.
*   WebGPU acceleration: Fast local model inference.
*   Model selection: Choose pre-optimized models from [MLC AI](https://mlc.ai/models).
*   Real-time updates: See progress and conversation logs.
*   Privacy-focused: Your API key and data stay in your browser.

### Quick Start

1.  Navigate to the WebGPU app directory: `cd apps/minions-webgpu`
2.  Install dependencies: `npm install`
3.  Start the development server: `npm start`
4.  Open your browser to the provided URL (usually `http://localhost:5173`)

## Example Code

Here's example code for using the `minion` and `minions` protocols.

### Minion (Singular)

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

context = """... (patient context) ..."""
task = "Based on the patient's blood pressure and LDL cholesterol readings in the context, evaluate whether these factors together suggest an increased risk for cardiovascular complications."

output = minion(
    task=task,
    context=[context],
    max_rounds=2
)
```

### Minions (Plural)

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

context = """... (patient context) ..."""

task = "Based on the patient's blood pressure and LDL cholesterol readings in the context, evaluate whether these factors together suggest an increased risk for cardiovascular complications."

output = minion(
    task=task,
    doc_metadata="Medical Report",
    context=[context],
    max_rounds=2
)
```

## Python Notebook

Find example usage in a Jupyter notebook, `minions.ipynb`.

## Docker Support

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

Use a JSON input format via stdin/stdout, with `local_client`, `remote_client`, `protocol`, and `call_params` keys.

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

*   **Basic Minion Protocol:** Example
    ```bash
    echo '{...}' | docker run -i -e OPENAI_API_KEY=$OPENAI_API_KEY minions-docker
    ```

*   **Minions (Parallel) Protocol:** Example
    ```bash
    echo '{...}' | docker run -i -e OPENAI_API_KEY=$OPENAI_API_KEY minions-docker
    ```

*   **Supported Client Types:** `ollama`, `docker_model_runner`, `openai`, `anthropic`.

*   **Supported Protocol Types:** `minion`, `minions`.

*   **Output Format:** JSON output with `success`, `result`, and `error` keys.

*   **Environment Variables:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_HOST`, `PYTHONPATH`, `PYTHONUNBUFFERED`.

*   **Advanced Usage:** Persistent containers, custom volumes, interactive mode.

## CLI

Run Minion/Minions via the command line interface using the `minions_cli.py` script.

```bash
export MINIONS_LOCAL=ollama/llama3.2
export MINIONS_REMOTE=openai/gpt-4o

minions --help
minions --context <path_to_context> --protocol <minion|minions>
```

## Secure Minions Local-Remote Protocol

The Secure Minions protocol offers end-to-end encryption with attestation, perfect forward secrecy, and replay protection.

**Prerequisites:**  Install secure dependencies: `pip install -e ".[secure]"`

### Basic Usage

#### Python API

```python
from minions.clients import OllamaClient
from secure.minions_secure import SecureMinionProtocol

local_client = OllamaClient(model_name="llama3.2")

protocol = SecureMinionProtocol(
    supervisor_url="https://your-supervisor-server.com",
    local_client=local_client,
    max_rounds=3,
    system_prompt="You are a helpful AI assistant."
)

result = protocol(
    task="Analyze this document for key insights",
    context=["Your document content here"],
    max_rounds=2
)

print(f"Final Answer: {result['final_answer']}")
print(f"Session ID: {result['session_id']}")
print(f"Log saved to: {result['log_file']}")

protocol.end_session()
```

#### Command Line Interface

```bash
python secure/minions_secure.py \
    --supervisor_url https://your-supervisor-server.com \
    --local_client_type ollama \
    --local_model llama3.2 \
    --max_rounds 3
```

## Secure Minions Chat

Secure Minions Chat provides an end-to-end encrypted chat experience. Install with `pip install -e ".[secure]"`. See the [Secure Minions Chat README](secure/README.md) for detailed setup and usage instructions.

## Inference Estimator

Estimate LLM inference speed to understand your hardware's performance.

### Command Line Usage

```bash
python -m minions.utils.inference_estimator --model llama3.2 --tokens 1000 --describe
```

*   `--model`: Model name (e.g., llama3.2, mistral7b)
*   `--tokens`: Number of tokens for estimation
*   `--describe`: Show detailed performance stats
*   `--quantized`: Model is quantized
*   `--quant-bits`: Quantization bits (4, 8, or 16)

### Python API Usage

```python
from minions.utils.inference_estimator import InferenceEstimator

estimator = InferenceEstimator(
    model_name="llama3.2",
    is_quant=True,
    quant_bits=4
)

tokens_per_second, estimated_time = estimator.estimate(1000)
print(f"Estimated speed: {tokens_per_second:.1f} tokens/sec")
print(f"Estimated time: {estimated_time:.2f} seconds")

detailed_info = estimator.describe(1000)
print(detailed_info)

estimator.calibrate(my_model_client, sample_tokens=32, prompt="Hello")
```

## Miscellaneous Setup

### Using Azure OpenAI with Minions

1.  Set environment variables:

```bash
export AZURE_OPENAI_API_KEY=your-api-key
export AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
export AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

2.  Example Code:

```python
from minions.clients.ollama import OllamaClient
from minions.clients.azure_openai import AzureOpenAIClient
from minions.minion import Minion

local_client = OllamaClient(
    model_name="llama3.2",
)

remote_client = AzureOpenAIClient(
    model_name="gpt-4o",  # Match your deployment name
    api_key="your-api-key",
    azure_endpoint="https://your-resource-name.openai.azure.com/",
    api_version="2024-02-15-preview",
)

minion = Minion(local_client, remote_client)
```

## Maintainers

*   Avanika Narayan (avanika@cs.stanford.edu)
*   Dan Biderman (biderman@stanford.edu)
*   Sabri Eyuboglu (eyuboglu@cs.stanford.edu)