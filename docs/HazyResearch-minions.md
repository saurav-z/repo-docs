![Minions Logo](assets/Ollama_minionS_background.png)

# Minions: Supercharge Your LLMs with On-Device and Cloud Collaboration

**Unlock cost-effective and efficient language model interactions by combining the power of on-device and cloud-based LLMs with Minions!**  [View the Minions Repository](https://github.com/HazyResearch/minions)

Minions is a communication protocol designed to seamlessly integrate small, on-device language models with powerful cloud-based models. This innovative approach allows you to leverage the benefits of both, reducing cloud costs while maintaining high-quality results.

**Key Features:**

*   **Cost-Effective:** Minimize cloud usage by processing long contexts locally.
*   **Collaborative:** Enables on-device models to work in tandem with frontier cloud models.
*   **Flexible:** Supports various local model servers and cloud LLM providers.
*   **Secure Option:**  Includes an end-to-end encrypted protocol for secure communication.
*   **WebGPU App:** Run the Minions protocol entirely in your browser.

**Explore Minions:**

*   **Paper:** [Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models](https://arxiv.org/pdf/2502.15964)
*   **Blog Post:** [Minions Blogpost](https://hazyresearch.stanford.edu/blog/2025-02-24-minions)
*   **Secure Minions Chat Blog Post:** [Secure Minions Chat Blogpost](https://hazyresearch.stanford.edu/blog/2025-05-12-security)

**Table of Contents**

> **Looking for Secure Minions Chat?**  Get secure, end-to-end encrypted chat with our dedicated system.  See the [Secure Minions Chat README](secure/README.md) for details.

*   [Setup](#setup)
    *   [Step 1: Clone and Install](#step-1-clone-and-install)
    *   [Step 2: Install a Local Model Server](#step-2-install-a-local-model-server)
    *   [Step 3: Set Cloud LLM API Keys](#step-3-set-cloud-llm-api-keys)
*   [Minions Demo Application](#minions-demo-application)
*   [Minions WebGPU App](#minions-webgpu-app)
*   [Example Code](#example-code)
    *   [Minion (Singular)](#example-code-minion-singular)
    *   [Minions (Plural)](#example-code-minions-plural)
*   [Python Notebook](#python-notebook)
*   [Docker Support](#docker-support)
    *   [Build the Docker Image](#build-the-docker-image)
    *   [Run the Container](#run-the-container)
    *   [Docker Minion Protocol Usage](#docker-minion-protocol-usage)
    *   [Supported Client Types](#supported-client-types)
    *   [Supported Protocol Types](#supported-protocol-types)
    *   [Output Format](#output-format)
    *   [Environment Variables](#environment-variables)
    *   [Tips and Advanced Usage](#tips-and-advanced-usage)
    *   [Persistent Container Usage](#persistent-container-usage)
*   [CLI](#cli)
*   [Secure Minions Local-Remote Protocol](#secure-minions-local-remote-protocol)
    *   [Prerequisites](#prerequisites)
    *   [Basic Usage](#basic-usage)
        *   [Python API](#python-api)
        *   [Command Line Interface](#command-line-interface)
*   [Secure Minions Chat](#secure-minions-chat)
*   [Apps](#apps)
*   [Inference Estimator](#inference-estimator)
    *   [Command Line Usage](#command-line-usage-1)
    *   [Python API Usage](#python-api-usage-1)
*   [Miscellaneous Setup](#miscellaneous-setup)
    *   [Using Azure OpenAI with Minions](#using-azure-openai-with-minions)
*   [Maintainers](#maintainers)

## Setup

*Tested on Mac and Ubuntu with Python 3.10-3.11 (Python 3.13 is not supported).*

<details>
  <summary>Optional: Create a virtual environment with your favorite package manager (e.g. conda, venv, uv)</summary>

```bash
conda create -n minions python=3.11
```

</details><br>

**Step 1: Clone and Install**

```bash
git clone https://github.com/HazyResearch/minions.git
cd minions
pip install -e .  # installs the minions package in editable mode
```

*   For optional MLX-LM, install with:

```bash
pip install -e ".[mlx]"
```

*   For Secure Minions Chat, install with:

```bash
pip install -e ".[secure]"
```

*   For optional Cartesia-MLX, install the basic package and follow the instructions below.

**Step 2: Install a Local Model Server**

Choose and install *at least one* of the following local model servers:

*   **Ollama:**  Recommended if you *don't* have NVIDIA/AMD GPUs.  Install instructions: [Ollama Download](https://ollama.com/download).  To enable Flash Attention: `launchctl setenv OLLAMA_FLASH_ATTENTION 1` and restart the Ollama app (Mac).
*   **Lemonade:** Use this if you have AMD CPUs/GPUs/NPUs. Install instructions: [Lemonade-server.ai](https://lemonade-server.ai/). Launch the Lemonade server (Windows GUI installer). *Note: Lemonade does not support the Minion-CUA protocol at this time.*  See the following for supported APU configurations: https://ryzenai.docs.amd.com/en/latest/llm/overview.html#supported-configurations
*   **Tokasaurus:**  For NVIDIA GPUs and high-throughput Minions protocol. Install with:

```bash
pip install tokasaurus
```

<details>
  <summary>Optional: Install Cartesia-MLX (Apple Silicon only)</summary>

1.  Download XCode
2.  Install command line tools: `xcode-select --install`
3.  Install Nanobind🧮

```bash
pip install nanobind@git+https://github.com/wjakob/nanobind.git@2f04eac452a6d9142dedb957701bdb20125561e4
```

4.  Install the Cartesia Metal backend:

```bash
pip install git+https://github.com/cartesia-ai/edge.git#subdirectory=cartesia-metal
```

5.  Install the Cartesia-MLX package:

```bash
pip install git+https://github.com/cartesia-ai/edge.git#subdirectory=cartesia-mlx
```

</details><br>

<details>
  <summary>Optional: Install llama-cpp-python</summary>

```bash
# CPU-only
pip install llama-cpp-python

# Metal (Apple Silicon/Intel)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

# CUDA (NVIDIA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# OpenBLAS CPU optimizations
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

See the [llama-cpp-python documentation](https://llama-cpp-python.readthedocs.io/en/latest/) for more options.

```python
from minions.clients import LlamaCppClient

# Initialize LlamaCppClient
client = LlamaCppClient(
    model_path="/path/to/model.gguf",
    chat_format="chatml",
    n_gpu_layers=35
)

# Example chat completion
messages = [
    {"role": "system", "content": "Helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
]

responses, usage, done_reasons = client.chat(messages)
print(responses[0])
```

```python
client = LlamaCppClient(
    model_path="dummy",
    model_repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file_pattern="*Q4_K_M.gguf",
    chat_format="chatml",
    n_gpu_layers=35
)
```

</details><br>

**Step 3: Set Cloud LLM API Keys**

*   Create API keys (if needed) for your chosen cloud LLM providers: [OpenAI](https://platform.openai.com/docs/overview), [TogetherAI](https://docs.together.ai/docs/quickstart), [DeepSeek](https://platform.deepseek.com/api_keys).

```bash
# OpenAI
export OPENAI_API_KEY=<your-openai-api-key>
export OPENAI_BASE_URL=<your-openai-base-url>

# Together AI
export TOGETHER_API_KEY=<your-together-api-key>

# OpenRouter
export OPENROUTER_API_KEY=<your-openrouter-api-key>
export OPENROUTER_BASE_URL=<your-openrouter-base-url>

# Perplexity
export PERPLEXITY_API_KEY=<your-perplexity-api-key>
export PERPLEXITY_BASE_URL=<your-perplexity-base-url>

# Tokasaurus
export TOKASAURUS_BASE_URL=<your-tokasaurus-base-url>

# DeepSeek
export DEEPSEEK_API_KEY=<your-deepseek-api-key>

# Anthropic
export ANTHROPIC_API_KEY=<your-anthropic-api-key>

# Mistral AI
export MISTRAL_API_KEY=<your-mistral-api-key>
```

## Minions Demo Application

[![Watch the video](https://img.youtube.com/vi/70Kot0_DFNs/0.jpg)](https://www.youtube.com/watch?v=70Kot0_DFNs)

To run the Minion or Minions protocol demo:

```bash
pip install torch transformers
streamlit run app.py
```

If you encounter an Ollama connection error, try:

```bash
OLLAMA_FLASH_ATTENTION=1 ollama serve
```

## Minions WebGPU App

The Minions WebGPU app enables the Minions protocol entirely within your browser, utilizing WebGPU for local model inference and cloud APIs.

**Features:**

*   **Browser-based:** No local server required.
*   **WebGPU Acceleration:** Fast local model inference.
*   **Model Selection:** Choose from pre-optimized models from [MLC AI](https://mlc.ai/models).
*   **Real-time Progress:** See loading and conversation logs.
*   **Privacy-Focused:** API keys and data stay within your browser.

**Quick Start:**

1.  Navigate to the WebGPU app directory:

```bash
cd apps/minions-webgpu
```

2.  Install dependencies:

```bash
npm install
```

3.  Start the development server:

```bash
npm start
```

4.  Open your browser to the displayed URL (usually `http://localhost:5173`).

## Example Code

### Minion (Singular)

This example uses an `ollama` local client and an `openai` remote client with the `minion` protocol:

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
Patient John Doe is a 60-year-old male with a history of hypertension. In his latest checkup, his blood pressure was recorded at 160/100 mmHg, and he reported occasional chest discomfort during physical activity.
Recent laboratory results show that his LDL cholesterol level is elevated at 170 mg/dL, while his HDL remains within the normal range at 45 mg/dL. Other metabolic indicators, including fasting glucose and renal function, are unremarkable.
"""

task = "Based on the patient's blood pressure and LDL cholesterol readings in the context, evaluate whether these factors together suggest an increased risk for cardiovascular complications."

output = minion(
    task=task,
    context=[context],
    max_rounds=2
)
```

### Minions (Plural)

This example uses an `ollama` local client and an `openai` remote client with the `minions` protocol:

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
Patient John Doe is a 60-year-old male with a history of hypertension. In his latest checkup, his blood pressure was recorded at 160/100 mmHg, and he reported occasional chest discomfort during physical activity.
Recent laboratory results show that his LDL cholesterol level is elevated at 170 mg/dL, while his HDL remains within the normal range at 45 mg/dL. Other metabolic indicators, including fasting glucose and renal function, are unremarkable.
"""

task = "Based on the patient's blood pressure and LDL cholesterol readings in the context, evaluate whether these factors together suggest an increased risk for cardiovascular complications."

output = minion(
    task=task,
    doc_metadata="Medical Report",
    context=[context],
    max_rounds=2
)
```

## Python Notebook

Explore `minions.ipynb` for running Minion/Minions in a notebook environment.

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

The Docker container accepts JSON input via stdin for running minion protocols.

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

#### Usage Examples

**Basic Minion Protocol:**

```bash
echo '{
  "local_client": {
    "type": "ollama",
    "model_name": "llama3.2:3b"
  },
  "remote_client": {
    "type": "openai",
    "model_name": "gpt-4o"
  },
  "protocol": {
    "type": "minion",
    "max_rounds": 3
  },
  "call_params": {
    "task": "Analyze the patient data and provide a diagnosis",
    "context": ["Patient John Doe is a 60-year-old male with hypertension. Blood pressure: 160/100 mmHg. LDL cholesterol: 170 mg/dL."]
  }
}' | docker run -i -e OPENAI_API_KEY=$OPENAI_API_KEY minions-docker
```

**Minions (Parallel) Protocol:**

```bash
echo '{
  "local_client": {
    "type": "ollama",
    "model_name": "llama3.2:3b"
  },
  "remote_client": {
    "type": "openai",
    "model_name": "gpt-4o"
  },
  "protocol": {
    "type": "minions"
  },
  "call_params": {
    "task": "Analyze the financial data and extract key insights",
    "doc_metadata": "Financial report",
    "context": ["Revenue increased by 15% year-over-year. Operating expenses rose by 8%. Net profit margin improved to 12%."]
  }
}' | docker run -i -e OPENAI_API_KEY=$OPENAI_API_KEY minions-docker
```

### Supported Client Types

*   **Local Clients:**
    *   `ollama`: Uses Ollama for local inference (included in container)
    *   `docker_model_runner`: Uses Docker Model Runner for local inference
*   **Remote Clients:**
    *   `openai`: OpenAI API
    *   `anthropic`: Anthropic API

### Supported Protocol Types

*   `minion`: Single conversation protocol
*   `minions`: Parallel processing protocol

### Output Format

```json
{
  "success": true,
  "result": {
    "final_answer": "The analysis result...",
    "supervisor_messages": [...],
    "worker_messages": [...],
    "remote_usage": {...},
    "local_usage": {...}
  },
  "error": null
}
```

or on error:

```json
{
  "success": false,
  "result": null,
  "error": "Error message"
}
```

### Environment Variables

*   `OPENAI_API_KEY`: OpenAI API key
*   `ANTHROPIC_API_KEY`: Anthropic API key
*   `OLLAMA_HOST`: Ollama service host (default: 0.0.0.0:11434)
*   `PYTHONPATH`: Python path (default: /app)
*   `PYTHONUNBUFFERED`: Unbuffered output (default: 1)

### Tips and Advanced Usage

1.  **Ollama Models:** The container will automatically pull models on first use.
2.  **Docker Model Runner:** Ensure Docker is running and accessible.
3.  **API Keys:** Pass as environment variables.
4.  **Volumes:** Use volumes for persistent storage.
5.  **Networking:** Use `--network host` for local service access.

**With Custom Volumes:**

```bash
docker run -i \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/logs:/app/minion_logs \
  -v $(pwd)/workspace:/app/workspace \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  minions-docker
```

**Interactive Mode:**

```bash
docker run -it \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  minions-docker bash
```

Then you can run the interface manually:

```bash
python minion_stdin_interface.py
```

### Persistent Container Usage

For running multiple queries without restarting the container and Ollama service each time:

**1. Start a persistent container:**

```bash
docker run -d --name minions-container -e OPENAI_API_KEY="$OPENAI_API_KEY" minions-docker
```

**2. Send queries to the running container:**

```bash
echo '{
  "local_client": {"type": "ollama", "model_name": "llama3.2:3b"},
  "remote_client": {"type": "openai", "model_name": "gpt-4o"},
  "protocol": {"type": "minion", "max_rounds": 1},
  "call_params": {"task": "How many times did Roger Federer end the year as No.1?"}
}' | docker exec -i minions-container /app/start_minion.sh
```

**3. Send additional queries (fast, no restart delay):**

```bash
echo '{
  "local_client": {"type": "ollama", "model_name": "llama3.2:3b"},
  "remote_client": {"type": "openai", "model_name": "gpt-4o"},
  "protocol": {"type": "minion", "max_rounds": 1},
  "call_params": {"task": "What is the capital of France?"}
}' | docker exec -i minions-container /app/start_minion.sh
```

**4. Clean up when done:**

```bash
docker stop minions-container
docker rm minions-container
```

**Advantages of persistent containers:**

*   ✅ **Ollama stays running** - no restart delays between queries
*   ✅ **Models stay loaded** - faster subsequent queries
*   ✅ **Resource efficient** - one container handles multiple queries
*   ✅ **Automatic model pulling** - models are downloaded on first use

## CLI

Run Minion/Minions in the CLI using `minions_cli.py`.

Set your local and remote models:

```bash
export MINIONS_LOCAL=ollama/llama3.2
export MINIONS_REMOTE=openai/gpt-4o
```

Available providers: `ollama`, `openai`, `anthropic`, `together`, `perplexity`, `openrouter`, `groq`, and `mlx`.

```bash
minions --help
```

```bash
minions --context <path_to_context> --protocol <minion|minions>
```

## Secure Minions Local-Remote Protocol

The `secure/minions_secure.py` file implements an end-to-end encrypted Minions protocol, ensuring secure communication between local and remote models.

### Prerequisites

Install secure dependencies:

```bash
pip install -e ".[secure]"
```

### Basic Usage

#### Python API

```python
from minions.clients import OllamaClient
from secure.minions_secure import SecureMinionProtocol

# Initialize local client
local_client = OllamaClient(model_name="llama3.2")

# Create secure protocol instance
protocol = SecureMinionProtocol(
    supervisor_url="https://your-supervisor-server.com",
    local_client=local_client,
    max_rounds=3,
    system_prompt="You are a helpful AI assistant."
)

# Run a secure task
result = protocol(
    task="Analyze this document for key insights",
    context=["Your document content here"],
    max_rounds=2
)

print(f"Final Answer: {result['final_answer']}")
print(f"Session ID: {result['session_id']}")
print(f"Log saved to: {result['log_file']}")

# Clean up the session
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

Install Secure Minions Chat:

```bash
pip install -e ".[secure]"
```

See the [Secure Minions Chat README](secure/README.md) for detailed setup and usage.

## Apps

Explore example applications in the `apps/` directory:

*   📊 **[A2A-Minions](apps/minions-a2a/)** - Agent-to-Agent integration server
*   🎭 **[Character Chat](apps/minions-character-chat/)** - Role-playing with AI personas
*   🔍 **[Document Search](apps/minions-doc-search/)** - Multi-method document retrieval
*   📚 **[Story Teller](apps/minions-story-teller/)** - Creative storytelling with illustrations
*   🛠️ **[Tools Comparison](apps/minions-tools/)** - MCP tools performance comparison
*   🌐 **[WebGPU App](apps/minions-webgpu/)** - Browser-based Minions protocol

## Inference Estimator

The Minions Inference Estimator provides a utility for measuring LLM inference speed.

**Key Features:**

1.  Analyze hardware capabilities.
2.  Calculate model performance.
3.  Estimate tokens/second and completion time.

### Command Line Usage

```bash
python -m minions.utils.inference_estimator --model llama3.2 --tokens 1000 --describe
```

*   `--model`: Model name (e.g., llama3.2, mistral7b)
*   `--tokens`: Number of tokens to estimate.
*   `--describe`: Show detailed statistics.
*   `--quantized`: Specify if model is quantized.
*   `--quant-bits`: Quantization bit-width (4, 8, or 16).

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

# Calibrate with your model client
estimator.calibrate(my_model_client, sample_tokens=32, prompt="Hello")
```

Calibration data is cached at `~/.cache/ie_calib.json`.

## Miscellaneous Setup

### Using Azure OpenAI with Minions

#### Set Environment Variables

```bash
export AZURE_OPENAI_API_KEY=your-api-key
export AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
export AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

#### Example Code

```python
from minions.clients.ollama import OllamaClient
from minions.clients.azure_openai import AzureOpenAIClient
from minions.minion import Minion

local_client = OllamaClient(
    model_name="llama3.2",
)

remote_client = AzureOpenAIClient(
    model_name="gpt-4o",  # This should match your deployment name
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