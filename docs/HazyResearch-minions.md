![Minions Logo](assets/Ollama_minionS_background.png)

# Minions: Unleash the Power of On-Device and Cloud LLMs

**Collaborate with efficiency!** Minions is a cutting-edge communication protocol designed to seamlessly connect small, on-device language models with powerful cloud-based models, dramatically reducing cloud costs without sacrificing quality.  [Explore the Minions Project on GitHub](https://github.com/HazyResearch/minions).

**Key Features:**

*   **Cost-Effective:** Leverage on-device models for long contexts, minimizing expensive cloud usage.
*   **Flexible Architecture:**  Supports various local and cloud LLM combinations.
*   **Easy Integration:** Simple to set up and use with popular LLM providers like OpenAI, Together AI, and more.
*   **Secure Communication:** Offers an end-to-end encrypted protocol for secure interactions (see the [Secure Minions Chat README](secure/README.md)).
*   **Browser-Based Demo:** Experience the Minions protocol directly in your browser with the WebGPU App.
*   **Inference Estimator:** Accurately assess LLM inference speeds on your hardware.

**Get Started:**

*   **Paper:** [Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models](https://arxiv.org/pdf/2502.15964)
*   **Blog Post:** https://hazyresearch.stanford.edu/blog/2025-02-24-minions
*   **Secure Minions Chat Blogpost:** https://hazyresearch.stanford.edu/blog/2025-05-12-security

## Table of Contents

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
    *   [Command Line Usage](#command-line-usage)
    *   [Python API Usage](#python-api-usage)
*   [Miscellaneous Setup](#miscellaneous-setup)
    *   [Using Azure OpenAI with Minions](#using-azure-openai-with-minions)
*   [Maintainers](#maintainers)

## Setup

_We have tested the following setup on Mac and Ubuntu with Python 3.10-3.11_ (Note: Python 3.13 is not supported)

<details>
  <summary>Optional: Create a virtual environment (e.g., conda, venv, uv)</summary>

```python
conda create -n minions python=3.11
```

</details><br>

**Step 1: Clone and Install**

```bash
git clone https://github.com/HazyResearch/minions.git
cd minions
pip install -e .  # installs the minions package in editable mode
```

_Optional package installations:_

```bash
pip install -e ".[mlx]" # MLX-LM
pip install -e ".[secure]" # Secure Minions Chat
```

**Step 2: Install a Local Model Server**

Choose one of the following servers for running your local model: `lemonade`, `ollama`, or `tokasaurus`.  You will need to install at least one.

*   **Ollama:** Use this if you don't have NVIDIA/AMD GPUs.  Install it following the instructions [here](https://ollama.com/download).  To enable Flash Attention, run `launchctl setenv OLLAMA_FLASH_ATTENTION 1` and, if on a Mac, restart the ollama app.
*   **Lemonade:** Use this if you have access to local AMD CPUs/GPUs/NPUs. Install it following the instructions [here](https://lemonade-server.ai/).
    *   See supported APU configurations at: https://ryzenai.docs.amd.com/en/latest/llm/overview.html#supported-configurations
    *   After installing `lemonade` make sure to launch the lemonade server. This can be done via the [one-click Windows GUI](https://lemonade-server.ai/) installer which installs the Lemonade Server as a standalone tool.
    *   Note: Lemonade does not support the Minion-CUA protocol at this time.
*   **Tokasaurus:** Use this if you have NVIDIA GPUs and are running the Minions protocol, which benefits from `tokasaurus`'s high throughput. Install it with:
    ```bash
    pip install tokasaurus
    ```

<details>
  <summary>Optional: Install Cartesia-MLX (Apple Silicon only)</summary>

1.  Download XCode
2.  Install command line tools: `xcode-select --install`
3.  Install Nanobind:
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
# CPU-only installation
pip install llama-cpp-python

# For Metal on Mac (Apple Silicon/Intel)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

# For CUDA on NVIDIA GPUs
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# For OpenBLAS CPU optimizations
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

For more installation options, see the [llama-cpp-python documentation](https://llama-cpp-python.readthedocs.io/en/latest/).

## Basic Usage

The client follows the basic pattern from the llama-cpp-python library:

```python
from minions.clients import LlamaCppClient

# Initialize the client with a local model
client = LlamaCppClient(
    model_path="/path/to/model.gguf",
    chat_format="chatml",     # Most modern models use "chatml" format
    n_gpu_layers=35           # Set to 0 for CPU-only inference
)

# Run a chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
]

responses, usage, done_reasons = client.chat(messages)
print(responses[0])  # The generated response
```

## Loading Models from Hugging Face

You can easily load models directly from Hugging Face:

```python
client = LlamaCppClient(
    model_path="dummy",  # Will be replaced by downloaded model
    model_repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file_pattern="*Q4_K_M.gguf",  # Optional - specify quantization
    chat_format="chatml",
    n_gpu_layers=35     # Offload 35 layers to GPU
)
```

</details><br>

**Step 3: Set Cloud LLM API Keys**

Set your API key for at least one of the following cloud LLM providers. If needed, create an account and get an API key from the provider of your choice:  [OpenAI](https://platform.openai.com/docs/overview), [TogetherAI](https://docs.together.ai/docs/quickstart), [DeepSeek](https://platform.deepseek.com/api_keys), or other supported services.

```bash
# OpenAI
export OPENAI_API_KEY=<your-openai-api-key>
export OPENAI_BASE_URL=<your-openai-base-url>  # Optional: Use a different OpenAI API endpoint

# Together AI
export TOGETHER_API_KEY=<your-together-api-key>

# OpenRouter
export OPENROUTER_API_KEY=<your-openrouter-api-key>
export OPENROUTER_BASE_URL=<your-openrouter-base-url>  # Optional: Use a different OpenRouter API endpoint

# Perplexity
export PERPLEXITY_API_KEY=<your-perplexity-api-key>
export PERPLEXITY_BASE_URL=<your-perplexity-base-url>  # Optional: Use a different Perplexity API endpoint

# Tokasaurus
export TOKASAURUS_BASE_URL=<your-tokasaurus-base-url>  # Optional: Use a different Tokasaurus API endpoint

# DeepSeek
export DEEPSEEK_API_KEY=<your-deepseek-api-key>

# Anthropic
export ANTHROPIC_API_KEY=<your-anthropic-api-key>

# Mistral AI
export MISTRAL_API_KEY=<your-mistral-ai-key>
```

## Minions Demo Application

[![Watch the video](https://img.youtube.com/vi/70Kot0_DFNs/0.jpg)](https://www.youtube.com/watch?v=70Kot0_DFNs)

Run the following commands to try the Minion or Minions protocol:

```bash
pip install torch transformers
streamlit run app.py
```

If you see an error about the `ollama` client, try running:

```bash
OLLAMA_FLASH_ATTENTION=1 ollama serve
```

## Minions WebGPU App

The Minions WebGPU app provides a browser-based experience for the Minions protocol using WebGPU for local inference and cloud APIs. No local server is needed.

### Features

*   **Browser-based:** Runs entirely in your web browser.
*   **WebGPU acceleration:** Uses WebGPU for fast local model inference.
*   **Model selection:** Choose from pre-optimized [MLC AI](https://mlc.ai/models) models.
*   **Real-time progress:** See model loading and conversation logs in real-time.
*   **Privacy-focused:** Your API key and data stay in your browser.

### Quick Start

1.  **Navigate to the WebGPU app directory:**
    ```bash
    cd apps/minions-webgpu
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Start the development server:**
    ```bash
    npm start
    ```

4.  **Open your browser** and navigate to the URL shown in the terminal (typically `http://localhost:5173`)

## Example Code

### Example Code: Minion (singular)

This example uses an `ollama` local client and an `openai` remote client with the `minion` protocol.

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

# Instantiate the Minion object with both clients
minion = Minion(local_client, remote_client)


context = """
Patient John Doe is a 60-year-old male with a history of hypertension. In his latest checkup, his blood pressure was recorded at 160/100 mmHg, and he reported occasional chest discomfort during physical activity.
Recent laboratory results show that his LDL cholesterol level is elevated at 170 mg/dL, while his HDL remains within the normal range at 45 mg/dL. Other metabolic indicators, including fasting glucose and renal function, are unremarkable.
"""

task = "Based on the patient's blood pressure and LDL cholesterol readings in the context, evaluate whether these factors together suggest an increased risk for cardiovascular complications."

# Execute the minion protocol for up to two communication rounds
output = minion(
    task=task,
    context=[context],
    max_rounds=2
)
```

### Example Code: Minions (plural)

This example uses an `ollama` local client and an `openai` remote client with the `minions` protocol.

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


# Instantiate the Minion object with both clients
minion = Minions(local_client, remote_client)


context = """
Patient John Doe is a 60-year-old male with a history of hypertension. In his latest checkup, his blood pressure was recorded at 160/100 mmHg, and he reported occasional chest discomfort during physical activity.
Recent laboratory results show that his LDL cholesterol level is elevated at 170 mg/dL, while his HDL remains within the normal range at 45 mg/dL. Other metabolic indicators, including fasting glucose and renal function, are unremarkable.
"""

task = "Based on the patient's blood pressure and LDL cholesterol readings in the context, evaluate whether these factors together suggest an increased risk for cardiovascular complications."

# Execute the minion protocol for up to two communication rounds
output = minion(
    task=task,
    doc_metadata="Medical Report",
    context=[context],
    max_rounds=2
)
```

## Python Notebook

Find example usage in a Jupyter Notebook by checking out `minions.ipynb`.

## Docker Support

### Build the Docker Image

```bash
docker build -t minions-docker .
```

### Run the Container

**Note:** The container automatically starts an Ollama service for local model inference. This allows you to use models like `llama3.2:3b` without extra setup.

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

The Docker container uses a stdin/stdout interface for running minion protocols. It expects JSON input with the following structure:

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

#### Supported Client Types

**Local Clients:**
-   `ollama`: Uses Ollama for local inference (included in container)
-   `docker_model_runner`: Uses Docker Model Runner for local inference

**Remote Clients:**
-   `openai`: OpenAI API
-   `anthropic`: Anthropic API

#### Supported Protocol Types

-   `minion`: Single conversation protocol
-   `minions`: Parallel processing protocol

#### Output Format

The container outputs JSON:

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

Or on error:

```json
{
  "success": false,
  "result": null,
  "error": "Error message"
}
```

#### Environment Variables

-   `OPENAI_API_KEY`: OpenAI API key
-   `ANTHROPIC_API_KEY`: Anthropic API key
-   `OLLAMA_HOST`: Ollama service host (defaults to 0.0.0.0:11434)
-   `PYTHONPATH`: Python path (defaults to /app)
-   `PYTHONUNBUFFERED`: Unbuffered output (defaults to 1)

#### Tips and Advanced Usage

1.  **Ollama Models**: The container will automatically pull models on first use (e.g., `llama3.2:3b`).
2.  **Docker Model Runner**: Docker must be running and accessible inside the container.
3.  **API Keys**: Pass API keys as environment variables for security.
4.  **Volumes**: Mount volumes for persistent workspaces or logs.
5.  **Networking**: Use `--network host` to access local services.

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

Then run the interface manually:
```bash
python minion_stdin_interface.py 
```

#### Persistent Container Usage

For running multiple queries without restarting:

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
-   ✅ **Ollama stays running** - no restart delays between queries
-   ✅ **Models stay loaded** - faster subsequent queries
-   ✅ **Resource efficient** - one container handles multiple queries
-   ✅ **Automatic model pulling** - models are downloaded on first use

## CLI

To run Minion/Minions in a CLI, check out `minions_cli.py`.

Set your choice of local and remote models: `<provider>/<model_name>`.  Choose from `ollama`, `openai`, `anthropic`, `together`, `perplexity`, `openrouter`, `groq`, and `mlx`.

```bash
export MINIONS_LOCAL=ollama/llama3.2
export MINIONS_REMOTE=openai/gpt-4o
```

```bash
minions --help
```

```bash
minions --context <path_to_context> --protocol <minion|minions>
```

## Secure Minions Local-Remote Protocol

The Secure Minions Local-Remote Protocol (`secure/minions_secure.py`) enables secure end-to-end encrypted communication, with attestation, forward secrecy, and replay protection.

### Prerequisites

Install the secure dependencies:

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

To install secure minions chat, install the package with the following command:

```bash
pip install -e ".[secure]"
```

See the [Secure Minions Chat README](secure/README.md) for setup and usage details.

## Apps

The `apps/` directory contains applications demonstrating use cases:

*   📊  [A2A-Minions](apps/minions-a2a/) - Agent-to-Agent integration server
*   🎭  [Character Chat](apps/minions-character-chat/) - Role-playing with AI personas
*   🔍  [Document Search](apps/minions-doc-search/) - Multi-method document retrieval
*   📚  [Story Teller](apps/minions-story-teller/) - Creative storytelling with illustrations
*   🛠️  [Tools Comparison](apps/minions-tools/) - MCP tools performance comparison
*   🌐  [WebGPU App](apps/minions-webgpu/) - Browser-based Minions protocol

## Inference Estimator

The Inference Estimator tool helps you estimate LLM inference speeds.

### Command Line Usage

Run the estimator directly:

```bash
python -m minions.utils.inference_estimator --model llama3.2 --tokens 1000 --describe
```

Arguments:

-   `--model`: Model name (e.g., llama3.2, mistral7b)
-   `--tokens`: Number of tokens to estimate
-   `--describe`: Show detailed hardware and model performance statistics
-   `--quantized`: Specify that the model is quantized
-   `--quant-bits`: Quantization bit-width (4, 8, or 16)

### Python API Usage

```python
from minions.utils.inference_estimator import InferenceEstimator

# Initialize the estimator for a specific model
estimator = InferenceEstimator(
    model_name="llama3.2",  # Model name
    is_quant=True,          # Is model quantized?
    quant_bits=4            # Quantization level (4, 8, 16)
)

# Estimate performance for 1000 tokens
tokens_per_second, estimated_time = estimator.estimate(1000)
print(f"Estimated speed: {tokens_per_second:.1f} tokens/sec")
print(f"Estimated time: {estimated_time:.2f} seconds")

# Get detailed stats
detailed_info = estimator.describe(1000)
print(detailed_info)

# Calibrate with your actual model client for better accuracy
# (requires a model client that implements a chat() method)
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

# Instantiate the Minion object with both clients
minion = Minion(local_client, remote_client)
```

## Maintainers

*   Avanika Narayan (avanika@cs.stanford.edu)
*   Dan Biderman (biderman@stanford.edu)
*   Sabri Eyuboglu (eyuboglu@cs.stanford.edu)