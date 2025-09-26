![Minions Logo](assets/Ollama_minionS_background.png)

# Minions: Unleash the Power of Hybrid LLMs for Cost-Effective AI

**Collaborate with on-device and cloud-based LLMs seamlessly to reduce costs and enhance performance!**  [Explore the Minions Repository](https://github.com/HazyResearch/minions)

Minions is a revolutionary communication protocol that empowers small, on-device language models to collaborate with powerful cloud-based models. This hybrid approach offers significant cost savings by processing long contexts locally while leveraging the strengths of frontier models in the cloud. Explore the power of Minions and discover how it can transform your AI workflows.

*   **Cost-Effective LLM Usage:** Reduce cloud costs by processing data locally.
*   **On-Device and Cloud Collaboration:** Seamlessly integrate on-device and cloud models for optimal performance.
*   **Flexible Implementation:**  Use the provided demonstration and adapt the protocol to your specific needs.

**Read the paper and blogpost for more information:**

*   Paper: [Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models](https://arxiv.org/pdf/2502.15964)
*   Blogpost: https://hazyresearch.stanford.edu/blog/2025-02-24-minions
*   Secure Minions Chat Blogpost: https://hazyresearch.stanford.edu/blog/2025-05-12-security

## Key Features

*   **Flexible Client Support:** Supports a variety of local and remote LLM clients, including Ollama, OpenAI, Together AI, and more.
*   **Multiple Protocol Options:** Choose between the Minion (singular) and Minions (plural) protocols.
*   **Secure Communication:** Benefit from the end-to-end encrypted Secure Minions Local-Remote Protocol for secure data transfer.
*   **WebGPU App:** Interact with Minions directly in your browser using the WebGPU app.
*   **Docker Support:** Easily run Minions using Docker for streamlined deployment.
*   **Inference Estimator:** Optimize performance with the built-in inference estimator.
*   **Comprehensive Examples:**  Get started quickly with provided example code, a Python notebook, and a CLI.
*   **Example Apps:** Includes several example apps demonstrating the power of Minions.

## Table of Contents

*   [Setup](#setup)
    *   [Step 1: Clone and Install](#step-1-clone-and-install)
    *   [Step 2: Install a Local Model Server](#step-2-install-a-local-model-server)
    *   [Step 3: Set API Keys](#step-3-set-api-keys)
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
*   [CLI](#cli)
*   [Secure Minions Local-Remote Protocol](#secure-minions-local-remote-protocol)
*   [Secure Minions Chat](#secure-minions-chat)
*   [Apps](#apps)
*   [Inference Estimator](#inference-estimator)
    *   [Command Line Usage](#command-line-usage)
    *   [Python API Usage](#python-api-usage)
*   [Miscellaneous Setup](#miscellaneous-setup)
    *   [Using Azure OpenAI](#using-azure-openai-with-minions)
*   [Maintainers](#maintainers)

## Setup

Follow these steps to get started with Minions:

**Note:** The following setup has been tested on Mac and Ubuntu with Python 3.10-3.11. (Python 3.13 is not supported)

### Step 1: Clone and Install

1.  Clone the repository and install the Python package:

```bash
git clone https://github.com/HazyResearch/minions.git
cd minions
pip install -e .  # Installs the minions package in editable mode
```

**Optional Installation Notes:**

*   For optional MLX-LM, install with:

```bash
pip install -e ".[mlx]"
```

*   For Secure Minions Chat, install with:

```bash
pip install -e ".[secure]"
```

### Step 2: Install a Local Model Server

Choose and install one of the following local model servers:

*   **Ollama:**  Use if you do not have access to NVIDIA/AMD GPUs.  Install instructions [here](https://ollama.com/download).
    *   Enable Flash Attention: `launchctl setenv OLLAMA_FLASH_ATTENTION 1` and, if on a mac, restart the ollama app.
*   **Lemonade:**  Use if you have access to local AMD CPUs/GPUs/NPUs. Install instructions [here](https://lemonade-server.ai/).
    *   See the following for supported APU configurations: https://ryzenai.docs.amd.com/en/latest/llm/overview.html#supported-configurations
    *   Launch the Lemonade server after installation.
    *   **Note:** Lemonade does not support the Minion-CUA protocol at this time.
*   **Tokasaurus:**  Use if you have access to NVIDIA GPUs and are running the Minions protocol. Install with:

```bash
pip install tokasaurus
```

**Optional: Install Cartesia-MLX (Apple Silicon only)**

Follow the instructions in the original README for Cartesia-MLX installation.

**Optional: Install llama-cpp-python**

Follow the instructions in the original README for llama-cpp-python installation.

### Step 3: Set API Keys

Set your API key for at least one cloud LLM provider:

*   Create an [OpenAI API Key](https://platform.openai.com/docs/overview) or [TogetherAI API key](https://docs.together.ai/docs/quickstart) or [DeepSeek API key](https://platform.deepseek.com/api_keys).

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
export MISTRAL_API_KEY=<your-mistral-api-key>
```

## Minions Demo Application

[![Watch the video](https://img.youtube.com/vi/70Kot0_DFNs/0.jpg)](https://www.youtube.com/watch?v=70Kot0_DFNs)

To quickly try the Minion or Minions protocol, run:

```bash
pip install torch transformers

streamlit run app.py
```

**Troubleshooting Ollama:**

If you encounter an error related to the `ollama` client, try running:

```bash
OLLAMA_FLASH_ATTENTION=1 ollama serve
```

## Minions WebGPU App

The Minions WebGPU app runs entirely in the browser, leveraging WebGPU for local model inference and cloud APIs for supervision. This eliminates the need for local server setup while offering a user-friendly web interface.

### Features

*   **Browser-based:** Runs entirely in your web browser.
*   **WebGPU Acceleration:** Uses WebGPU for fast local model inference.
*   **Model Selection:** Choose from pre-optimized models from [MLC AI](https://mlc.ai/models).
*   **Real-time Progress:** See model loading and conversation logs in real-time.
*   **Privacy-focused:** API keys and data never leave your browser.

### Quick Start

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

4.  Open your browser and navigate to the displayed URL (typically `http://localhost:5173`).

## Example Code

### Minion (Singular)

This example demonstrates using an `ollama` local client and an `openai` remote client with the `minion` protocol.

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

### Minions (Plural)

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

Explore `minions.ipynb` to run Minion/Minions within a Jupyter notebook environment.

## Docker Support

### Build the Docker Image

```bash
docker build -t minions-docker .
```

### Run the Container

**Note:**  The container automatically starts an Ollama service for local model inference.

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

The Docker container supports a stdin/stdout interface for running minion protocols using JSON input.

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

*   `ollama`: Uses Ollama for local inference (included in container)
*   `docker_model_runner`: Uses Docker Model Runner for local inference

**Remote Clients:**

*   `openai`: OpenAI API
*   `anthropic`: Anthropic API

#### Supported Protocol Types

*   `minion`: Single conversation protocol
*   `minions`: Parallel processing protocol

#### Output Format

The container outputs JSON with:

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

Or, on error:

```json
{
  "success": false,
  "result": null,
  "error": "Error message"
}
```

#### Environment Variables

*   `OPENAI_API_KEY`: OpenAI API key
*   `ANTHROPIC_API_KEY`: Anthropic API key
*   `OLLAMA_HOST`: Ollama service host (set to 0.0.0.0:11434 by default)
*   `PYTHONPATH`: Python path (set to /app by default)
*   `PYTHONUNBUFFERED`: Unbuffered output (set to 1 by default)

#### Tips and Advanced Usage

1.  **Ollama Models:** The container will automatically pull models on first use (e.g., `llama3.2:3b`).
2.  **Docker Model Runner:** Ensure Docker is running and accessible from within the container.
3.  **API Keys:** Pass API keys as environment variables for security.
4.  **Volumes:** Mount volumes for persistent workspaces or logs.
5.  **Networking:** Use `--network host` if you need to access local services.

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

Use the CLI to run Minion/Minions through the command line.  Configure local and remote models using the following command, with `<provider>/<model_name>` format:

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

The Secure Minions Local-Remote Protocol (`secure/minions_secure.py`) provides end-to-end encrypted communication for the Minions protocol, ensuring secure interaction between a local worker model and a remote supervisor server.  This includes attestation verification, perfect forward secrecy, and replay protection.

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

For setup and usage instructions, install the package with:

```bash
pip install -e ".[secure]"
```

And then see the [Secure Minions Chat README](secure/README.md).

## Apps

Explore a variety of specialized applications within the `apps/` directory:

*   📊 **[A2A-Minions](apps/minions-a2a/)** - Agent-to-Agent integration server
*   🎭 **[Character Chat](apps/minions-character-chat/)** - Role-playing with AI personas
*   🔍 **[Document Search](apps/minions-doc-search/)** - Multi-method document retrieval
*   📚 **[Story Teller](apps/minions-story-teller/)** - Creative storytelling with illustrations
*   🛠️ **[Tools Comparison](apps/minions-tools/)** - MCP tools performance comparison
*   🌐 **[WebGPU App](apps/minions-webgpu/)** - Browser-based Minions protocol

## Inference Estimator

The Minions Inference Estimator allows you to estimate LLM inference speed on your hardware.

### Command Line Usage

```bash
python -m minions.utils.inference_estimator --model llama3.2 --tokens 1000 --describe
```

Arguments:

*   `--model`: Model name (e.g., llama3.2, mistral7b)
*   `--tokens`: Number of tokens to estimate generation time for
*   `--describe`: Show detailed hardware and model performance statistics
*   `--quantized`: Specify that the model is quantized
*   `--quant-bits`: Quantization bit-width (4, 8, or 16)

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

The estimator uses a roofline model and empirical calibration for improved accuracy. Calibration data is cached at `~/.cache/ie_calib.json`.

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

*   Avanika Narayan (contact: avanika@cs.stanford.edu)
*   Dan Biderman (contact: biderman@stanford.edu)
*   Sabri Eyuboglu (contact: eyuboglu@cs.stanford.edu)