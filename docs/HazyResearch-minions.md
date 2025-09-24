<!-- Minions Logo -->
<img src="assets/Ollama_minionS_background.png" alt="Minions Logo" width="300">

# Minions: Revolutionizing LLM Collaboration with On-Device and Cloud Models

**Harness the power of local and cloud language models with Minions, a communication protocol designed for cost-effective and high-quality AI experiences.** Learn more and explore the code on [GitHub](https://github.com/HazyResearch/minions).

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/jfJyxXwFVa)

**Key Features:**

*   **Cost-Effective AI:** Minimize cloud costs by leveraging on-device models for long context processing.
*   **Enhanced Privacy:** Process sensitive data locally while utilizing cloud models for complex tasks.
*   **Flexible Architecture:** Compatible with various local and cloud LLMs.
*   **Secure Communication:** Implement end-to-end encryption for secure data exchange using the Secure Minions Protocol (See below).
*   **Browser-Based Demo:**  A fully functional web app showcasing the Minions protocol, running entirely in your browser.
*   **Multi-Protocol Support:** Supports `minion` (single agent) and `minions` (parallel agents) protocols.
*   **Comprehensive Tooling:** Includes a CLI, Docker support, and an Inference Estimator for optimizing model performance.

**Jump to:**

*   [Setup](#setup)
*   [Minions Demo Application](#minions-demo-application)
*   [WebGPU App](#minions-webgpu-app)
*   [Example Code](#example-code)
*   [Secure Minions Chat](#secure-minions-chat)
*   [Docker Support](#docker-support)
*   [CLI](#cli)
*   [Inference Estimator](#inference-estimator)

## Setup

Get started with Minions in a few simple steps.  We recommend using Python 3.10-3.11. Python 3.13 is not supported.

**Prerequisites:** Ensure you have Python installed.  Optional: Create a virtual environment.

```bash
# Example using conda
conda create -n minions python=3.11
conda activate minions
```

**Step 1: Clone and Install**

```bash
git clone https://github.com/HazyResearch/minions.git
cd minions
pip install -e .  # Installs the minions package in editable mode
```

**Optional Installations:**

*   For optional MLX-LM:
    ```bash
    pip install -e ".[mlx]"
    ```
*   For Secure Minions Chat:
    ```bash
    pip install -e ".[secure]"
    ```

**Step 2: Install a Local Model Server**

Choose *one* of the following options for running a local model server:

*   **Ollama:** Recommended if you do not have access to NVIDIA/AMD GPUs. Install [Ollama](https://ollama.com/download). To enable Flash Attention, run `launchctl setenv OLLAMA_FLASH_ATTENTION 1` and restart the Ollama app (macOS).
*   **Lemonade:** If you have access to local AMD CPUs/GPUs/NPUs. Install [Lemonade](https://lemonade-server.ai/).  Launch the Lemonade server after installation. Note: Lemonade currently does not support the Minion-CUA protocol.
*   **Tokasaurus:** If you have access to NVIDIA GPUs. Install with: `pip install tokasaurus`

**Optional: Install Cartesia-MLX (Apple Silicon only):** Follow the detailed instructions in the original README.

**Optional: Install llama-cpp-python:** Follow the detailed instructions in the original README.

**Step 3: Set Cloud LLM API Keys**

Set your API key for *at least one* of the following cloud LLM providers:

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

<a href="https://www.youtube.com/watch?v=70Kot0_DFNs" target="_blank"><img src="https://img.youtube.com/vi/70Kot0_DFNs/0.jpg" alt="Minions Demo Video" width="480"></a>

Quickly experience the Minion or Minions protocol with the demo application:

```bash
pip install torch transformers
streamlit run app.py
```

If you encounter an Ollama connection error, try running:  `OLLAMA_FLASH_ATTENTION=1 ollama serve`

## Minions WebGPU App

Explore the Minions protocol directly in your browser with the WebGPU app. This demo eliminates local server dependencies, providing a streamlined user experience.

### Features

*   **Browser-Based:** Runs entirely in your browser.
*   **WebGPU Acceleration:**  Fast local model inference.
*   **Model Selection:** Choose from pre-optimized models from [MLC AI](https://mlc.ai/models).
*   **Real-Time Progress:** See model loading and conversation logs.
*   **Privacy-Focused:**  API keys and data remain within your browser.

### Quick Start

1.  **Navigate:** `cd apps/minions-webgpu`
2.  **Install:** `npm install`
3.  **Run:** `npm start`
4.  **Open Browser:**  Navigate to the provided URL (e.g., `http://localhost:5173`).

## Example Code

Here's how to use `Minion` (singular) and `Minions` (plural) protocols.

**Example: Minion (Singular)**

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

task = "Based on the patient's blood pressure and LDL cholesterol..."

output = minion(
    task=task,
    context=[context],
    max_rounds=2
)
```

**Example: Minions (Plural)**

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

task = "Based on the patient's blood pressure and LDL cholesterol..."

output = minion(
    task=task,
    doc_metadata="Medical Report",
    context=[context],
    max_rounds=2
)
```

## Python Notebook

Explore Minion/Minions in a Python notebook; see `minions.ipynb`.

## Docker Support

Minions offers Docker support for easy deployment and testing.

### Build

```bash
docker build -t minions-docker .
```

### Run

The container automatically runs an Ollama service.

```bash
docker run -i minions-docker  # Basic usage

# With Docker socket mounted
docker run -i -v /var/run/docker.sock:/var/run/docker.sock minions-docker

# With API keys
docker run -i -e OPENAI_API_KEY=your_key -e ANTHROPIC_API_KEY=your_key minions-docker

# With custom Ollama host
docker run -i -e OLLAMA_HOST=0.0.0.0:11434 minions-docker

# For Streamlit app (legacy)
docker run -p 8501:8501 --env OPENAI_API_KEY=<your-openai-api-key> --env DEEPSEEK_API_KEY=<your-deepseek-api-key> minions-docker
```

### Docker Minion Protocol Usage

Use a JSON input structure to run the minion protocols within the Docker container via stdin/stdout.

```json
{
  "local_client": { ... },
  "remote_client": { ... },
  "protocol": { ... },
  "call_params": { ... }
}
```

**Examples**  (See original README for detailed JSON examples.)

```bash
# Basic Minion
echo '{...}' | docker run -i -e OPENAI_API_KEY=$OPENAI_API_KEY minions-docker

# Minions (Parallel)
echo '{...}' | docker run -i -e OPENAI_API_KEY=$OPENAI_API_KEY minions-docker
```

**See original README for detailed information on Client Types, Protocol Types, Output Format, and Environment Variables.**

## CLI

Use the command-line interface for interacting with the Minions.

Set Local/Remote Model:

```bash
export MINIONS_LOCAL=ollama/llama3.2
export MINIONS_REMOTE=openai/gpt-4o
```

Get help:

```bash
minions --help
```

Run:

```bash
minions --context <path_to_context> --protocol <minion|minions>
```

## Secure Minions Chat

**For end-to-end encrypted Minions Chat, see the [Secure Minions Chat README](secure/README.md).**

```bash
pip install -e ".[secure]"
```

The Secure Minions Local-Remote Protocol provides end-to-end encryption and secure communication. See the original README for detailed instructions.

## Inference Estimator

Use the Inference Estimator to estimate LLM inference speed.

### Command Line

```bash
python -m minions.utils.inference_estimator --model llama3.2 --tokens 1000 --describe
```

### Python API

```python
from minions.utils.inference_estimator import InferenceEstimator

estimator = InferenceEstimator(model_name="llama3.2")
tokens_per_second, estimated_time = estimator.estimate(1000)
```

## Miscellaneous Setup

### Using Azure OpenAI with Minions

1.  Set Environment Variables:

```bash
export AZURE_OPENAI_API_KEY=your-api-key
export AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
export AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

2.  Example Code:  See original README.

## Maintainers

*   Avanika Narayan (avanika@cs.stanford.edu)
*   Dan Biderman (biderman@stanford.edu)
*   Sabri Eyuboglu (eyuboglu@cs.stanford.edu)