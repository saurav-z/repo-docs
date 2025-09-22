![Minions Logo](assets/Ollama_minionS_background.png)

# Minions: Supercharge LLMs with On-Device Power and Cloud Intelligence

**Unlock cost-efficient and high-quality language model interactions by combining the strengths of on-device and cloud-based AI models.**  Explore how Minions enables collaborative workflows for faster, more private, and more accessible LLM experiences.  [Visit the original Minions repository on GitHub](https://github.com/HazyResearch/minions).

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/jfJyxXwFVa)

## Key Features

*   **Hybrid LLM Architecture:** Leverage the benefits of both on-device and cloud-based language models.
*   **Cost-Effective:** Reduce cloud costs by executing long contexts locally.
*   **Privacy-Focused:** Process data locally for enhanced privacy.
*   **Versatile Protocol:** Support diverse use cases, from single-agent to multi-agent workflows.
*   **Secure Communication:** Utilize end-to-end encryption with the Secure Minions Protocol.
*   **Browser-Based Demo:** Explore Minions directly in your browser with the WebGPU app.

## Quickstart

### Prerequisites

*   Python 3.10-3.11 (Python 3.13 is not supported)

### Setup Steps

1.  **Clone and Install:**

    ```bash
    git clone https://github.com/HazyResearch/minions.git
    cd minions
    pip install -e .
    ```
    *For optional features, use `pip install -e ".[mlx]"` for MLX-LM or `pip install -e ".[secure]"` for Secure Minions Chat.*

2.  **Install a Local Model Server:**  Choose *one* of the following:

    *   **Ollama:** Best if you don't have NVIDIA/AMD GPUs. Follow installation instructions [here](https://ollama.com/download).  Enable Flash Attention with `launchctl setenv OLLAMA_FLASH_ATTENTION 1` and restart Ollama.
    *   **Lemonade:** (AMD CPUs/GPUs/NPUs). Follow instructions [here](https://lemonade-server.ai/).  Launch the Lemonade server after installation. *Note:  Lemonade does not support Minion-CUA protocol.*
    *   **Tokasaurus:** (NVIDIA GPUs).  Install with `pip install tokasaurus`.

    *Optional: Install Cartesia-MLX on Apple Silicon following the instructions in the original README.*
    *Optional: Install llama-cpp-python for additional model support; see details in the original README.*

3.  **Set Cloud LLM API Keys:**  Configure API keys for at least one cloud provider:

    ```bash
    # OpenAI
    export OPENAI_API_KEY=<your-openai-api-key>
    export OPENAI_BASE_URL=<your-openai-base-url> # Optional

    # Together AI
    export TOGETHER_API_KEY=<your-together-api-key>

    # OpenRouter
    export OPENROUTER_API_KEY=<your-openrouter-api-key>
    export OPENROUTER_BASE_URL=<your-openrouter-base-url> # Optional

    # Perplexity
    export PERPLEXITY_API_KEY=<your-perplexity-api-key>
    export PERPLEXITY_BASE_URL=<your-perplexity-base-url> # Optional

    # Tokasaurus
    export TOKASAURUS_BASE_URL=<your-tokasaurus-base-url> # Optional

    # DeepSeek
    export DEEPSEEK_API_KEY=<your-deepseek-api-key>

    # Anthropic
    export ANTHROPIC_API_KEY=<your-anthropic-api-key>

    # Mistral AI
    export MISTRAL_API_KEY=<your-mistral-api-key>
    ```

## Demo Applications

*   **Minions Demo Application:** Run the example streamlit app for Minion and Minions protocol.

    ```bash
    pip install torch transformers streamlit
    streamlit run app.py
    ```
*   **Minions WebGPU App:**  Experience the Minions protocol in your browser.

    ```bash
    cd apps/minions-webgpu
    npm install
    npm start
    ```

    (Navigate to the provided URL in your browser.)

## Code Examples

Explore the practical implementation of the Minions protocol:

### Minion (Singular)

```python
# Example using Ollama (local) and OpenAI (remote)
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minion import Minion

local_client = OllamaClient(model_name="llama3.2")
remote_client = OpenAIClient(model_name="gpt-4o")
minion = Minion(local_client, remote_client)

context = """... Patient data ..."""
task = "Evaluate the patient's risk..."

output = minion(task=task, context=[context], max_rounds=2)
```

### Minions (Plural)

```python
# Example using Ollama (local) and OpenAI (remote)
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
context = """... Patient data ..."""
task = "Evaluate the patient's risk..."

output = minion(task=task, doc_metadata="Medical Report", context=[context], max_rounds=2)
```

## Resources

*   **Python Notebook:** `minions.ipynb` provides a notebook environment for experimentation.
*   **Docker Support:** Build, run, and interact with Minions using Docker.
*   **CLI:** Use the command-line interface for streamlined interaction.
*   **Secure Minions Local-Remote Protocol:** End-to-end encrypted communication (see `secure/README.md`).
*   **Secure Minions Chat:** Secure and encrypted chat system (see `secure/README.md`).

## Apps

Explore various applications demonstrating the power of Minions:

*   A2A-Minions (Agent-to-Agent integration)
*   Character Chat (Role-playing with AI personas)
*   Document Search (Multi-method document retrieval)
*   Story Teller (Creative storytelling with illustrations)
*   Tools Comparison (MCP tools performance comparison)
*   WebGPU App (Browser-based Minions protocol)

## Inference Estimator

Estimate LLM inference speed on your hardware using the utility.

### Command Line

```bash
python -m minions.utils.inference_estimator --model llama3.2 --tokens 1000 --describe
```

### Python API

```python
from minions.utils.inference_estimator import InferenceEstimator

estimator = InferenceEstimator(model_name="llama3.2", is_quant=True, quant_bits=4)
tokens_per_second, estimated_time = estimator.estimate(1000)
```

## Miscellaneous Setup

### Using Azure OpenAI

*   Set environment variables:

    ```bash
    export AZURE_OPENAI_API_KEY=your-api-key
    export AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
    export AZURE_OPENAI_API_VERSION=2024-02-15-preview
    ```

*   Example code:

    ```python
    from minions.clients.ollama import OllamaClient
    from minions.clients.azure_openai import AzureOpenAIClient
    from minions.minion import Minion

    local_client = OllamaClient(model_name="llama3.2")
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