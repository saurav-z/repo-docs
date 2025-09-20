<!-- Minions Logo -->
![Minions Logo](assets/Ollama_minionS_background.png)

# Minions: Unleash the Power of Hybrid LLMs

**Minions revolutionizes how you use Large Language Models (LLMs) by enabling cost-effective collaboration between on-device and cloud-based models.**  [Explore the Minions GitHub Repo](https://github.com/HazyResearch/minions)

## Key Features:

*   **Cost-Effective LLM Collaboration:**  Leverage the strengths of both on-device and cloud models to reduce costs.
*   **On-Device Privacy:**  Process long contexts locally, minimizing data transfer to the cloud.
*   **Flexible Protocol:** Integrates seamlessly with various local model servers (Ollama, Lemonade, Tokasaurus) and cloud providers.
*   **Secure Communication:**  Includes a secure protocol with attestation and replay protection for private interactions.
*   **Browser-Based Demo:** Experience the Minions protocol directly in your web browser with the WebGPU app.
*   **Versatile Applications:** Explore diverse use cases with pre-built apps for character chat, document search, and more.
*   **Easy Integration:**  Simple setup with clear examples, CLI, and Docker support.
*   **Inference Estimator:** Analyze your hardware capabilities and predict LLM inference speed.

## Getting Started

### Setup

Follow these steps to set up your Minions environment:

**Prerequisites:**  Python 3.10-3.11 (3.13 is not supported).  Create a virtual environment for best practices.

**Step 1: Clone and Install**

```bash
git clone https://github.com/HazyResearch/minions.git
cd minions
pip install -e .
```

**Optional Installations:**

*   For MLX-LM: `pip install -e ".[mlx]"`
*   For Secure Minions Chat: `pip install -e ".[secure]"`
*   For Cartesia-MLX: Follow the detailed instructions in the original README.

**Step 2: Install a Local Model Server**

Choose and install *at least one* of the following:

*   **Ollama:** (Recommended if you don't have a dedicated GPU)  Install from [Ollama's website](https://ollama.com/download).  Enable Flash Attention with `launchctl setenv OLLAMA_FLASH_ATTENTION 1` and restart the Ollama app on macOS.
*   **Lemonade:** (For AMD CPUs/GPUs/NPUs) Install from [Lemonade-server.ai](https://lemonade-server.ai/). Launch the server.  Note:  Does not support Minion-CUA.
*   **Tokasaurus:** (For NVIDIA GPUs - best performance for Minions protocol) `pip install tokasaurus`

**Optional: Install Llama-cpp-python for local model access:**
Install and configure as per instructions in original README.

**Step 3: Configure Cloud LLM API Keys**

Set your API key for *at least one* of the cloud providers.  Create API keys if you don't have them.

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

## Demo Applications

### Minions Demo

Quickly test the core Minions functionality:

```bash
pip install torch transformers
streamlit run app.py
```

Troubleshooting Ollama:  If you see an Ollama connection error, try `OLLAMA_FLASH_ATTENTION=1 ollama serve`.

### Minions WebGPU App

Experience Minions directly in your browser!

1.  `cd apps/minions-webgpu`
2.  `npm install`
3.  `npm start`
4.  Open the provided URL in your browser (usually `http://localhost:5173`).

## Code Examples

### Minion (Singular)

```python
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minion import Minion

local_client = OllamaClient(model_name="llama3.2")
remote_client = OpenAIClient(model_name="gpt-4o")
minion = Minion(local_client, remote_client)

context = """..."""
task = "Based on the patient's ... "

output = minion(task=task, context=[context], max_rounds=2)
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

local_client = OllamaClient(model_name="llama3.2", temperature=0.0, structured_output_schema=StructuredLocalOutput)
remote_client = OpenAIClient(model_name="gpt-4o")

minion = Minions(local_client, remote_client)

context = """..."""
task = "Based on the patient's ... "

output = minion(task=task, doc_metadata="Medical Report", context=[context], max_rounds=2)
```

## Additional Resources

*   **Python Notebook:**  Explore `minions.ipynb`.
*   **Docker Support:** Build and run Minions with Docker for easy deployment and testing.  See the original README for comprehensive Docker instructions.
*   **CLI:** Use the command-line interface for quick interactions. See `minions_cli.py`.
*   **Secure Minions Chat:** For a secure, end-to-end encrypted chat implementation, see the [Secure Minions Chat README](secure/README.md) .
*   **Apps:** Explore the `apps/` directory for specialized applications like A2A-Minions, Character Chat, and Document Search.
*   **Inference Estimator:** Measure LLM inference speed with the provided utility (see below).

## Inference Estimator

Calculate LLM inference speed:

### Command Line:

```bash
python -m minions.utils.inference_estimator --model llama3.2 --tokens 1000 --describe
```

### Python API:

```python
from minions.utils.inference_estimator import InferenceEstimator

estimator = InferenceEstimator(model_name="llama3.2", is_quant=True, quant_bits=4)
tokens_per_second, estimated_time = estimator.estimate(1000)
```

## Miscellaneous Setup

### Using Azure OpenAI:

```bash
export AZURE_OPENAI_API_KEY=your-api-key
export AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
export AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

See the original README for the example code.

## Maintainers

*   Avanika Narayan (avanika@cs.stanford.edu)
*   Dan Biderman (biderman@stanford.edu)
*   Sabri Eyuboglu (eyuboglu@cs.stanford.edu)