# Agent S: Revolutionizing Computer Use with Autonomous AI Agents

> **Agent S** is an open-source framework that empowers AI agents to interact with computers, providing a new state-of-the-art approach to computer use agents.

[![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/E2XfsK9fPV?style=flat)](https://discord.gg/E2XfsK9fPV)
[![PyPI Downloads](https://static.pepy.tech/badge/gui-agents)](https://pepy.tech/projects/gui-agents)

**Key Features:**

*   **Autonomous Computer Interaction:** Enables AI agents to perform complex tasks on your computer.
*   **Compositional Generalist-Specialist Framework:**  A robust architecture designed for effective task execution.
*   **State-of-the-Art Performance:** Outperforms existing solutions in benchmarks like OSWorld, WindowsAgentArena, and AndroidWorld.
*   **Open Source & Accessible:**  Easily integrate and contribute to cutting-edge agent-based systems.
*   **Modular Design:** Supports various LLMs, visual grounding models, and platforms.
*   **Integration with Perplexica:** Web knowledge retrieval for enhanced reasoning capabilities.

**[View the Original Repository on GitHub](https://github.com/simular-ai/Agent-S)**

## 🚀 Key Updates

*   **[2025/07/07]:**  The Agent S2 paper accepted to COLM 2025!
*   **[2025/04/01]:**  Released Agent S2 paper with improved SOTA results on OSWorld, WindowsAgentArena, and AndroidWorld.
*   **[2025/03/12]:**  Released Agent S2 and gui-agents v0.2.0, setting a new SOTA for computer use agents.
*   **[2025/01/22]:**  Agent S paper accepted to ICLR 2025!
*   **[2025/01/21]:**  Released gui-agents v0.1.2, adding support for Linux and Windows.
*   **[2024/12/05]:**  Released gui-agents v0.1.0, supporting Mac, OSWorld, and WindowsAgentArena.
*   **[2024/10/10]:** Released Agent S paper and codebase.

## 💻 Current Results

Agent S2 showcases impressive results, including significant gains in key benchmarks:

<p align="center">
    <img src="./images/agent_s2_osworld_result.png" width="600">
    <br>
    Results of Agent S2's Successful Rate (%) on the OSWorld full test set using Screenshot input only.
</p>

<div align="center">
  <table border="0" cellspacing="0" cellpadding="5">
    <tr>
      <th>Benchmark</th>
      <th>Agent S2</th>
      <th>Previous SOTA</th>
      <th>Δ improve</th>
    </tr>
    <tr>
      <td>OSWorld (15 step)</td>
      <td>27.0%</td>
      <td>22.7% (UI-TARS)</td>
      <td>+4.3%</td>
    </tr>
    <tr>
      <td>OSWorld (50 step)</td>
      <td>34.5%</td>
      <td>32.6% (OpenAI CUA)</td>
      <td>+1.9%</td>
    </tr>
    <tr>
      <td>WindowsAgentArena</td>
      <td>29.8%</td>
      <td>19.5% (NAVI)</td>
      <td>+10.3%</td>
    </tr>
    <tr>
      <td>AndroidWorld</td>
      <td>54.3%</td>
      <td>46.8% (UI-TARS)</td>
      <td>+7.5%</td>
    </tr>
  </table>
</div>

## ⚙️ Installation and Setup

> **Important Notes:**

> *   Our agent returns `pyautogui` code and is intended for a single monitor screen.
> *   On Linux, avoid using `conda` for the initial setup, as it may interfere with `pyatspi`.

> **Disclaimer:**  Agent S2 utilizes [UI-TARS](https://github.com/bytedance/UI-TARS) as a grounding model for optimal results, which can be hosted locally or on Hugging Face Inference Endpoints.

Install the package:

```bash
pip install gui-agents
```

Set your API keys and environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`).

```bash
export OPENAI_API_KEY=<YOUR_API_KEY>
export ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
```

Or set them directly in your Python script:

```python
import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
```

Supports Azure OpenAI, Anthropic, Gemini, Open Router, and vLLM inference (see [models.md](models.md)).

### 🌐 Setting Up Web Knowledge Retrieval with Perplexica

Agent S benefits from web-knowledge retrieval.  Here's how to set up Perplexica:

1.  Ensure Docker Desktop is installed and running.
2.  Navigate to the project directory:

    ```bash
    cd Perplexica
    git submodule update --init
    ```

3.  Rename `sample.config.toml` to `config.toml`. Fill in the necessary API keys (e.g.  `OPENAI`,  `OLLAMA`, `GROQ`, `ANTHROPIC`):

    *   `OPENAI`: Your OpenAI API key.
    *   `OLLAMA`:  Your Ollama API URL (e.g., `http://host.docker.internal:11434`).
    *   `GROQ`: Your Groq API key.
    *   `ANTHROPIC`: Your Anthropic API key.

    **Note**:  You can change these settings after starting Perplexica from the settings dialog.

    *   `SIMILARITY_MEASURE`:  (Leave this field as is.)

4.  Run Docker Compose:

    ```bash
    docker compose up -d
    ```

5.  Export the Perplexica URL (using the port from `docker-compose.yaml`):

    ```bash
    export PERPLEXICA_URL=http://localhost:{port}/api/search
    ```

6.  Customize the Perplexica API if needed (modify `agent_s/query_perplexica.py`). See the [Perplexica Search API Documentation](https://github.com/ItzCrazyKns/Perplexica/blob/master/docs/API/SEARCH.md).

> **Warning:** The agent executes Python code to control your computer; use with caution.

## 🚀 Usage

> **Note:** The best performance is achieved using Claude 3.7 with extended thinking and UI-TARS-72B-DPO.  If resource constraints prevent UI-TARS-72B-DPO, UI-TARS-7B-DPO is a viable alternative.

### Command Line Interface (CLI)

Run Agent S2 (defaults to `gpt-4o`):

```bash
agent_s2 \
  --provider "anthropic" \
  --model "claude-3-7-sonnet-20250219" \
  --grounding_model_provider "anthropic" \
  --grounding_model "claude-3-7-sonnet-20250219"
```

Or use a custom endpoint:

```bash
agent_s2 \
  --provider "anthropic" \
  --model "claude-3-7-sonnet-20250219" \
  --endpoint_provider "huggingface" \
  --endpoint_url "<endpoint_url>/v1/"
```

#### Main Model Settings

*   `--provider`, `--model`: Specifies the main generation model. Supported models are listed in [models.md](models.md) (default: `--provider "anthropic" --model "claude-3-7-sonnet-20250219"`).
*   `--model_url`, `--model_api_key`: Specifies a custom endpoint for the generation model and your API key.  Optional; defaults to environment variables.  Supports models in [models.md](models.md).

#### Grounding Configuration Options

You can use either Configuration 1 or Configuration 2:

##### Configuration 1: API-Based Models (Default)
*   `--grounding_model_provider`, `--grounding_model`: Specifies the visual grounding model. Supported providers in [models.md](models.md) (default: `--grounding_model_provider "anthropic" --grounding_model "claude-3-7-sonnet-20250219"`).
*   `--grounding_model_resize_width`: **Important**:  Correctly set for accurate grounding; required when using providers like Anthropic that automatically resize images.  (default: `--grounding_model_resize_width 1366` for Anthropic)

##### Configuration 2: Custom Endpoint
*   `--endpoint_provider`: Specifies the endpoint provider (HuggingFace TGI, vLLM, Open Router).
*   `--endpoint_url`: The URL for your custom endpoint.
*   `--endpoint_api_key`: Your API key for your custom endpoint (optional; defaults to environment variables).

> **Note**: Configuration 2 takes precedence over Configuration 1.

This will display a prompt, allowing you to interact with Agent S2.

### `gui_agents` SDK

```python
import pyautogui
import io
from gui_agents.s2.agents.agent_s import AgentS2
from gui_agents.s2.agents.grounding import OSWorldACI

# Load in your API keys.
from dotenv import load_dotenv
load_dotenv()

current_platform = "linux"  # "darwin", "windows"

# Engine parameters for generation (main agent)
engine_params = {
  "engine_type": provider,
  "model": model,
  "base_url": model_url,     # Optional
  "api_key": model_api_key,  # Optional
}

# Engine parameters for grounding configuration 1
grounding_model_provider = "<your_grounding_model_provider>"
grounding_model = "<your_grounding_model>"
grounding_model_resize_width = 1366
screen_width, screen_height = pyautogui.size()

engine_params_for_grounding = {
  "engine_type": grounding_model_provider,
  "model": grounding_model,
  "grounding_width": grounding_model_resize_width,
  "grounding_height": screen_height
  * grounding_model_resize_width
  / screen_width,
}

# Engine parameters for grounding configuration 2
endpoint_provider = "<your_endpoint_provider>"
endpoint_url = "<your_endpoint_url>"
endpoint_api_key = "<your_api_key>"

engine_params_for_grounding = {
  "engine_type": endpoint_provider,
  "base_url": endpoint_url,
  "api_key": endpoint_api_key,  # Optional
}

# Create the grounding agent and Agent S2.
grounding_agent = OSWorldACI(
    platform=current_platform,
    engine_params_for_generation=engine_params,
    engine_params_for_grounding=engine_params_for_grounding
)

agent = AgentS2(
  engine_params,
  grounding_agent,
  platform=current_platform,
  action_space="pyautogui",
  observation_type="screenshot",
  search_engine="Perplexica",  # Assuming you have set up Perplexica.
  embedding_engine_type="openai"  # Supports "gemini", "openai"
)

# Get screenshot.
screenshot = pyautogui.screenshot()
buffered = io.BytesIO()
screenshot.save(buffered, format="PNG")
screenshot_bytes = buffered.getvalue()

obs = {
  "screenshot": screenshot_bytes,
}

instruction = "Close VS Code"
info, action = agent.predict(instruction=instruction, observation=obs)

exec(action[0])
```

Refer to `gui_agents/s2/cli_app.py` for detailed inference loop information.

#### Downloading the Knowledge Base

The knowledge base is downloaded when initializing `AgentS2`.

```python
download_kb_data(
    version="s2",
    release_tag="v0.2.2",
    download_dir="kb_data",
    platform="linux"  # "darwin", "windows"
)
```

## 🌐 Deployment Guides

*   **OSWorld:** See [OSWorld Deployment Instructions](osworld_setup/s2/OSWorld.md).
*   **WindowsAgentArena:** See [WindowsAgentArena Deployment Instructions](WAA_setup.md).

## 💬 Citations

If you use this code, please cite the following:

```
@misc{Agent-S2,
      title={Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents},
      author={Saaket Agashe and Kyle Wong and Vincent Tu and Jiachen Yang and Ang Li and Xin Eric Wang},
      year={2025},
      eprint={2504.00906},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.00906},
}

@inproceedings{Agent-S,
    title={{Agent S: An Open Agentic Framework that Uses Computers Like a Human}},
    author={Saaket Agashe and Jiuzhou Han and Shuyu Gan and Jiachen Yang and Ang Li and Xin Eric Wang},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2025},
    url={https://arxiv.org/abs/2410.08164}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=simular-ai/Agent-S&type=Date)](https://www.star-history.com/#agent-s/agent-s&simular-ai/Agent-S&Date)