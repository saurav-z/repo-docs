<h1 align="center">
  <img src="images/agent_s.png" alt="Agent S Logo" style="vertical-align:middle" width="60"> Agent S: Empowering Intelligent Computer Interaction
</h1>

<p align="center">
  <a href="https://github.com/simular-ai/Agent-S">
    <img src="https://img.shields.io/github/stars/simular-ai/Agent-S?style=social" alt="GitHub Stars">
  </a>
</p>

<p align="center">
  <a href="https://www.simular.ai/articles/agent-s2-technical-review">[S2 Blog]</a>&nbsp;
  <a href="https://arxiv.org/abs/2504.00906">[S2 Paper (COLM 2025)]</a>&nbsp;
  <a href="https://www.youtube.com/watch?v=wUGVQl7c0eg">[S2 Video]</a>
</p>

<p align="center">
  <a href="https://www.simular.ai/agent-s">[S1 Blog]</a>&nbsp;
  <a href="https://arxiv.org/abs/2410.08164">[S1 Paper (ICLR 2025)]</a>&nbsp;
  <a href="https://www.youtube.com/watch?v=OBDE3Knte0g">[S1 Video]</a>
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/13151" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/13151" alt="simular-ai%2FAgent-S | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</p>

<p align="center">
  <a href="https://discord.gg/E2XfsK9fPV">
    <img src="https://dcbadge.limes.pink/api/server/https://discord.gg/E2XfsK9fPV?style=flat" alt="Discord">
  </a>
  &nbsp;&nbsp;
  <a href="https://pepy.tech/projects/gui-agents">
    <img src="https://static.pepy.tech/badge/gui-agents" alt="PyPI Downloads">
  </a>
</p>

<div align="center">
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/simular-ai/Agent-S?lang=de">Deutsch</a> |
  <a href="https://www.readme-i18n.com/simular-ai/Agent-S?lang=es">Español</a> |
  <a href="https://www.readme-i18n.com/simular-ai/Agent-S?lang=fr">français</a> |
  <a href="https://www.readme-i18n.com/simular-ai/Agent-S?lang=ja">日本語</a> |
  <a href="https://www.readme-i18n.com/simular-ai/Agent-S?lang=ko">한국어</a> |
  <a href="https://www.readme-i18n.com/simular-ai/Agent-S?lang=pt">Português</a> |
  <a href="https://www.readme-i18n.com/simular-ai/Agent-S?lang=ru">Русский</a> |
  <a href="https://www.readme-i18n.com/simular-ai/Agent-S?lang=zh">中文</a>
</div>

## 🚀 Agent S: The Future of Computer Automation

**Agent S** is a powerful open-source framework designed to revolutionize computer interaction, enabling autonomous agents to perform complex tasks with unprecedented accuracy and efficiency. [Explore the Agent S repository on GitHub](https://github.com/simular-ai/Agent-S).

**Key Features:**

*   **Agent S2 Advancements**: Leveraging the compositional generalist-specialist framework, Agent S2 sets a new standard in Computer Use Agents (CUA).
*   **State-of-the-Art Performance:** Outperforms competitors in complex scenarios, including OSWorld, WindowsAgentArena, and AndroidWorld.
*   **Open-Source & Accessible:** Built for researchers, developers, and AI enthusiasts alike, offering flexibility and community support.
*   **GUI Interaction:** The agent directly controls the computer.
*   **Modular Design:** Offers modular design for easy customization and integration.

## ✨ Highlights

*   **Agent S2 Paper Accepted**: The [Agent S2 paper](https://arxiv.org/abs/2504.00906) accepted at COLM 2025!
*   **Cutting-Edge Results**: Agent S2 achieved new state-of-the-art results in OSWorld, WindowsAgentArena, and AndroidWorld.
*   **GUI Agents Library**:  v0.2.0 of [gui-agents](https://github.com/simular-ai/Agent-S) library released with a new state-of-the-art for computer use agents (CUA).
*   **Agent S Paper Accepted:** The [Agent S paper](https://arxiv.org/abs/2410.08164) is accepted to ICLR 2025!
*   **Cross-Platform Support**: Linux and Windows support added.
*   **User-Friendly**:  Easy to use with intuitive Python SDK and CLI interfaces.

## 📊 Performance & Results

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

## 🛠️ Installation & Setup

> **Important Note:**  The agent returns `pyautogui` code and is designed for single-monitor setups.

> **Warning:** On Linux, creating a `conda` environment can interfere with `pyatspi`. Proceed without a virtual environment in that case.

> **Disclaimer**: Agent S2 utilizes [UI-TARS](https://github.com/bytedance/UI-TARS) as a grounding model (7B-DPO or 72B-DPO). These can be hosted locally or with Hugging Face Inference Endpoints. Our code supports HF Inference Endpoints.

Install the package:

```bash
pip install gui-agents
```

Set your LLM API Keys and environment variables. This can be done in your `.bashrc` (Linux) or `.zshrc` (MacOS) file:

```bash
export OPENAI_API_KEY=<YOUR_API_KEY>
export ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
```

Alternatively, set them within your Python script:

```python
import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
```

### Web Knowledge Retrieval (Perplexica Setup)

Agent S uses Perplexica for web-knowledge retrieval. Follow these steps to configure Perplexica:

1.  Install and run Docker Desktop.
2.  Navigate to your project directory (Perplexica directory), run `git submodule update --init`.
3.  Rename `sample.config.toml` to `config.toml`.  For Docker, fill in:

    *   `OPENAI`: Your OpenAI API key (if using OpenAI models).
    *   `OLLAMA`: Your Ollama API URL, e.g., `http://host.docker.internal:11434` (if using Ollama).
    *   `GROQ`: Your Groq API key (if using Groq models).
    *   `ANTHROPIC`: Your Anthropic API key (if using Anthropic models).
    *   `SIMILARITY_MEASURE`: (Default setting).
4.  Run `docker compose up -d`.
5.  Export the Perplexica URL (use the port from the `docker-compose.yaml` file):

    ```bash
    export PERPLEXICA_URL=http://localhost:{port}/api/search
    ```
6.  Modify the URL and message of request parameters in  `agent_s/query_perplexica.py` for customization. See the [Perplexica Search API Documentation](https://github.com/ItzCrazyKns/Perplexica/blob/master/docs/API/SEARCH.md). For more details, see the [Perplexica Repository](https://github.com/ItzCrazyKns/Perplexica.git).

> **Warning:** The agent runs Python code to control your computer. Use with caution.

## 🚀 Usage

> **Recommended**: Use Claude 3.7 with extended thinking and UI-TARS-72B-DPO. UI-TARS-7B-DPO can be used as a lighter alternative.

### CLI

Run Agent S2 with your desired model (defaults to `gpt-4o`):

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

*   **`--provider`**, **`--model`**: Specifies the main generation model (supports all providers listed in [models.md](models.md)). Defaults to `--provider "anthropic" --model "claude-3-7-sonnet-20250219"`.
*   **`--model_url`**, **`--model_api_key`**:  Specifies a custom endpoint and API key for the main generation model.  (Optional; defaults to environment variables).

#### Grounding Configuration Options

Choose either Configuration 1 or Configuration 2:

##### **Configuration 1: API-Based Models (Default)**

*   **`--grounding_model_provider`**, **`--grounding_model`**: Specifies the model for visual grounding. Supports all providers in [models.md](models.md).  Defaults to `--grounding_model_provider "anthropic" --grounding_model "claude-3-7-sonnet-20250219"`.
*   **`--grounding_model_resize_width`**: **Important:** For API providers that auto-rescale images,  adjust this parameter. Use for Anthropic rescaling. Default: `--grounding_model_resize_width 1366` (Anthropic).

##### **Configuration 2: Custom Endpoint**

*   **`--endpoint_provider`**: The endpoint provider. Supports HuggingFace TGI, vLLM, and Open Router.
*   **`--endpoint_url`**: The URL for your custom endpoint.
*   **`--endpoint_api_key`**: Your API key for the endpoint. (Optional; defaults to environment variables).

> **Note**: Configuration 2 overrides Configuration 1.

The CLI will prompt you for queries.

### `gui_agents` SDK

Import modules:

```python
import pyautogui
import io
from gui_agents.s2.agents.agent_s import AgentS2
from gui_agents.s2.agents.grounding import OSWorldACI

# Load in your API keys.
from dotenv import load_dotenv
load_dotenv()

current_platform = "linux"  # "darwin", "windows"
```

Define engine parameters:

```python
# Main Agent parameters
engine_params = {
  "engine_type": provider,
  "model": model,
  "base_url": model_url,     # Optional
  "api_key": model_api_key,  # Optional
}

# Grounding Configuration 1: API-Based Models
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

# Grounding Configuration 2: Custom Endpoint
endpoint_provider = "<your_endpoint_provider>"
endpoint_url = "<your_endpoint_url>"
endpoint_api_key = "<your_api_key>"

engine_params_for_grounding = {
  "engine_type": endpoint_provider,
  "base_url": endpoint_url,
  "api_key": endpoint_api_key,  # Optional
}
```

Create grounding and Agent S2 instances:

```python
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
  search_engine="Perplexica",  # Requires Perplexica setup.
  embedding_engine_type="openai"  # Supports "gemini", "openai"
)
```

Make predictions:

```python
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

See `gui_agents/s2/cli_app.py` for inference loop details.

#### Knowledge Base Download

Agent S2 uses a knowledge base downloaded at initialization. Download programmatically using:

```python
from gui_agents.s2.download_kb import download_kb_data

download_kb_data(
    version="s2",
    release_tag="v0.2.2",
    download_dir="kb_data",
    platform="linux"  # "darwin", "windows"
)
```

Find knowledge bases on our [GitHub Releases](https://github.com/simular-ai/Agent-S/releases).

### OSWorld & WindowsAgentArena

Deploy Agent S2 in OSWorld and WindowsAgentArena using the respective deployment instructions: [OSWorld Deployment Instructions](osworld_setup/s2/OSWorld.md) and [WindowsAgentArena Deployment Instructions](WAA_setup.md).

## 🤝 Acknowledgements

We would like to acknowledge the contributions of the research community and the open-source community.

## 💬 Citations

```bibtex
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