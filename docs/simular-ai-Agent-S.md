<h1 align="center">
  <img src="images/agent_s.png" alt="Agent S Logo" style="vertical-align:middle" width="60"> Agent S: Empowering Computers with Intelligent Agents
</h1>

<p align="center">
  <a href="https://github.com/simular-ai/Agent-S">
    <img src="https://img.shields.io/github/stars/simular-ai/Agent-S?style=social" alt="Stars">
  </a>
</p>

<p align="center">
  Agent S revolutionizes computer interaction, enabling autonomous task completion with cutting-edge AI.
</p>

<p align="center">&nbsp;
  🌐 <a href="https://www.simular.ai/articles/agent-s2-technical-review">[S2 blog]</a>&nbsp;
  📄 <a href="https://arxiv.org/abs/2504.00906">[S2 Paper (COLM 2025)]</a>&nbsp;
  🎥 <a href="https://www.youtube.com/watch?v=wUGVQl7c0eg">[S2 Video]</a>
</p>

<p align="center">&nbsp;
  🌐 <a href="https://www.simular.ai/agent-s">[S1 blog]</a>&nbsp;
  📄 <a href="https://arxiv.org/abs/2410.08164">[S1 Paper (ICLR 2025)]</a>&nbsp;
  🎥 <a href="https://www.youtube.com/watch?v=OBDE3Knte0g">[S1 Video]</a>
</p>

<p align="center">&nbsp;
<a href="https://trendshift.io/repositories/13151" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13151" alt="simular-ai%2FAgent-S | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
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

## Key Features

*   **Compositional Generalist-Specialist Framework:** Agent S employs a modular design for advanced computer use agents.
*   **State-of-the-Art Performance:** Agent S2 outperforms competitors in OSWorld, WindowsAgentArena, and AndroidWorld.
*   **Open-Source and Accessible:** Build and experiment with cutting-edge GUI agents.
*   **Integration with UI-TARS:** Leverages UI-TARS for robust visual grounding and improved accuracy.
*   **Supports Multiple Platforms:** Includes support for Mac, OSWorld, and Windows.
*   **Flexible API Support:** Integrates with OpenAI, Anthropic, Gemini, Open Router, and vLLM.
*   **Perplexica Web Retrieval:** Integrated with Perplexica for enhanced web-knowledge retrieval.

## 🚀 Updates

*   **2025/07/07:** [Agent S2 paper](https://arxiv.org/abs/2504.00906) accepted to COLM 2025!
*   **2025/04/01:** Released [Agent S2 paper](https://arxiv.org/abs/2504.00906) with SOTA results.
*   **2025/03/12:** Released Agent S2 and v0.2.0 of [gui-agents](https://github.com/simular-ai/Agent-S), setting a new standard for CUAs.
*   **2025/01/22:** [Agent S paper](https://arxiv.org/abs/2410.08164) accepted to ICLR 2025!
*   **2025/01/21:** Released v0.1.2 of [gui-agents](https://github.com/simular-ai/Agent-S), with Linux and Windows support.
*   **2024/12/05:** Released v0.1.0 of [gui-agents](https://github.com/simular-ai/Agent-S) library, supporting Mac, OSWorld, and WindowsAgentArena.
*   **2024/10/10:** Released the [Agent S paper](https://arxiv.org/abs/2410.08164) and codebase.

## 🎯 Current Results

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

> **Note:** Agent S2 returns `pyautogui` code for single-monitor setups.

> ❗**Warning:** Avoid using `conda` environments on Linux due to conflicts with `pyatspi`.

> ⚠️**Disclaimer:**  Agent S2 utilizes [UI-TARS](https://github.com/bytedance/UI-TARS) as a grounding model (7B-DPO or 72B-DPO).  You can use a Hugging Face Inference Endpoint.

### Installation

```bash
pip install gui-agents
```

### Environment Variables

Set your API keys:

```bash
export OPENAI_API_KEY=<YOUR_API_KEY>
export ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
```

Or, set them in your Python script:

```python
import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
```

### Model Support

Agent S supports Azure OpenAI, Anthropic, Gemini, Open Router, and vLLM inference. Refer to [models.md](models.md) for details.

### Perplexica Setup (for Web Knowledge Retrieval)

1.  Install and run Docker Desktop.
2.  Navigate to the `Perplexica` directory and update submodules:

    ```bash
    cd Perplexica
    git submodule update --init
    ```
3.  Rename `sample.config.toml` to `config.toml`.  Fill in your API keys (at minimum your OpenAI API key if you're using OpenAI)
4.  Run the Docker Compose file:

    ```bash
    docker compose up -d
    ```
5.  Set the `PERPLEXICA_URL` environment variable:

    ```bash
    export PERPLEXICA_URL=http://localhost:{port}/api/search
    ```

    *   Replace `{port}` with the port from `docker-compose.yaml` (e.g., 3000).
6.  Modify the `agent_s/query_perplexica.py` file as needed. For more on configuring the Perplexica API, please refer to [Perplexica Search API Documentation](https://github.com/ItzCrazyKns/Perplexica/blob/master/docs/API/SEARCH.md).
7.  For a detailed setup and usage guide, please refer to the [Perplexica Repository](https://github.com/ItzCrazyKns/Perplexica.git).

> ❗**Warning:** The agent executes Python code. Use with caution.

## 🚀 Usage

> **Note:** For best results, use Claude 3.7 with extended thinking and UI-TARS-72B-DPO.  If resources are limited, use UI-TARS-7B-DPO.

### CLI

Run Agent S2:

```bash
agent_s2 \
  --provider "anthropic" \
  --model "claude-3-7-sonnet-20250219" \
  --grounding_model_provider "anthropic" \
  --grounding_model "claude-3-7-sonnet-20250219"
```

Or with a custom endpoint:

```bash
agent_s2 \
  --provider "anthropic" \
  --model "claude-3-7-sonnet-20250219" \
  --endpoint_provider "huggingface" \
  --endpoint_url "<endpoint_url>/v1/"
```

#### Model Settings

*   **`--provider`**, **`--model`**: Specify the main generation model. Supports all providers in [models.md](models.md). Default: `anthropic/claude-3-7-sonnet-20250219`
*   **`--model_url`**, **`--model_api_key`**: Use a custom endpoint. Optional.

#### Grounding Configuration

##### **Configuration 1: API-Based Models (Default)**

*   **`--grounding_model_provider`**, **`--grounding_model`**: Specify the grounding model. Supports all providers in [models.md](models.md). Default: `anthropic/claude-3-7-sonnet-20250219`
*   ❗**Important**❗ **`--grounding_model_resize_width`**: Handle image rescaling for accurate grounding.  Default: `1366` (Anthropic).

##### **Configuration 2: Custom Endpoint**

*   **`--endpoint_provider`**:  Endpoint provider. Supports: HuggingFace TGI, vLLM, Open Router.
*   **`--endpoint_url`**: Your custom endpoint URL.
*   **`--endpoint_api_key`**: Your API key for the custom endpoint. Optional.

> **Note:** Configuration 2 takes precedence over Configuration 1.

### `gui_agents` SDK

Example using the SDK:

```python
import pyautogui
import io
from gui_agents.s2.agents.agent_s import AgentS2
from gui_agents.s2.agents.grounding import OSWorldACI
from dotenv import load_dotenv
load_dotenv()

current_platform = "linux"  # "darwin", "windows"

# Engine parameters
engine_params = {
  "engine_type": provider,
  "model": model,
  "base_url": model_url,     # Optional
  "api_key": model_api_key,  # Optional
}

# Grounding Configuration 1: API based model
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

# Grounding Configuration 2: HuggingFace TGI endpoint
endpoint_provider = "<your_endpoint_provider>"
endpoint_url = "<your_endpoint_url>"
endpoint_api_key = "<your_api_key>"

engine_params_for_grounding = {
  "engine_type": endpoint_provider,
  "base_url": endpoint_url,
  "api_key": endpoint_api_key,  # Optional
}

# Initialize agents
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
  search_engine="Perplexica",
  embedding_engine_type="openai"
)

# Get screenshot and make prediction
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

Refer to `gui_agents/s2/cli_app.py` for more.

#### Downloading the Knowledge Base

```python
from gui_agents.utils.download_kb_data import download_kb_data

download_kb_data(
    version="s2",
    release_tag="v0.2.2",
    download_dir="kb_data",
    platform="linux"  # "darwin", "windows"
)
```

## OSWorld

See [OSWorld Deployment Instructions](osworld_setup/s2/OSWorld.md) for OSWorld deployment.

## WindowsAgentArena

See [WindowsAgentArena Deployment Instructions](WAA_setup.md) for WindowsAgentArena deployment.

## 💬 Citations

If you use this codebase, please cite our papers:

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

[![Star History Chart](https://api.star-history.com/svg?repos=simular-ai/Agent-S&type=Date)](https://www.star-history.com/#simular-ai/Agent-S&Date)