# Agent S: Revolutionizing Computer Use with AI Agents

**Agent S** is an open-source framework that empowers AI agents to autonomously interact with computers, opening new possibilities for automation and intelligent task execution. ([View Original Repo](https://github.com/simular-ai/Agent-S))

**Key Features:**

*   🚀 **Autonomous Computer Control:** Enables agents to perform complex tasks on your computer, mimicking human interaction.
*   🧠 **Compositional Generalist-Specialist Architecture:** Leverages a unique architecture for efficient task execution and adaptability.
*   🌐 **Open-Source and Accessible:** Designed for researchers and developers to contribute to the advancement of AI agents.
*   🛠️ **GUI Agent Framework:** Built to interact with your computer through a graphical user interface (GUI).
*   🏆 **SOTA Performance:** Achieves state-of-the-art results across benchmarks like OSWorld, WindowsAgentArena, and AndroidWorld.

## 🌟 What's New: Agent S2 is Here!

*   **[Agent S2 Paper](https://arxiv.org/abs/2504.00906) accepted to COLM 2025!**
*   **[Agent S2 Paper](https://arxiv.org/abs/2504.00906) Released:** Achieve new SOTA results on OSWorld, WindowsAgentArena, and AndroidWorld.
*   **Agent S2 Released:** Outperforms OpenAI's CUA/Operator and Anthropic's Claude 3.7 Sonnet Computer-Use.
*   **[Agent S Paper](https://arxiv.org/abs/2410.08164) Accepted to ICLR 2025!**
*   **GUI Agents Library Updates:** Enhanced support for Linux, Windows, and MacOS.

## 🎯 Current Results

Agent S2 demonstrates impressive performance in various benchmarks:

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

> ❗**Warning**❗: If you are on a Linux machine, creating a `conda` environment will interfere with `pyatspi`. As of now, there's no clean solution for this issue. Proceed through the installation without using `conda` or any virtual environment.

> ⚠️**Disclaimer**⚠️: To leverage the full potential of Agent S2, we utilize [UI-TARS](https://github.com/bytedance/UI-TARS) as a grounding model (7B-DPO or 72B-DPO for better performance). They can be hosted locally, or on Hugging Face Inference Endpoints. Our code supports Hugging Face Inference Endpoints. Check out [Hugging Face Inference Endpoints](https://huggingface.co/learn/cookbook/en/enterprise_dedicated_endpoints) for more information on how to set up and query this endpoint. However, running Agent S2 does not require this model, and you can use alternative API based models for visual grounding, such as Claude.

```bash
pip install gui-agents
```

Configure your environment variables for API keys:

```bash
export OPENAI_API_KEY=<YOUR_API_KEY>
export ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
```

Or, set environment variables within your Python script:

```python
import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
```

### Setup Retrieval from Web using Perplexica
Agent S works best with web-knowledge retrieval. To enable this feature, you need to setup Perplexica: 

1. Ensure Docker Desktop is installed and running on your system.

2. Navigate to the directory containing the project files.

   ```bash
    cd Perplexica
    git submodule update --init
   ```

3. Rename the `sample.config.toml` file to `config.toml`. For Docker setups, you need only fill in the following fields:

   - `OPENAI`: Your OpenAI API key. **You only need to fill this if you wish to use OpenAI's models**.
   - `OLLAMA`: Your Ollama API URL. You should enter it as `http://host.docker.internal:PORT_NUMBER`. If you installed Ollama on port 11434, use `http://host.docker.internal:11434`. For other ports, adjust accordingly. **You need to fill this if you wish to use Ollama's models instead of OpenAI's**.
   - `GROQ`: Your Groq API key. **You only need to fill this if you wish to use Groq's hosted models**.
   - `ANTHROPIC`: Your Anthropic API key. **You only need to fill this if you wish to use Anthropic models**.

     **Note**: You can change these after starting Perplexica from the settings dialog.

   - `SIMILARITY_MEASURE`: The similarity measure to use (This is filled by default; you can leave it as is if you are unsure about it.)

4. Ensure you are in the directory containing the `docker-compose.yaml` file and execute:

   ```bash
   docker compose up -d
   ```
5. Export your Perplexica URL using the port found in the [`docker-compose.yaml`](https://github.com/ItzCrazyKns/Perplexica/blob/master/docker-compose.yaml) file Under `app/ports`, you'll see `3000:3000`. The port is the left-hand number (in this case, 3000).

   ```bash
   export PERPLEXICA_URL=http://localhost:{port}/api/search
   ```
6. Our implementation of Agent S incorporates the Perplexica API to integrate a search engine capability, which allows for a more convenient and responsive user experience. If you want to tailor the API to your settings and specific requirements, you may modify the URL and the message of request parameters in  `agent_s/query_perplexica.py`. For a comprehensive guide on configuring the Perplexica API, please refer to [Perplexica Search API Documentation](https://github.com/ItzCrazyKns/Perplexica/blob/master/docs/API/SEARCH.md).
For a more detailed setup and usage guide, please refer to the [Perplexica Repository](https://github.com/ItzCrazyKns/Perplexica.git).

> ❗**Warning**❗: The agent will directly run python code to control your computer. Please use with care.

## 🚀 Usage

> **Note**: Our best configuration uses Claude 3.7 with extended thinking and UI-TARS-72B-DPO. If you are unable to run UI-TARS-72B-DPO due to resource constraints, UI-TARS-7B-DPO can be used as a lighter alternative with minimal performance degradation.

### CLI

Run Agent S2:

```bash
agent_s2 \
  --provider "anthropic" \
  --model "claude-3-7-sonnet-20250219" \
  --grounding_model_provider "anthropic" \
  --grounding_model "claude-3-7-sonnet-20250219" \
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
- **`--provider`**, **`--model`** 
  - Purpose: Specifies the main generation model
  - Supports: all model providers in [models.md](models.md)
  - Default: `--provider "anthropic" --model "claude-3-7-sonnet-20250219"`
- **`--model_url`**, **`--model_api_key`**
   - Purpose: Specifies the custom endpoint for the main generation model and your API key
   - Note: These are optional. If not specified, `gui-agents` will default to your environment variables for the URL and API key.
   - Supports: all model providers in [models.md](models.md)
   - Default: None

#### Grounding Configuration Options

You can use either Configuration 1 or Configuration 2:

##### **(Default) Configuration 1: API-Based Models**
- **`--grounding_model_provider`**, **`--grounding_model`**
  - Purpose: Specifies the model for visual grounding (coordinate prediction)
  - Supports: all model providers in [models.md](models.md)
  - Default: `--grounding_model_provider "anthropic" --grounding_model "claude-3-7-sonnet-20250219"`
- ❗**Important**❗ **`--grounding_model_resize_width`**
  - Purpose:  Some API providers automatically rescale images. Therefore, the generated (x, y) will be relative to the rescaled image dimensions, instead of the original image dimensions.
  - Supports: [Anthropic rescaling](https://docs.anthropic.com/en/docs/build-with-claude/vision#)
  - Tips: If your grounding is inaccurate even for very simple queries, double check your rescaling width is correct for your machine's resolution.
  - Default: `--grounding_model_resize_width 1366` (Anthropic)

##### **Configuration 2: Custom Endpoint**
- **`--endpoint_provider`**
  - Purpose: Specifies the endpoint provider
  - Supports: HuggingFace TGI, vLLM, Open Router
  - Default: None

- **`--endpoint_url`**
  - Purpose: The URL for your custom endpoint
  - Default: None

- **`--endpoint_api_key`**
   - Purpose: Your API key for your custom endpoint
   - Note: This is optional. If not specified, `gui-agents` will default to your environment variables for the API key.
   - Default: None

> **Note**: Configuration 2 takes precedence over Configuration 1.

### `gui_agents` SDK

```python
import pyautogui
import io
from gui_agents.s2.agents.agent_s import AgentS2
from gui_agents.s2.agents.grounding import OSWorldACI

from dotenv import load_dotenv
load_dotenv()

current_platform = "linux"  # "darwin", "windows"

# Configuration for Main Agent
engine_params = {
  "engine_type": provider,
  "model": model,
  "base_url": model_url,     # Optional
  "api_key": model_api_key,  # Optional
}

# Grounding Configuration 1: API Based Model
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

# Grounding and Agent Initialization
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

# Inference Example
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

#### Downloading the Knowledge Base

```python
download_kb_data(
    version="s2",
    release_tag="v0.2.2",
    download_dir="kb_data",
    platform="linux"  # "darwin", "windows"
)
```

### OSWorld

Follow the [OSWorld Deployment instructions](OSWorld.md) for deployment.

## 💬 Citations

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
```

```
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