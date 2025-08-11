<h1 align="center">
  <img src="images/agent_s.png" alt="Agent S Logo" style="vertical-align:middle" width="60"> Agent S: Revolutionizing Computer Interaction with AI
</h1>

<p align="center">
  **Agent S empowers AI to interact with your computer like a human, opening doors to unprecedented automation and intelligent task execution.** <br>
  Explore the next generation of GUI agents!
  <br>
  <br>
  🌐 [S2 Blog](https://www.simular.ai/articles/agent-s2-technical-review) | 
  📄 [S2 Paper (COLM 2025)](https://arxiv.org/abs/2504.00906) | 
  🎥 [S2 Video](https://www.youtube.com/watch?v=wUGVQl7c0eg)
  <br>
  🌐 [S1 Blog](https://www.simular.ai/agent-s) | 
  📄 [S1 Paper (ICLR 2025)](https://arxiv.org/abs/2410.08164) | 
  🎥 [S1 Video](https://www.youtube.com/watch?v=OBDE3Knte0g)
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/13151" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13151" alt="simular-ai%2FAgent-S | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/OS-Windows-blue?logo=windows&logoColor=white" alt="Windows">
  <img src="https://img.shields.io/badge/OS-macOS-black?logo=apple&logoColor=white" alt="macOS">
  <img src="https://img.shields.io/badge/OS-Linux-yellow?logo=linux&logoColor=black" alt="Linux">
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

<div align="center">
  &nbsp;&nbsp;
  <p>Skip the setup? Try Agent S in <a href="https://cloud.simular.ai/">Simular Cloud</a></p>
</div>

## Key Features

*   **Advanced GUI Interaction:** Agent S enables AI to seamlessly interact with graphical user interfaces (GUIs).
*   **Autonomous Task Execution:** Automate complex tasks on your computer without human intervention.
*   **Open-Source and Customizable:** Build upon an open-source framework to suit your specific needs and explore cutting-edge agent-based systems.
*   **Cross-Platform Compatibility:** Supports Windows, macOS, and Linux.
*   **SOTA Performance:** Achieves state-of-the-art results on OSWorld and other benchmarks.

## 🚀 What's New?

*   **[Agent S2.5 released (gui-agents v0.2.5):** Simpler, better, and faster! New SOTA on [OSWorld-Verified](https://os-world.github.io)! (2025/08/01)
*   **[Agent S2 paper accepted to COLM 2025](https://arxiv.org/abs/2504.00906)!** (2025/07/07)
*   **Agent S paper won Best Paper Award at ICLR 2025 Agentic AI for Science Workshop!** (2025/04/27)
*   **[Agent S2 paper](https://arxiv.org/abs/2504.00906) released** with new SOTA results on OSWorld, WindowsAgentArena, and AndroidWorld! (2025/04/01)
*   **Agent S2 and gui-agents v0.2.0 released,** outperforming OpenAI's and Anthropic's CUA/Operator (2025/03/12)
*   **Agent S paper accepted to ICLR 2025!** (2025/01/22)
*   **gui-agents v0.1.2 released,** with Linux and Windows support! (2025/01/21)
*   **gui-agents v0.1.0 released,** enabling Agent S for Mac, OSWorld, and WindowsAgentArena! (2024/12/05)
*   **Agent S paper and codebase released!** (2024/10/10)

## 💡 Introduction

Agent S is an open-source framework designed for autonomous computer interaction. By leveraging cutting-edge AI, Agent S allows you to control and automate tasks on your computer, emulating human-like interactions with the GUI. Dive into the future of AI-powered automation.

## 🎯 Current Results

<div align="center">
  <table border="0" cellspacing="0" cellpadding="5">
    <tr>
      <th>Benchmark</th>
      <th>Agent S2.5</th>
      <th>Previous SOTA</th>
    </tr>
    <tr>
      <td>OSWorld Verified (100 step)</td>
      <td><b>56.0%</b></td>
      <td>53.1%</td>
    </tr>
    <tr>
      <td>OSWorld Verified (50 step)</td>
      <td><b>54.2%</b></td>
      <td>50.6%</td>
    </tr>
<!--     <tr>
      <td>WindowsAgentArena</td>
      <td>29.8%</td>
      <td>19.5% (NAVI)</td>
    </tr>
    <tr>
      <td>AndroidWorld</td>
      <td>54.3%</td>
      <td>46.8% (UI-TARS)</td>
    </tr> -->
  </table>
</div>

## 🛠️ Installation & Setup

### Prerequisites

*   **Single Monitor:** Ensure you are using a single monitor for optimal performance.
*   **Security:** Be aware that Agent S runs Python code to control your computer. Use with caution.
*   **Supported Platforms:** Linux, macOS, and Windows.

### Installation

```bash
pip install gui-agents
```

### API Configuration

#### Option 1: Environment Variables

Add the following to your `.bashrc` (Linux) or `.zshrc` (MacOS) file:

```bash
export OPENAI_API_KEY=<YOUR_API_KEY>
export ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
```

#### Option 2: Python Script

```python
import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
```

### Supported Models

Agent S supports a variety of models including Azure OpenAI, Anthropic, Gemini, Open Router, and vLLM inference. For more details, see [models.md](models.md).

### Grounding Models (Required)

For best performance, we recommend [UI-TARS-1.5-7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B) hosted on Hugging Face Inference Endpoints or another provider. See [Hugging Face Inference Endpoints](https://huggingface.co/learn/cookbook/en/enterprise_dedicated_endpoints) for setup instructions.

## 🚀 Usage

> ⚡️ **Recommended Setup:**  
> For the best configuration, we recommend using **OpenAI o3-2025-04-16** as the main model, paired with **UI-TARS-1.5-7B** for grounding.  

### CLI

Run Agent S2.5 using the command-line interface:

```bash
agent_s \
    --provider openai \
    --model o3-2025-04-16 \
    --ground_provider huggingface \
    --ground_url http://localhost:8080 \
    --ground_model ui-tars-1.5-7b \
    --grounding_width 1920 \
    --grounding_height 1080
```

#### Required Parameters

*   **`--provider`**: Main generation model provider (e.g., openai, anthropic, etc.) - Default: "openai"
*   **`--model`**: Main generation model name (e.g., o3-2025-04-16) - Default: "o3-2025-04-16"
*   **`--ground_provider`**: The provider for the grounding model - **Required**
*   **`--ground_url`**: The URL of the grounding model - **Required**
*   **`--ground_model`**: The model name for the grounding model - **Required**
*   **`--grounding_width`**: Width of the output coordinate resolution from the grounding model - **Required**
*   **`--grounding_height`**: Height of the output coordinate resolution from the grounding model - **Required**

#### Grounding Model Dimensions

Adjust the grounding width and height to match your grounding model's output coordinate resolution:

*   **UI-TARS-1.5-7B**: Use `--grounding_width 1920 --grounding_height 1080`
*   **UI-TARS-72B**: Use `--grounding_width 1000 --grounding_height 1000`

#### Optional Parameters

*   **`--model_url`**: Custom API URL for main generation model - Default: ""
*   **`--model_api_key`**: API key for main generation model - Default: ""
*   **`--ground_api_key`**: API key for grounding model endpoint - Default: ""
*   **`--max_trajectory_length`**: Maximum number of image turns to keep in trajectory - Default: 8
*   **`--enable_reflection`**: Enable reflection agent to assist the worker agent - Default: True

### `gui_agents` SDK

First, import necessary modules:

```python
import pyautogui
import io
from gui_agents.s2_5.agents.agent_s import AgentS2_5
from gui_agents.s2_5.agents.grounding import OSWorldACI

# Load in your API keys.
from dotenv import load_dotenv
load_dotenv()

current_platform = "linux"  # "darwin", "windows"
```

Then, define the engine parameters:

```python
engine_params = {
  "engine_type": provider,
  "model": model,
  "base_url": model_url,     # Optional
  "api_key": model_api_key,  # Optional
}

# Load the grounding engine from a custom endpoint
ground_provider = "<your_ground_provider>"
ground_url = "<your_ground_url>"
ground_model = "<your_ground_model>"
ground_api_key = "<your_ground_api_key>"

# Set grounding dimensions based on your model's output coordinate resolution
# UI-TARS-1.5-7B: grounding_width=1920, grounding_height=1080
# UI-TARS-72B: grounding_width=1000, grounding_height=1000
grounding_width = 1920  # Width of output coordinate resolution
grounding_height = 1080  # Height of output coordinate resolution

engine_params_for_grounding = {
  "engine_type": ground_provider,
  "model": ground_model,
  "base_url": ground_url,
  "api_key": ground_api_key,  # Optional
  "grounding_width": grounding_width,
  "grounding_height": grounding_height,
}
```

Next, instantiate the grounding agent and Agent S2.5:

```python
grounding_agent = OSWorldACI(
    platform=current_platform,
    engine_params_for_generation=engine_params,
    engine_params_for_grounding=engine_params_for_grounding,
    width=1920,  # Optional: screen width
    height=1080  # Optional: screen height
)

agent = AgentS2_5(
    engine_params,
    grounding_agent,
    platform=current_platform,
    max_trajectory_length=8,  # Optional: maximum image turns to keep
    enable_reflection=True     # Optional: enable reflection agent
)
```

Finally, query the agent:

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

For more details on the inference loop, refer to `gui_agents/s2_5/cli_app.py`.

### OSWorld

For deploying Agent S2.5 in OSWorld, consult the [OSWorld Deployment instructions](osworld_setup/s2_5/OSWorld.md).

## 🤝 Acknowledgements

We thank all contributors to this project and the open-source community.

## 💬 Citations

If you use this codebase, please cite the following:

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

[![Star History Chart](https://api.star-history.com/svg?repos=simular-ai/Agent-S&type=Date)](https://star-history.com/#simular-ai/Agent-S&Date)

[Back to Top](#-agent-s-revolutionizing-computer-interaction-with-ai)