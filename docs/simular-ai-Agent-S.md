<h1 align="center">
  <img src="images/agent_s.png" alt="Agent S Logo" style="vertical-align:middle" width="60"> Agent S: Automate Your Computer with Human-Like Intelligence
</h1>

<p align="center">
  Agent S empowers your computer to perform tasks autonomously, mimicking human interaction. Explore the future of AI-driven automation!
</p>

<p align="center">
  🌐 <a href="https://www.simular.ai/articles/agent-s2-technical-review">[S2 blog]</a>&nbsp;
  📄 <a href="https://arxiv.org/abs/2504.00906">[S2 Paper (COLM 2025)]</a>&nbsp;
  🎥 <a href="https://www.youtube.com/watch?v=wUGVQl7c0eg">[S2 Video]</a>
</p>

<p align="center">
  🌐 <a href="https://www.simular.ai/agent-s">[S1 blog]</a>&nbsp;
  📄 <a href="https://arxiv.org/abs/2410.08164">[S1 Paper (ICLR 2025)]</a>&nbsp;
  🎥 <a href="https://www.youtube.com/watch?v=OBDE3Knte0g">[S1 Video]</a>
</p>

<p align="center">
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

<div align="center">
  &nbsp;&nbsp;
  <p>Skip the setup? Try Agent S in <a href="https://cloud.simular.ai/">Simular Cloud</a>
</div>

## Key Features

*   **Advanced Automation:** Automates computer tasks with human-like interaction.
*   **GUI Interaction:**  Operates directly through the graphical user interface (GUI).
*   **Open Source:** Leverage a powerful, open-source framework for GUI agents.
*   **Cutting-Edge Results:** See the latest performance metrics (SOTA)
*   **Multi-Platform Support:**  Works on Linux, macOS, and Windows.

## 🥳 Updates

*   **2025/08/01**: Agent S2.5 is released (gui-agents v0.2.5): simpler, better, and faster! New SOTA on [OSWorld-Verified](https://os-world.github.io)!
*   **2025/07/07**: The [Agent S2 paper](https://arxiv.org/abs/2504.00906) is accepted to COLM 2025! See you in Montreal!
*   **2025/04/01**: Released the [Agent S2 paper](https://arxiv.org/abs/2504.00906) with new SOTA results on OSWorld, WindowsAgentArena, and AndroidWorld!
*   **2025/03/12**: Released Agent S2 along with v0.2.0 of [gui-agents](https://github.com/simular-ai/Agent-S), the new state-of-the-art for computer use agents (CUA), outperforming OpenAI's CUA/Operator and Anthropic's Claude 3.7 Sonnet Computer-Use!
*   **2025/01/22**: The [Agent S paper](https://arxiv.org/abs/2410.08164) is accepted to ICLR 2025!
*   **2025/01/21**: Released v0.1.2 of [gui-agents](https://github.com/simular-ai/Agent-S) library, with support for Linux and Windows!
*   **2024/12/05**: Released v0.1.0 of [gui-agents](https://github.com/simular-ai/Agent-S) library, allowing you to use Agent-S for Mac, OSWorld, and WindowsAgentArena with ease!
*   **2024/10/10**: Released the [Agent S paper](https://arxiv.org/abs/2410.08164) and codebase!

## Table of Contents

1.  [💡 Introduction](#-introduction)
2.  [🎯 Current Results](#-current-results)
3.  [🛠️ Installation & Setup](#-installation--setup)
4.  [🚀 Usage](#-usage)
5.  [🤝 Acknowledgements](#-acknowledgements)
6.  [💬 Citation](#-citation)

## 💡 Introduction

Agent S is an open-source framework designed to enable autonomous computer interaction through an Agent-Computer Interface, allowing you to automate complex tasks with ease. This framework builds intelligent GUI agents that learn and perform autonomously on your computer.

## 🎯 Current Results

Agent S achieves state-of-the-art results in various benchmarks.

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
  </table>
</div>

## 🛠️ Installation & Setup

### Prerequisites

*   **Single Monitor:**  Agent S is designed for single-monitor setups.
*   **Security:** Agent S executes Python code to control your computer; use with caution.
*   **Supported Platforms:** Linux, macOS, and Windows.

### Installation

```bash
pip install gui-agents
```

### API Configuration

#### Option 1: Environment Variables

Add the following to your `.bashrc` (Linux) or `.zshrc` (macOS):

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

Agent S supports Azure OpenAI, Anthropic, Gemini, Open Router, and vLLM inference. Refer to [models.md](models.md) for detailed information.

### Grounding Models (Required)

For optimal performance, we recommend [UI-TARS-1.5-7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B) hosted on Hugging Face Inference Endpoints or a similar provider. See [Hugging Face Inference Endpoints](https://huggingface.co/learn/cookbook/en/enterprise_dedicated_endpoints) for setup instructions.

## 🚀 Usage

> ⚡️ **Recommended Setup:**
> For the best configuration, we recommend using **OpenAI o3-2025-04-16** as the main model, paired with **UI-TARS-1.5-7B** for grounding.

### CLI

Run Agent S2.5 using the command-line interface with the following parameters:

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

*   **`--provider`**:  Main generation model provider (e.g., openai, anthropic). Default: "openai"
*   **`--model`**:  Main generation model name (e.g., o3-2025-04-16). Default: "o3-2025-04-16"
*   **`--ground_provider`**: The provider for the grounding model - **Required**
*   **`--ground_url`**: The URL of the grounding model - **Required**
*   **`--ground_model`**: The model name for the grounding model - **Required**
*   **`--grounding_width`**: Width of the output coordinate resolution from the grounding model - **Required**
*   **`--grounding_height`**: Height of the output coordinate resolution from the grounding model - **Required**

#### Grounding Model Dimensions

Ensure the grounding width and height match your grounding model's output coordinate resolution:

*   **UI-TARS-1.5-7B**: Use `--grounding_width 1920 --grounding_height 1080`
*   **UI-TARS-72B**: Use `--grounding_width 1000 --grounding_height 1000`

#### Optional Parameters

*   **`--model_url`**:  Custom API URL for main generation model. Default: ""
*   **`--model_api_key`**:  API key for main generation model. Default: ""
*   **`--ground_api_key`**: API key for grounding model endpoint. Default: ""
*   **`--max_trajectory_length`**: Maximum number of image turns to keep in trajectory. Default: 8
*   **`--enable_reflection`**: Enable reflection agent to assist the worker agent. Default: True

### `gui_agents` SDK

First, import the necessary modules. `AgentS2_5` is the main agent class, and `OSWorldACI` is the grounding agent.

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

Next, define the engine parameters for both the main agent and the grounding agent.  For grounding, you can use custom endpoints like HuggingFace TGI or vLLM.

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

Define the grounding agent and Agent S2.5.

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

Finally, make a query to the agent!

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

Refer to `gui_agents/s2_5/cli_app.py` for more details on the inference loop.

### OSWorld

To deploy Agent S2.5 in OSWorld, follow the instructions in [OSWorld Deployment](osworld_setup/s2_5/OSWorld.md).

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

[![Star History Chart](https://api.star-history.com/svg?repos=simular-ai/Agent-S&type=Date)](https://www.star-history.com/#agent-s/agent-s&simular-ai/Agent-S&Date)

[Back to Top](#-agent-s-automate-your-computer-with-human-like-intelligence) (link back to top)
```
Key improvements and explanations:

*   **SEO Optimization:** The title is now very keyword-rich. The introduction uses keywords like "automate," "computer," "human-like," and "AI".  The key features are highlighted.
*   **Concise Hook:** The one-sentence hook is added to make a strong first impression.
*   **Clear Headings:**  Uses H1, H2, and bullet points for better readability and scannability.
*   **Focus on Benefits:**  Highlights *what* Agent S does (automates computers) rather than just *what* it is.
*   **Clear Structure:**  The README is well-organized, making it easy for users to find information.
*   **Actionable Steps:**  Installation and usage instructions are clear and easy to follow.
*   **Consistent Formatting:** Used Markdown formatting consistently.
*   **Links:** Corrected all links and added a link back to the original repository.
*   **Call to Action**: Includes a call to action to try the Agent S in the cloud.
*   **"Back to Top" link**: Added a back to top link to improve user experience.
*   **Improved Descriptions**: Improved the descriptions for grounding and CLI usage.
*   **Grounding model dimensions**: Specified the model dimensions to better guide users.

This improved README is much more engaging, informative, and helpful for potential users of Agent S.  It is also better optimized for search engines.