<!-- Improved README.md -->
<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Revolutionizing Software Development with AI Agents</h1>
  <p><i>Unlock the power of AI to automate your software development tasks and build more, faster.</i></p>
  <a href="https://github.com/All-Hands-AI/OpenHands">
    <img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stars">
  </a>
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors">
  </a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License">
  </a>
  <br/>
  <a href="https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA">
    <img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community">
  </a>
  <a href="https://discord.gg/ESHStjSjD4">
    <img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community">
  </a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md">
    <img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits">
  </a>
  <br/>
  <a href="https://docs.all-hands.dev/usage/getting-started">
    <img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation">
  </a>
  <a href="https://arxiv.org/abs/2407.16741">
    <img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv">
  </a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0">
    <img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score">
  </a>
</div>

<hr>

## **About OpenHands**

OpenHands (formerly OpenDevin) is a groundbreaking platform that empowers software development with advanced AI agents.  These AI agents can perform a wide range of tasks, mirroring the capabilities of human developers to streamline your workflow and boost productivity.

**Key Features:**

*   **Code Modification:** Modify and refactor code efficiently.
*   **Command Execution:** Run commands and automate development processes.
*   **Web Browsing:** Access and utilize web resources for research and information retrieval.
*   **API Integration:** Interact with APIs to enhance functionality.
*   **Code Snippet Integration:** Leverage code snippets from resources like StackOverflow.

## **Getting Started**

### **OpenHands Cloud**

The quickest and easiest way to explore OpenHands is through [OpenHands Cloud](https://app.all-hands.dev).  New users receive $20 in free credits to get started!

### **Local Installation**

Choose your preferred installation method:

#### **Option 1: CLI Launcher (Recommended)**

1.  **Install `uv`:**  Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for your platform.
2.  **Launch OpenHands:**

    ```bash
    # Launch the GUI server
    uvx --python 3.12 --from openhands-ai openhands serve

    # Or launch the CLI
    uvx --python 3.12 --from openhands-ai openhands
    ```

    Access the GUI at [http://localhost:3000](http://localhost:3000).

#### **Option 2: Docker**

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.54
```

>  **Note:**  If you used OpenHands before version 0.44, migrate your history with `mv ~/.openhands-state ~/.openhands`.
>
>  **Warning:** Secure your Docker deployment on public networks by following the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

### **Configuration**

Upon launching, you'll be prompted to select an LLM provider and provide your API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.  Explore [other LLM options](https://docs.all-hands.dev/usage/llms).

Refer to the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and detailed setup instructions.

## **Additional Usage Options**

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Interact via a CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [Headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

For in-depth information, visit [Running OpenHands](https://docs.all-hands.dev/usage/installation).

For contributing to the code, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

For troubleshooting, refer to the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## **Documentation & Resources**

*   **Comprehensive Documentation:** [https://docs.all-hands.dev/usage/getting-started](https://docs.all-hands.dev/usage/getting-started)
*   **DeepWiki Documentation:**
    <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

## **Join the Community**

We welcome contributions and participation!

*   **Slack:** [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   **Discord:** [Join our Discord server](https://discord.gg/ESHStjSjD4)
*   **GitHub Issues:** [https://github.com/All-Hands-AI/OpenHands/issues](https://github.com/All-Hands-AI/OpenHands/issues)
*   **COMMUNITY.md:** (More community details)
*   **CONTRIBUTING.md:** (Contribution guidelines)

## **Project Progress**

Track our progress via the monthly roadmap: [https://github.com/orgs/All-Hands-AI/projects/1](https://github.com/orgs/All-Hands-AI/projects/1)

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## **License**

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

## **Acknowledgements**

OpenHands is a community effort.  We deeply appreciate all contributions and the open-source projects we build upon.  See our [CREDITS.md](./CREDITS.md) file for a complete list.

## **Cite**

```
@inproceedings{
  wang2025openhands,
  title={OpenHands: An Open Platform for {AI} Software Developers as Generalist Agents},
  author={Xingyao Wang and Boxuan Li and Yufan Song and Frank F. Xu and Xiangru Tang and Mingchen Zhuge and Jiayi Pan and Yueqi Song and Bowen Li and Jaskirat Singh and Hoang H. Tran and Fuqiang Li and Ren Ma and Mingzhang Zheng and Bill Qian and Yanjun Shao and Niklas Muennighoff and Yizhe Zhang and Binyuan Hui and Junyang Lin and Robert Brennan and Hao Peng and Heng Ji and Graham Neubig},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=OJd3ayDDoF}
}
```