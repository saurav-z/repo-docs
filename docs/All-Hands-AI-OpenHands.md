# OpenHands: Code Smarter, Not Harder with AI-Powered Software Development

**OpenHands is an AI-powered platform designed to revolutionize software development by automating tasks and empowering developers.** ([View on GitHub](https://github.com/All-Hands-AI/OpenHands))

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![MIT License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
[![Join Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Join Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Project Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Paper on Arxiv](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
[![Benchmark Score](https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge)](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0)

## Key Features

*   **AI-Powered Agents:** OpenHands agents can perform a wide array of software development tasks.
*   **Code Modification & Execution:** Modify and run code directly within the platform.
*   **Web Browsing & API Integration:** Access web resources and interact with APIs.
*   **Stack Overflow Integration:**  Leverage existing code solutions seamlessly.
*   **Cloud & Local Deployment:** Easily run on OpenHands Cloud or locally.
*   **CLI & GUI Options:** Access the platform via CLI and GUI.

## Getting Started

The easiest way to try OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), which offers $20 in free credits for new users.

## Run OpenHands Locally

### Option 1: CLI Launcher (Recommended)

Use the CLI launcher with [uv](https://docs.astral.sh/uv/) for better isolation:

**Install uv:**

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

**Launch OpenHands:**

```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

Find the GUI at [http://localhost:3000](http://localhost:3000).

### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

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
</details>

> **Note:** Migrate your conversation history if necessary: `mv ~/.openhands-state ~/.openhands`

> [!WARNING]
> For public networks, see the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation)

### Configuration

Choose an LLM provider and add an API key upon application startup. Anthropic's Claude Sonnet 4 (`anthropic/claude-sonnet-4-20250514`) works best. Explore other [LLM options](https://docs.all-hands.dev/usage/llms).

See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for detailed information.

## Explore More

### Additional Run Options

*   [Connect to your filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [CLI Mode](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [Headless Mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [Github Action](https://docs.all-hands.dev/usage/how-to/github-action)

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for more information.

## Development

Modify the OpenHands source code, by checking out [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

## Troubleshooting

Having issues? The [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) can help.

## 📖 Documentation

Comprehensive documentation, including LLM provider guidance, troubleshooting tips, and advanced configuration options, is available at [docs.all-hands.dev/usage/getting-started](https://docs.all-hands.dev/usage/getting-started).

## 🤝 Join the Community

Join our vibrant community and contribute to OpenHands!

*   [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   [Join our Discord server](https://discord.gg/ESHStjSjD4)
*   [Read or post Github Issues](https://github.com/All-Hands-AI/OpenHands/issues)

Find more community information in [COMMUNITY.md](./COMMUNITY.md) and contribution details in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Progress

See the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

OpenHands is a collaborative project. We thank all contributors and the creators of the open-source projects we build upon.

See [CREDITS.md](./CREDITS.md) for a list of open-source projects and licenses used.

## 📚 Cite

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