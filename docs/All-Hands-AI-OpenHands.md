<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: AI-Powered Software Development for the Modern Developer</h1>
  <p><em>Code less, make more with OpenHands, your AI-powered software development companion.</em></p>
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
      <img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Documentation">
  </a>
  <a href="https://arxiv.org/abs/2407.16741">
      <img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Arxiv Paper">
  </a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0">
      <img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Benchmark Score">
  </a>
</div>

<hr>

## Key Features of OpenHands

OpenHands revolutionizes software development, allowing you to leverage the power of AI.  Here's how:

*   🤖 **AI-Powered Agents:**  OpenHands agents act as your AI development team, capable of executing complex tasks.
*   💻 **Code Modification and Execution:** Modify code, run commands, and manage your projects effectively.
*   🌐 **Web Browsing & API Integration:** Access information online and integrate with various APIs for streamlined development.
*   🔗 **Stack Overflow Integration:**  Leverage the power of Stack Overflow to quickly find and implement code solutions.
*   ☁️ **Cloud & Local Deployment:**  Get started quickly with OpenHands Cloud, or run it locally using Docker.
*   🤝 **Community Driven:**  Join our active Slack and Discord communities to connect with other developers.

## Get Started with OpenHands

OpenHands (formerly OpenDevin) is an open-source platform designed to empower developers with AI-driven assistance.  Build software faster and more efficiently.  Visit the [OpenHands GitHub repository](https://github.com/All-Hands-AI/OpenHands) for the source code and more details.

### ☁️ OpenHands Cloud

The easiest way to experience OpenHands is through [OpenHands Cloud](https://app.all-hands.dev), which offers new users $20 in free credits.

### 💻 Running OpenHands Locally

OpenHands can be run locally using Docker. This allows you to customize your environment and experiment with various configurations. For detailed instructions, see the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.

> [!WARNING]
> For secure deployments on public networks, see our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.50-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.50-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.50
```

Access OpenHands at [http://localhost:3000](http://localhost:3000) after starting the container.

### 💡 Other Ways to Run OpenHands

*   **Connect to your local filesystem:** ([https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem))
*   **Headless Mode:** Run OpenHands in scriptable mode ([https://docs.all-hands.dev/usage/how-to/headless-mode](https://docs.all-hands.dev/usage/how-to/headless-mode)).
*   **CLI Mode:** Interact with OpenHands via a CLI interface ([https://docs.all-hands.dev/usage/how-to/cli-mode](https://docs.all-hands.dev/usage/how-to/cli-mode)).
*   **GitHub Action:**  Integrate OpenHands with GitHub Actions ([https://docs.all-hands.dev/usage/how-to/github-action](https://docs.all-hands.dev/usage/how-to/github-action)).

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for comprehensive installation guides.

### 📝 Development

For developers looking to modify OpenHands, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

### 🐛 Troubleshooting

Encountering issues? Consult the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) for solutions.

## 📖 Documentation

Discover more about OpenHands and how to use it effectively:

*   <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>
*   Comprehensive documentation is available at [docs.all-hands.dev/usage/getting-started](https://docs.all-hands.dev/usage/getting-started).
*   Find resources on LLM provider configuration, troubleshooting, and advanced settings.

## 🤝 Community

Join the OpenHands community and contribute to its ongoing development:

*   [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) - Discuss research, architecture, and future development.
*   [Join our Discord server](https://discord.gg/ESHStjSjD4) - Get involved in discussions and receive support.
*   [View or submit GitHub issues](https://github.com/All-Hands-AI/OpenHands/issues) - Track progress and contribute ideas.

See [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md) for more information.

## 📈 Progress

Stay updated on the project's progress:  See the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

OpenHands is distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

Thank you to all contributors and the open-source projects used by OpenHands. See [CREDITS.md](./CREDITS.md) for a comprehensive list.

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