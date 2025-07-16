<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands">
    <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  </a>
  <h1>OpenHands: Code Less, Make More with AI-Powered Software Development</h1>
  <p>
    <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors">
      <img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/stargazers">
      <img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE">
      <img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License">
    </a>
  </p>
  <p>
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
  </p>

  <!-- Keep these links. Translations will automatically update with the README. -->
  <p>
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de">Deutsch</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es">Español</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr">français</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja">日本語</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko">한국어</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt">Português</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru">Русский</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh">中文</a>
  </p>
  <hr>
</div>

<p>OpenHands, formerly OpenDevin, empowers developers by providing AI agents that can perform complex software development tasks, significantly boosting productivity.</p>

## Key Features of OpenHands

*   **AI-Powered Agents:**  Utilize intelligent agents capable of understanding and executing development tasks.
*   **Code Modification & Generation:**  Modify existing codebases and generate new code based on specifications.
*   **Web Browsing and API Integration:**  Browse the web, call APIs, and integrate external services to enhance functionality.
*   **Code Snippet Integration:** Seamlessly incorporate code snippets from platforms like StackOverflow.
*   **Local & Cloud Deployment:** Run OpenHands locally with Docker or utilize the convenient OpenHands Cloud.
*   **Community-Driven:** Benefit from a growing community and contribute to the project's evolution.

Explore the comprehensive documentation at [docs.all-hands.dev](https://docs.all-hands.dev) and get started today!

> [!IMPORTANT]
> Interested in using OpenHands for your work?  Join our Design Partner program by filling out [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform) for early access and input on our product roadmap.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

The quickest way to start is with [OpenHands Cloud](https://app.all-hands.dev), which offers $20 in free credits for new users.

## 💻 Running OpenHands Locally

You can also run OpenHands locally using Docker.  See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for details.

> [!WARNING]
> For secure local deployments on public networks, see our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.48-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands-dev/all-hands-ai/runtime:0.48-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands-dev/all-hands-ai/openhands:0.48
```

> **Note**: If you used OpenHands before version 0.44, run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history.

Access OpenHands at [http://localhost:3000](http://localhost:3000) after setup. Configure your LLM provider (e.g., [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api)) with an API key; see [LLM options](https://docs.all-hands.dev/usage/llms).

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user, local workstation use. Multi-tenant deployments are not supported.  Consider the [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud) for multi-user environments.

Explore these additional deployment options:

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode) for scripting
*   [CLI mode](https://docs.all-hands.dev/usage/how-to/cli-mode) for a command-line interface
*   [Github Action](https://docs.all-hands.dev/usage/how-to/github-action) to run on tagged issues

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for detailed instructions.

## 📖 Documentation

  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Find comprehensive resources and guides on using OpenHands at [documentation](https://docs.all-hands.dev/usage/getting-started).

## 🤝 Join the OpenHands Community

Contribute and engage with the OpenHands community via:

*   [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA): Discuss research, architecture, and development.
*   [Discord](https://discord.gg/ESHStjSjD4): Community-run server for discussions, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues):  Share ideas, or track issues.

## 📈 Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

OpenHands is released under the [MIT License](./LICENSE).

## 🙏 Acknowledgements

OpenHands thrives on contributions from many individuals. Our credits are detailed in [CREDITS.md](./CREDITS.md).

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