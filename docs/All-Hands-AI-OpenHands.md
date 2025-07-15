<!-- Improved & SEO-Optimized README for OpenHands -->
<a name="readme-top"></a>

<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1 align="center">OpenHands: AI-Powered Software Development, Simplified</h1>
  <p align="center">
    <i>Code less, achieve more with OpenHands, the AI agent for software development.</i>
  </p>
</div>

<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors"><img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/stargazers"><img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE"><img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License"></a>
  <br/>
  <a href="https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA"><img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community"></a>
  <a href="https://discord.gg/ESHStjSjD4"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md"><img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits"></a>
  <br/>
  <a href="https://docs.all-hands.dev/usage/getting-started"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <a href="https://arxiv.org/abs/2407.16741"><img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv"></a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0"><img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score"></a>

  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de">Deutsch</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es">Español</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr">français</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja">日本語</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko">한국어</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt">Português</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru">Русский</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh">中文</a>

  <hr>
</div>

## About OpenHands

OpenHands (formerly OpenDevin) is an open-source platform built to enable AI-powered software development. It provides a powerful AI agent that can perform a wide range of development tasks, streamlining your workflow and boosting productivity.  Access the original repository on [GitHub](https://github.com/All-Hands-AI/OpenHands).

## Key Features

*   **AI-Powered Automation:** Leverage the power of AI to automate coding, debugging, and other development tasks.
*   **Code Modification:**  Easily modify existing code with intelligent suggestions and automated changes.
*   **Web Browsing & API Integration:**  Allowing agents to browse the web and integrate with APIs to complete tasks.
*   **StackOverflow Integration:** Automate the process of fetching and implementing code snippets from popular resources.
*   **Cloud & Local Deployment:** Choose from convenient cloud access or run OpenHands locally using Docker.
*   **Community Driven:** Benefit from a vibrant and supportive community through Slack, Discord, and GitHub.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

Get started instantly with OpenHands on [OpenHands Cloud](https://app.all-hands.dev) and receive $20 in free credits.

## 💻 Running OpenHands Locally

OpenHands can also be run locally using Docker, allowing you to customize your development environment.

1.  **Prerequisites:** Ensure you have Docker installed.
2.  **Docker Pull:** Pull the OpenHands Docker image.
3.  **Run OpenHands:** Execute the docker run command provided below.
4.  **Access:** Open your web browser and go to [http://localhost:3000](http://localhost:3000).

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.48-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.48-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands-ai/openhands:0.48
```

> **Note**: If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

For more detailed instructions, see the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.  For secure deployments, see the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

When you open the application, you'll be asked to choose an LLM provider and add an API key.
[Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`)
works best, but you have [many options](https://docs.all-hands.dev/usage/llms).

## 💡 Other Ways to Run OpenHands

OpenHands offers flexible deployment options to suit your needs:

*   **Connect to Local Filesystem:** Access and manage your local files directly.
*   **Headless Mode:** Run OpenHands in a scriptable, automated fashion.
*   **CLI Mode:** Interact with OpenHands via a user-friendly command-line interface.
*   **GitHub Action:** Integrate OpenHands into your CI/CD pipeline.

For detailed instructions on alternative deployment methods, see the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.

## 📖 Documentation

Find detailed information and tips on using OpenHands in our comprehensive [documentation](https://docs.all-hands.dev/usage/getting-started).
Learn about LLM providers, troubleshooting, and advanced configuration options.

## 🤝 Join the Community

OpenHands is a community-driven project and welcomes contributions. Connect with us and share your feedback:

*   **Slack:** [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) - Discuss research, architecture, and development.
*   **Discord:** [Join our Discord server](https://discord.gg/ESHStjSjD4) - Get general discussion, ask questions, and give feedback.
*   **GitHub Issues:** [Read or post Github Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Explore and contribute to open issues.

Explore the community in [COMMUNITY.md](./COMMUNITY.md) and find details on contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Progress

Stay updated on the project's progress by viewing the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated at the maintainer's meeting at the end of each month).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

OpenHands is licensed under the MIT License.  See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

We are deeply grateful to the many contributors and open-source projects that make OpenHands possible!

For a list of open source projects and licenses used in OpenHands, please see our [CREDITS.md](./CREDITS.md) file.

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