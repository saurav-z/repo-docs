<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands"><img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200"></a>
  <h1>OpenHands: AI-Powered Software Development, Simplified</h1>
</div>

<div align="center">
  <!-- Badges -->
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors"><img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/stargazers"><img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE"><img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License"></a>
  <br/>
  <a href="https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA"><img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community"></a>
  <a href="https://discord.gg/ESHStjSjD4"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md"><img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits"></a>
  <br/>
  <a href="https://docs.all-hands.dev/usage/getting-started"><img src="https://img.shields.io/badge/Documentation-Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <a href="https://arxiv.org/abs/2407.16741"><img src="https://img.shields.io/badge/Paper%20on%20Arxiv-Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv"></a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0"><img src="https://img.shields.io/badge/Benchmark%20score-Benchmarks-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score"></a>
</div>

<hr>

OpenHands (formerly OpenDevin) empowers developers with AI agents capable of a wide range of software development tasks, letting you **code less and build more.**  [Get started on GitHub](https://github.com/All-Hands-AI/OpenHands)

## Key Features

*   **AI-Powered Code Modification:**  Intelligently modify code, fix bugs, and implement new features.
*   **Web Browsing & API Integration:** Agents can browse the web for information and integrate with APIs for enhanced functionality.
*   **Automated Task Execution:**  Execute commands, run tests, and automate repetitive development tasks.
*   **Open Source & Customizable:**  Leverage the power of AI within your own development workflow with this open-source platform.
*   **Stack Overflow Integration:**  Access and utilize code snippets from Stack Overflow to accelerate development.
*   **Cloud and Local Deployment:** Run OpenHands on the cloud or locally using Docker.

## Get Started with OpenHands

### OpenHands Cloud

The easiest way to begin is on [OpenHands Cloud](https://app.all-hands.dev), which offers $20 in free credits for new users.

### Running OpenHands Locally

You can also run OpenHands on your local system using Docker.  See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and installation instructions.

**Docker Run Command:**
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
    docker.all-hands.dev/all-hands-ai/openhands:0.48
```

Access OpenHands at [http://localhost:3000](http://localhost:3000) after launching.

**Note:** If you used OpenHands before version 0.44, you may need to run `mv ~/.openhands-state ~/.openhands`.

#### LLM Provider

You'll be prompted to select an LLM provider and provide an API key; [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but many [other options](https://docs.all-hands.dev/usage/llms) are supported.

## Other Ways to Run OpenHands

*   [Connecting to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [CLI mode](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [Github Action](https://docs.all-hands.dev/usage/how-to/github-action)

## Documentation

For in-depth information, tutorials, and advanced configuration options, explore the [OpenHands Documentation](https://docs.all-hands.dev/usage/getting-started).

## Community

Join the OpenHands community for support, discussions, and contributions:

*   [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   [Discord](https://discord.gg/ESHStjSjD4)
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues)

## Roadmap

*   View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

OpenHands is licensed under the MIT License.  See the [`LICENSE`](./LICENSE) file for details.

## Acknowledgements

OpenHands is a community effort, and we are thankful for all contributions.  See [CREDITS.md](./CREDITS.md) for a list of open-source projects and licenses utilized.

## Cite

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