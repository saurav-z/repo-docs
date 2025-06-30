# OpenHands: The AI Software Developer Agent – Code Less, Achieve More

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![MIT License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
[![Join Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Join Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Project Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Paper on Arxiv](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
[![Benchmark Score](https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge)](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0)
[![DeepWiki Documentation](https://deepwiki.com/badge.svg)](https://deepwiki.com/All-Hands-AI/OpenHands)

<br/>
<br/>

OpenHands is an open-source platform that empowers software development with AI agents, enabling you to automate tasks, accelerate development, and boost productivity.  Check out the original repository [here](https://github.com/All-Hands-AI/OpenHands).

## Key Features:

*   **AI-Powered Development:** Leverage AI agents to perform tasks typically handled by human developers.
*   **Code Modification:** OpenHands agents can modify existing codebases.
*   **Web Browsing:** Agents can browse the internet to gather information and research solutions.
*   **API Integration:** Seamlessly integrate with APIs for extended functionality.
*   **Code Snippet Retrieval:** Access and utilize code snippets from platforms like StackOverflow.
*   **Local and Cloud Options:** Run OpenHands locally via Docker or utilize the cloud-based platform.

![App screenshot](./docs/static/img/screenshot.png)

## Getting Started

### OpenHands Cloud

The easiest way to begin is through [OpenHands Cloud](https://app.all-hands.dev), which offers new users \$20 in free credits.

### Running Locally with Docker

1.  **Prerequisites:** Ensure Docker is installed on your system.

2.  **Pull the Docker Image:**
    ```bash
    docker pull docker.all-hands.dev/all-hands-ai/runtime:0.47-nikolaik
    ```

3.  **Run the Docker Container:**
    ```bash
    docker run -it --rm --pull=always \
        -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.47-nikolaik \
        -e LOG_ALL_EVENTS=true \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v ~/.openhands:/.openhands \
        -p 3000:3000 \
        --add-host host.docker.internal:host-gateway \
        --name openhands-app \
        docker.all-hands.dev/all-hands-ai/openhands:0.47
    ```

4.  **Access OpenHands:** Open your web browser and navigate to [http://localhost:3000](http://localhost:3000).

5.  **Configure LLM:** When prompted, select your preferred LLM provider and input your API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.

> **Note:** If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

## Additional Run Options

*   **Connect to your local filesystem:** ([Documentation](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem))
*   **Headless Mode:** Run OpenHands in a scriptable mode. ([Documentation](https://docs.all-hands.dev/usage/how-to/headless-mode))
*   **CLI Mode:** Interact with OpenHands through a friendly CLI. ([Documentation](https://docs.all-hands.dev/usage/how-to/cli-mode))
*   **GitHub Action:** Integrate OpenHands with GitHub actions. ([Documentation](https://docs.all-hands.dev/usage/how-to/github-action))

## Important Considerations

*   **Multi-Tenant Environments:** OpenHands is designed for single-user, local workstation use and is not intended for multi-tenant deployments.
*   **Hardened Docker Installation:** If running on a public network, secure your deployment using the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

## Resources

*   **Documentation:** For comprehensive guides, tutorials, and advanced configuration options, please visit the [documentation](https://docs.all-hands.dev/usage/getting-started).
*   **Troubleshooting:** If you encounter any issues, consult the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).
*   **Development:** Contribute to the project by referring to [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

## Join the Community

*   **Slack:** Connect with the community for discussions on research, architecture, and development: [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   **Discord:** Engage in general discussion and feedback: [Join our Discord server](https://discord.gg/ESHStjSjD4)
*   **GitHub Issues:** Report issues or contribute ideas: [Read or post Github Issues](https://github.com/All-Hands-AI/OpenHands/issues)
*   **Community:** See more about the community in [COMMUNITY.md](./COMMUNITY.md) or find details on contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Roadmap

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

## Progress

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

OpenHands is distributed under the [MIT License](./LICENSE).

## Acknowledgements

We are deeply grateful for every contribution and the open-source projects that enable OpenHands. Detailed credits can be found in [CREDITS.md](./CREDITS.md).

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