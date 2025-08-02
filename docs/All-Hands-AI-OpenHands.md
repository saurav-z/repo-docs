# OpenHands: Code Less, Make More with AI-Powered Software Development (Formerly OpenDevin)

[<img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=social" alt="Stars">](https://github.com/All-Hands-AI/OpenHands/stargazers)
[View on GitHub](https://github.com/All-Hands-AI/OpenHands)

OpenHands is an innovative AI platform that empowers software developers to automate tasks and accelerate their workflow.

**Key Features:**

*   **AI-Powered Agents:** OpenHands agents can perform a wide range of tasks, just like a human developer.
*   **Code Modification & Execution:** Modify code, run commands, and automate development processes.
*   **Web Browsing & API Integration:** Browse the web, call APIs, and integrate external services seamlessly.
*   **Stack Overflow Integration:**  Access and utilize code snippets from Stack Overflow to speed up development.
*   **Cloud & Local Deployment:** Easily get started with OpenHands Cloud or run it locally using Docker.
*   **Community Support:** Join a vibrant community through Slack and Discord to collaborate and get help.

## Getting Started

The easiest way to experience OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), where new users receive $20 in free credits.

### Running OpenHands Locally

You can also run OpenHands locally using Docker.

**1. Pull the Docker Image:**

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.51-nikolaik
```

**2. Run the Container:**

```bash
docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.51-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands-dev/all-hands-ai/openhands:0.51
```

**3. Access OpenHands:**

Open the application in your browser at [http://localhost:3000](http://localhost:3000).  You will be prompted to choose an LLM provider and add an API key. Anthropic's Claude Sonnet 4 works best.

### Other Run Options

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [CLI Mode](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [Github Action](https://docs.all-hands.dev/usage/how-to/github-action)

See [Running OpenHands](https://docs.all-hands.dev/usage/installation) for more information.

## Community & Resources

*   **Documentation:** [https://docs.all-hands.dev/usage/getting-started](https://docs.all-hands.dev/usage/getting-started)
*   **Slack:** [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   **Discord:** [Join our Discord server](https://discord.gg/ESHStjSjD4)
*   **GitHub Issues:** [https://github.com/All-Hands-AI/OpenHands/issues](https://github.com/All-Hands-AI/OpenHands/issues)
*   **Project Roadmap:** [https://github.com/orgs/All-Hands-AI/projects/1](https://github.com/orgs/All-Hands-AI/projects/1)

## Contributing

We welcome contributions from the community!  Please see [CONTRIBUTING.md](https://github.com/All-Hands-AI/OpenHands/blob/main/CONTRIBUTING.md) for details.

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

## Acknowledgements

OpenHands is built with the help of many contributors. See [CREDITS.md](./CREDITS.md) for a list of open-source projects used.

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