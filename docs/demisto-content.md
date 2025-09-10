<!-- Improved README -->
<!-- Cortex XSOAR Content Repository -->
![Cortex XSOAR Content Logo](xsoar_content_logo.png)

[![CircleCI](https://circleci.com/gh/demisto/content.svg?style=svg)](https://circleci.com/gh/demisto/content)
[![Open in Visual Studio Code](https://img.shields.io/badge/Open%20in%20Visual%20Studio%20Code-0078d7.svg?&logo=visual-studio-code)](https://open.vscode.dev/demisto/content)
[![Open in Remote-Containers](https://img.shields.io/static/v1?label=Remote%20-%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=git@github.com:demisto/content.git)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=master&repo=60525392&machine=standardLinux32gb&devcontainer_path=.devcontainer%2Fdevcontainer.json&location=WestEurope)

# Cortex XSOAR Content: Automate and Orchestrate Your Security Operations

**Supercharge your security operations with Cortex XSOAR content, including playbooks, integrations, and scripts, all open-source and ready for collaboration.** You can explore the original repository here: [https://github.com/demisto/content](https://github.com/demisto/content).

Cortex XSOAR (formerly Demisto) provides a comprehensive platform for security automation and orchestration. This repository provides a wealth of content, enabling security teams to streamline their workflows and improve their incident response capabilities.

Clicking the VS Code badge above or [here](https://vscode.dev/redirect?url=vscode://ms-vscode-remote-containers/cloneInVolume?url=git@github.com:demisto/content.git) will open VS Code, install the `Remote-Containers` extension (if not installed), clone the source code into a container volume, and spins up a development container, configured with all recommended settings.

## Key Features

*   **Playbooks:** Automate incident response with visual playbooks.
*   **Scripts:** Extend platform capabilities with custom Python or JavaScript scripts.
*   **Integrations:** Connect Cortex XSOAR with your existing security tools and services, written in Javascript or Python.
*   **Reports:** Generate insightful reports in JSON format.
*   **Open Source & Collaborative:** Benefit from a community-driven approach.

## Documentation and Contributing

*   **Content Developer Portal:**  For detailed information on developing and contributing content, visit the [Content Developer Portal](https://xsoar.pan.dev/).
*   **Contribution Guide:**  Learn how to contribute content through the [Content Contribution Guide](https://xsoar.pan.dev/docs/contributing/contributing).

## Content Overview

### Playbooks

The platform's visual playbook editor allows you to add and modify tasks, create control flow, and automate everything with your existing security tools, services, and products.

### Scripts

Write your own scripts to customize your security operations tasks in Python or Javascript.

### Integrations

These enable the Cortex XSOAR Platform to orchestrate security and IT products. Each integration provides capabilities in the form of commands and each command usually reflects a product capability (API) and returns both a human-readable and computer-readable response.

### Docker

Use Docker to run python scripts and integrations in a controlled environment. You can configure an existing docker image from the [Cortex XSOAR Docker Hub Organization](https://hub.docker.com/u/demisto/) or create a new docker image to suite your needs. More information about how to use Docker is available [here](https://demisto.pan.dev/docs/docker).

### Reports

Cortex XSOAR Platform supports flexible reports written in JSON. All of our standard reports calculating various incident statistics and metrics are stored in this repo.

---

Join the [DFIR Community Slack channel](https://www.demisto.com/community/) to connect with the community.