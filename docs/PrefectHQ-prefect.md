<p align="center"><img src="https://github.com/PrefectHQ/prefect/assets/3407835/c654cbc6-63e8-4ada-a92a-efd2f8f24b85" width=1000></p>

<p align="center">
    <a href="https://pypi.org/project/prefect/" alt="PyPI version">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/prefect?color=0052FF&labelColor=090422" />
    </a>
    <a href="https://pypi.org/project/prefect/" alt="PyPI downloads/month">
        <img alt="Downloads" src="https://img.shields.io/pypi/dm/prefect?color=0052FF&labelColor=090422" />
    </a>
    <a href="https://github.com/prefecthq/prefect/" alt="Stars">
        <img src="https://img.shields.io/github/stars/prefecthq/prefect?color=0052FF&labelColor=090422" />
    </a>
    <a href="https://github.com/prefecthq/prefect/pulse" alt="Activity">
        <img src="https://img.shields.io/github/commit-activity/m/prefecthq/prefect?color=0052FF&labelColor=090422" />
    </a>
    <br>
    <a href="https://prefect.io/slack" alt="Slack">
        <img src="https://img.shields.io/badge/slack-join_community-red.svg?color=0052FF&labelColor=090422&logo=slack" />
    </a>
    <a href="https://www.youtube.com/c/PrefectIO/" alt="YouTube">
        <img src="https://img.shields.io/badge/youtube-watch_videos-red.svg?color=0052FF&labelColor=090422&logo=youtube" />
    </a>
</p>

<p align="center">
    <a href="https://docs.prefect.io/v3/get-started/index?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none">
        Installation
    </a>
    ·
    <a href="https://docs.prefect.io/v3/get-started/quickstart?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none">
        Quickstart
    </a>
    ·
    <a href="https://docs.prefect.io/v3/how-to-guides/workflows/write-and-run?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none">
        Build workflows
    </a>
    ·
    <a href="https://docs.prefect.io/v3/concepts/deployments?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none">
        Deploy workflows
    </a>
    ·
    <a href="https://app.prefect.cloud/?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none">
        Prefect Cloud
    </a>
</p>

# Prefect: Build, Observe, and Orchestrate Data Workflows in Python

Prefect is a powerful workflow orchestration framework that simplifies the creation and management of data pipelines in Python. [Visit the Prefect GitHub Repository](https://github.com/PrefectHQ/prefect) for the latest updates and contributions.

## Key Features

*   **Simplified Workflow Creation:** Easily convert Python scripts into production-ready workflows with minimal code.
*   **Resilient Pipelines:** Build dynamic data pipelines that automatically handle retries, dependencies, and complex branching logic.
*   **Scheduling and Automation:** Schedule workflows and trigger them based on events for automated data processing.
*   **Observability and Monitoring:** Track workflow activity and performance using the Prefect server or Prefect Cloud.
*   **Caching and Retries:** Optimize workflow execution with built-in caching and automatic retry mechanisms.
*   **Event-Driven Architecture:** React to changes in the environment and trigger workflows based on specific events.

## Getting Started

Prefect requires Python 3.9+. To install the latest version, run:

```bash
pip install -U prefect
```

Or, with `uv`:

```bash
uv add prefect
```

### Example: Get GitHub Stars

Here's a simple example demonstrating how to get the number of GitHub stars for a repository:

```python
from prefect import flow, task
import httpx

@task(log_prints=True)
def get_stars(repo: str):
    url = f"https://api.github.com/repos/{repo}"
    count = httpx.get(url).json()["stargazers_count"]
    print(f"{repo} has {count} stars!")

@flow(name="GitHub Stars")
def github_stars(repos: list[str]):
    for repo in repos:
        get_stars(repo)

# Run the flow
if __name__ == "__main__":
    github_stars(["PrefectHQ/Prefect"])
```

To monitor this workflow, start the Prefect server:

```bash
prefect server start
```

Then, access the UI at http://localhost:4200.

### Deployment and Scheduling

To schedule your workflow, convert it into a deployment:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

This creates a deployment that runs every minute. You can also trigger deployments manually from the UI or CLI and set them up to respond to events.

## Prefect Cloud

Prefect Cloud offers enterprise-grade workflow orchestration. Automate millions of data tasks, increase engineering productivity, reduce pipeline errors, and cut costs. Learn more and sign up for a free trial at [Prefect Cloud](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For interacting with Prefect Cloud or a remote Prefect server, explore the lightweight [prefect-client](https://pypi.org/project/prefect-client/). This is a great option for ephemeral environments.

## Connect and Contribute

Join the vibrant Prefect community of over 25,000 practitioners. Contribute, ask questions, and share your ideas!

### Community Resources

*   🌐 **Documentation:** [Explore the Documentation](https://docs.prefect.io)
*   💬 **Slack:** [Join the Slack Community](https://prefect.io/slack)
*   🤝 **Contribute:** [Contribute to Prefect](https://docs.prefect.io/contribute/)
*   🔌 **Integrations:** [Support or Create Prefect Integrations](https://docs.prefect.io/contribute/contribute-integrations)
*   📋 **Dev Log:** [Tail the Dev Log](https://dev-log.prefect.io/)

### Stay Informed

*   📥 **Newsletter:** [Subscribe to our Newsletter](https://prefect.io/newsletter)
*   📣 **Twitter/X:** [Twitter/X](https://x.com/PrefectIO)
*   📺 **YouTube:** [YouTube](https://www.youtube.com/@PrefectIO)
*   📱 **LinkedIn:** [LinkedIn](https://www.linkedin.com/company/prefect)