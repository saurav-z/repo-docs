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

# Prefect: Build, Observe, and Manage Production-Ready Data Pipelines in Python

Prefect is a powerful workflow orchestration framework that transforms your Python scripts into robust, production-ready data pipelines.  [See the original repository](https://github.com/PrefectHQ/prefect).

## Key Features of Prefect

*   **Simplified Workflow Creation:** Easily define and orchestrate workflows using Python decorators.
*   **Resilient Pipelines:** Build pipelines that can automatically retry tasks, handle dependencies, and recover from failures.
*   **Scheduling & Automation:** Schedule workflows to run at specific times or trigger them based on events.
*   **Real-time Monitoring:** Track workflow activity and performance through a self-hosted server or the managed Prefect Cloud.
*   **Caching & State Management:**  Leverage built-in caching and state management for efficient data processing.
*   **Event-Driven Automations:** Trigger workflows in response to external events for dynamic data processing.
*   **Scalability & Flexibility:** Design pipelines that can adapt to changing data volumes and business requirements.

## Getting Started

Install Prefect using pip:

```bash
pip install -U prefect
```
or using `uv`:
```bash
uv add prefect
```

Here's a basic example of how to use Prefect to orchestrate a simple workflow:

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


# run the flow!
if __name__ == "__main__":
    github_stars(["PrefectHQ/Prefect"])
```

Then, start the Prefect server:

```bash
prefect server start
```

Access the UI at `http://localhost:4200` to monitor your workflow.

To schedule your workflow, turn it into a deployment and schedule it to run every minute:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Prefect Cloud

Prefect Cloud provides a fully managed platform for workflow orchestration, empowering data teams to increase productivity, reduce errors, and optimize costs.  It automates over 200 million data tasks monthly for organizations like Progressive Insurance and Cash App.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or sign up to [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For client-side functionality, especially in ephemeral environments, explore the lightweight [prefect-client](https://pypi.org/project/prefect-client/).

## Connect & Contribute

Join the Prefect community of over 25,000 practitioners!

### Community Resources

*   🌐 [Explore the Documentation](https://docs.prefect.io) - Comprehensive guides and API references
*   💬 [Join the Slack Community](https://prefect.io/slack) - Connect with thousands of practitioners
*   🤝 [Contribute to Prefect](https://docs.prefect.io/contribute/) - Help shape the future of the project
*   🔌 [Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations) - Extend Prefect's capabilities
*   📋 [Tail the Dev Log](https://dev-log.prefect.io/) - Prefect's open source development blog

### Stay Informed

*   📥 [Subscribe to our Newsletter](https://prefect.io/newsletter) - Get the latest Prefect news and updates
*   📣 [Twitter/X](https://x.com/PrefectIO) - Latest updates and announcements
*   📺 [YouTube](https://www.youtube.com/@PrefectIO) - Video tutorials and webinars
*   📱 [LinkedIn](https://www.linkedin.com/company/prefect) - Professional networking and company news