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

# Prefect: Build, Observe, and Orchestrate Data Pipelines with Ease

Prefect is a powerful and user-friendly workflow orchestration framework designed to simplify and streamline the creation, monitoring, and management of your data pipelines in Python.  [Check out the original repo](https://github.com/PrefectHQ/prefect).

**Key Features:**

*   **Simplified Workflow Definition:** Easily convert Python scripts into production-ready workflows using intuitive `@flow` and `@task` decorators.
*   **Resilient and Dynamic Pipelines:** Build data pipelines that automatically handle retries, dependencies, and complex logic, adapting to changing environments.
*   **Scheduling and Automation:** Schedule workflows with ease and trigger them based on events, ensuring timely data processing.
*   **Comprehensive Monitoring and Observability:** Track workflow activity and visualize results using a self-hosted Prefect server or the managed Prefect Cloud dashboard.
*   **Built-in Features:** Benefit from features like scheduling, caching, retries, and event-based automations.
*   **Flexible Deployment Options:** Deploy workflows to various environments, from local machines to production systems, with minimal configuration.

## Getting Started

Prefect requires Python 3.9+.

**Installation:**

Install the latest version of Prefect using pip or uv:

```bash
pip install -U prefect
```

```bash
uv add prefect
```

**Example:**

Here's a simple example of a Prefect workflow that fetches the number of GitHub stars for a repository:

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

To run the workflow, start the Prefect server and open the UI at `http://localhost:4200`:

```bash
prefect server start
```

To schedule the workflow, create a deployment:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Prefect Cloud

Prefect Cloud is a managed platform that provides enterprise-grade workflow orchestration.  It helps increase engineering productivity, reduce pipeline errors, and cut data workflow compute costs.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For interacting with Prefect Cloud or a remote Prefect server, explore the [prefect-client](https://pypi.org/project/prefect-client/).  It's a lightweight option for accessing client-side functionality in ephemeral execution environments.

## Connect & Contribute

Join the Prefect community of over 25,000 practitioners!

**Community Resources:**

*   🌐 **[Explore the Documentation](https://docs.prefect.io)**
*   💬 **[Join the Slack Community](https://prefect.io/slack)**
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)**
*   🔌 **[Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)**
*   📋 **[Tail the Dev Log](https://dev-log.prefect.io/)**

**Stay Informed:**

*   📥 **[Subscribe to our Newsletter](https://prefect.io/newsletter)**
*   📣 **[Twitter/X](https://x.com/PrefectIO)**
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)**
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)**

Your contributions and feedback are valuable to the Prefect community!