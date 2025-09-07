<p align="center">
    <img src="https://github.com/PrefectHQ/prefect/assets/3407835/c654cbc6-63e8-4ada-a92a-efd2f8f24b85" width=1000>
</p>

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

# Prefect: The Python Workflow Orchestration Framework

**Prefect simplifies data pipeline building with Python, enabling you to transform scripts into production-ready, resilient workflows.**

*   **Effortless Automation:** Convert scripts into automated, production-ready workflows with just a few lines of code.
*   **Resilient Workflows:** Build pipelines that handle retries, dependencies, and complex logic, adapting to dynamic environments.
*   **Scheduling & Monitoring:** Schedule workflows and monitor activity with a self-hosted Prefect server or the managed Prefect Cloud dashboard.
*   **Event-Driven Automations:** React to changes in your data with event-based automation capabilities.
*   **Built-in Features:** Leverage features like caching, retries, and event-based automations to build robust workflows.

## Getting Started

Prefect requires Python 3.9+.

**Installation:**

```bash
pip install -U prefect
```

```bash
uv add prefect
```

**Example:**

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

**Run Prefect Server:**

```bash
prefect server start
```

Open the UI at `http://localhost:4200` to monitor your workflows.

**Scheduling Deployments:**

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Prefect Cloud

Prefect Cloud provides a fully managed workflow orchestration platform for the modern data enterprise. Automate millions of data tasks monthly with Prefect and improve engineering productivity.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) and [try it out](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For client-side interactions with Prefect Cloud or a remote server, explore the [prefect-client](https://pypi.org/project/prefect-client/).

## Connect & Contribute

Join the Prefect community!

*   **[Explore the Documentation](https://docs.prefect.io)**
*   **[Join the Slack Community](https://prefect.io/slack)**
*   **[Contribute to Prefect](https://docs.prefect.io/contribute/)**
*   **[Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)**
*   **[Tail the Dev Log](https://dev-log.prefect.io/)**
*   **[Subscribe to our Newsletter](https://prefect.io/newsletter)**
*   **[Twitter/X](https://x.com/PrefectIO)**
*   **[YouTube](https://www.youtube.com/@PrefectIO)**
*   **[LinkedIn](https://www.linkedin.com/company/prefect)**

[Back to the top](https://github.com/PrefectHQ/prefect)