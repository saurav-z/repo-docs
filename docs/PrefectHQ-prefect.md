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

# Prefect: Orchestrate Your Data Workflows with Ease

Prefect is a powerful and user-friendly workflow orchestration framework for Python, enabling you to build, monitor, and manage data pipelines effectively.  ([See the original repository](https://github.com/PrefectHQ/prefect))

## Key Features:

*   **Simplified Workflow Definition:**  Transform your Python scripts into production-ready workflows with just a few lines of code using `@flow` and `@task` decorators.
*   **Resilient Pipelines:** Build dynamic data pipelines that react to changes and recover from failures.
*   **Scheduling & Automation:** Easily schedule workflows and automate tasks based on events.
*   **Built-in Capabilities:** Benefit from features like scheduling, caching, retries, and event-based automations.
*   **Monitoring & Observability:** Track workflow activity and monitor performance through a self-hosted Prefect server or managed Prefect Cloud.
*   **Scalability:**  Designed to handle complex and large-scale data pipelines.
*   **Integration:** Integrates with your existing tools and data infrastructure.
*   **Deployment:** Deploy workflows to production environments seamlessly.

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

# Run the flow!
if __name__ == "__main__":
    github_stars(["PrefectHQ/Prefect"])
```

Start the Prefect server:

```bash
prefect server start
```

Access the UI at `http://localhost:4200` to monitor your workflow.

To schedule the workflow:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Prefect Cloud

Prefect Cloud offers a managed workflow orchestration solution for modern data enterprises. It automates millions of data tasks monthly, increasing engineering productivity and reducing pipeline errors.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or sign up to [try it](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For communicating with Prefect Cloud or a remote Prefect server, explore the [prefect-client](https://pypi.org/project/prefect-client/), a lighter-weight client-side SDK for ephemeral environments.

## Connect & Contribute

Join the Prefect community of over 25,000 practitioners.

### Community Resources

*   🌐 **[Explore the Documentation](https://docs.prefect.io)** - Guides and API references
*   💬 **[Join the Slack Community](https://prefect.io/slack)** - Connect with practitioners
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)** - Help shape the future
*   🔌 **[Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)** - Extend Prefect
*   📋 **[Tail the Dev Log](https://dev-log.prefect.io/)** - Prefect's development blog

Your contributions are welcome!