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

# Prefect: Orchestrate Data Pipelines with Ease

Prefect is a powerful, flexible, and user-friendly workflow orchestration framework for Python, making it simple to build, monitor, and manage data pipelines.  [Learn more on GitHub](https://github.com/PrefectHQ/prefect).

**Key Features:**

*   **Simplified Workflow Creation:** Transform Python scripts into production-ready workflows with just a few lines of code.
*   **Resilient Pipelines:** Build dynamic pipelines that handle retries, dependencies, and complex logic.
*   **Scheduling and Automation:** Schedule workflows, trigger them based on events, and automate your data processes.
*   **Monitoring and Observability:** Track workflow activity and performance with a self-hosted server or Prefect Cloud.
*   **Caching:** Utilize built-in caching for improved performance.
*   **Event-Driven Automations:** Respond to external events to trigger workflows.

## Getting Started

Prefect requires Python 3.9+. To install the latest version:

```bash
pip install -U prefect
```

or

```bash
uv add prefect
```

Run the following example to get the number of GitHub stars from a repository.

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

Then, run the Prefect server and view the UI:

```bash
prefect server start
```

To schedule this example to run every minute:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Prefect Cloud

Prefect Cloud provides workflow orchestration for modern data enterprises, automating over 200 million data tasks monthly.  This empowers organizations to improve engineering productivity, reduce errors, and cut compute costs.

Learn more at the [Prefect Cloud Documentation](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or try it for yourself [here](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For interacting with Prefect Cloud or a remote server, explore the [prefect-client](https://pypi.org/project/prefect-client/).  It's a lightweight option, perfect for ephemeral environments.

## Connect & Contribute

Join the vibrant Prefect community!

### Community Resources
*   🌐 **[Documentation](https://docs.prefect.io)**
*   💬 **[Slack Community](https://prefect.io/slack)**
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)**
*   🔌 **[Prefect Integrations](https://docs.prefect.io/contribute/contribute-integrations)**
*   📋 **[Dev Log](https://dev-log.prefect.io/)**

### Stay Informed
*   📥 **[Newsletter](https://prefect.io/newsletter)**
*   📣 **[Twitter/X](https://x.com/PrefectIO)**
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)**
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)**

Your contributions and feedback are essential to the Prefect project.