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

# Prefect: Build Reliable Data Pipelines in Python

Prefect is a powerful Python workflow orchestration framework that transforms your scripts into robust and manageable data pipelines.

**Key Features:**

*   **Simple Workflow Definition:** Define workflows with just a few lines of Python code using `@flow` and `@task` decorators.
*   **Scheduling and Automation:** Schedule workflows to run automatically or trigger them based on events.
*   **Resilience and Reliability:** Built-in features for retries, dependencies, and complex branching logic to handle failures gracefully.
*   **Monitoring and Observability:** Track workflow activity, monitor progress, and debug issues through a self-hosted server or managed Prefect Cloud.
*   **Extensible and Integratable:** Integrate with your existing tools and systems to create powerful data pipelines.

## Getting Started

### Installation

To install Prefect, ensure you have Python 3.9+ and run:

```bash
pip install -U prefect
```

or

```bash
uv add prefect
```

### Example: Building a Simple Workflow

Here's how to build a simple workflow to fetch the number of GitHub stars from a repository:

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

### Running Your Workflow

1.  **Start the Prefect Server:**

    ```bash
    prefect server start
    ```

2.  **View the UI:** Open your browser and go to `http://localhost:4200` to monitor your workflow's execution.

3.  **Deploying Your Workflow:** Deploy and schedule the workflow to run every minute:

    ```python
    if __name__ == "__main__":
        github_stars.serve(
            name="first-deployment",
            cron="* * * * *",
            parameters={"repos": ["PrefectHQ/prefect"]}
        )
    ```

## Prefect Cloud

Prefect Cloud provides a managed workflow orchestration service for the modern data enterprise, automating over 200 million data tasks monthly for organizations like Progressive Insurance and Cash App.

*   **Learn More:** [Prefect Cloud vs. OSS](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   **Try it:** [Sign up for Prefect Cloud](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)

## prefect-client

For interacting with Prefect Cloud or a remote Prefect server, explore the lightweight [prefect-client](https://pypi.org/project/prefect-client/).

## Connect & Contribute

Join a thriving community of over 25,000 practitioners to solve data challenges with Prefect!  [Learn more about Prefect on GitHub](https://github.com/PrefectHQ/prefect).

### Community Resources

*   🌐 **Documentation:** [https://docs.prefect.io](https://docs.prefect.io)
*   💬 **Slack:** [https://prefect.io/slack](https://prefect.io/slack)
*   🤝 **Contribute:** [https://docs.prefect.io/contribute/](https://docs.prefect.io/contribute/)
*   🔌 **Integrations:** [https://docs.prefect.io/contribute/contribute-integrations](https://docs.prefect.io/contribute/contribute-integrations)

### Stay Informed

*   📥 **Newsletter:** [https://prefect.io/newsletter](https://prefect.io/newsletter)
*   📣 **Twitter/X:** [https://x.com/PrefectIO](https://x.com/PrefectIO)
*   📺 **YouTube:** [https://www.youtube.com/@PrefectIO](https://www.youtube.com/@PrefectIO)
*   📱 **LinkedIn:** [https://www.linkedin.com/company/prefect](https://www.linkedin.com/company/prefect)