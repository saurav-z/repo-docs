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

**Prefect is a powerful workflow orchestration framework that empowers data teams to build, monitor, and manage robust and reliable data pipelines in Python.**

## Key Features

*   **Simplified Workflow Creation:** Transform Python scripts into production-ready workflows with minimal code using Prefect's intuitive `@flow` and `@task` decorators.
*   **Resilient Pipelines:** Build dynamic data pipelines that react to changes and automatically recover from failures.
*   **Scheduling & Automation:** Easily schedule workflows to run at specific times or trigger them based on events.
*   **Advanced Features:** Leverage built-in capabilities for caching, retries, dependencies, and complex branching logic to handle complex data processes.
*   **Monitoring & Observability:** Track workflow activity with a self-hosted Prefect server or a managed Prefect Cloud dashboard.
*   **Integration:** Integrate with your existing tools.
*   **Open Source:** Prefect is free and open-source.

## Getting Started

Prefect requires Python 3.9+. [Install the latest version](https://docs.prefect.io/v3/get-started/install) using pip or uv:

```bash
pip install -U prefect
```

```bash
uv add prefect
```

Here's a quick example of a simple workflow to fetch GitHub stars:

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

1.  **Run the Prefect Server:** Start the Prefect server to monitor your workflows:

    ```bash
    prefect server start
    ```
2.  **View the UI:** Access the Prefect UI at [http://localhost:4200](http://localhost:4200) to observe your workflow runs.

3.  **Deploy and Schedule:** Convert your workflow into a deployment for scheduled execution:

    ```python
    if __name__ == "__main__":
        github_stars.serve(
            name="first-deployment",
            cron="* * * * *",
            parameters={"repos": ["PrefectHQ/prefect"]}
        )
    ```

    Now, your workflow will run every minute, and you can also trigger it manually or via events.

## Prefect Cloud

Prefect Cloud provides managed workflow orchestration for enterprises.

*   Automates over 200 million data tasks monthly.
*   Helps organizations increase engineering productivity, reduce pipeline errors, and cut data workflow compute costs.

Learn more about [Prefect Cloud](https://www.prefect.io/cloud-vs-oss).

## prefect-client

For interacting with Prefect Cloud or a remote Prefect server from ephemeral environments, explore the [prefect-client](https://pypi.org/project/prefect-client/).

## Join the Prefect Community

Connect with over 25,000 data practitioners and contribute to the Prefect project!

### Resources:

*   🌐 **[Documentation](https://docs.prefect.io)**
*   💬 **[Slack Community](https://prefect.io/slack)**
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)**
*   🔌 **[Create Integrations](https://docs.prefect.io/contribute/contribute-integrations)**

### Stay Connected:

*   📥 **[Newsletter](https://prefect.io/newsletter)**
*   📣 **[Twitter/X](https://x.com/PrefectIO)**
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)**
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)**

Your contributions help make Prefect better!

---

**[Back to Top](https://github.com/PrefectHQ/prefect)**