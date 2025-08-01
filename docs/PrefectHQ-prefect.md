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

# Prefect: Build, Observe, and Orchestrate Data Pipelines in Python

Prefect is a powerful Python-based workflow orchestration platform that empowers data teams to build, monitor, and manage complex data pipelines with ease.  [Explore the Prefect repository on GitHub](https://github.com/PrefectHQ/prefect).

**Key Features:**

*   **Simplified Workflow Creation:** Transform Python scripts into robust, production-ready workflows with minimal code using `@flow` and `@task` decorators.
*   **Resilient Pipelines:** Build pipelines that automatically handle retries, dependencies, and even complex branching logic, ensuring reliability.
*   **Dynamic Workflows:** Create pipelines that react to the world around them and recover from unexpected changes.
*   **Scheduling and Automation:** Easily schedule workflows and trigger them based on events for automated data processing.
*   **Monitoring and Observability:** Track workflow activity and performance with a self-hosted Prefect server or the managed Prefect Cloud dashboard.
*   **Error Handling and Retries:** Built-in mechanisms for handling errors and retrying tasks, ensuring data pipeline resilience.
*   **Integration with Existing Tools:** Seamlessly integrate Prefect with your existing tools and services.

## Getting Started

Prefect requires Python 3.9+.

**Installation:**

Install the latest version of Prefect using either of the following commands:

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

**Running the Workflow:**

1.  **Start the Prefect Server:**  `prefect server start`
2.  **Open the UI:**  Access the Prefect UI at http://localhost:4200.
3.  **Deployment and Scheduling (Optional):** To schedule your workflow, modify your script as follows:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Prefect Cloud

Prefect Cloud offers workflow orchestration for modern data enterprises, automating over 200 million data tasks monthly.

**Key Benefits:**

*   Increased engineering productivity.
*   Reduced pipeline errors.
*   Cut data workflow compute costs.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) and [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For interacting with Prefect Cloud or a remote Prefect server, explore the [prefect-client](https://pypi.org/project/prefect-client/), a lighter-weight option.

## Connect & Contribute

Join the Prefect community to collaborate and contribute to data pipeline innovation.

**Community Resources:**

*   🌐 [Explore the Documentation](https://docs.prefect.io)
*   💬 [Join the Slack Community](https://prefect.io/slack)
*   🤝 [Contribute to Prefect](https://docs.prefect.io/contribute/)
*   🔌 [Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)

**Stay Informed:**

*   📥 [Subscribe to our Newsletter](https://prefect.io/newsletter)
*   📣 [Twitter/X](https://x.com/PrefectIO)
*   📺 [YouTube](https://www.youtube.com/@PrefectIO)
*   📱 [LinkedIn](https://www.linkedin.com/company/prefect)