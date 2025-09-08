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

# Prefect: The Python Workflow Orchestration Framework for Data Pipelines

Prefect simplifies data pipeline creation by transforming Python scripts into production-ready workflows, offering robust features like scheduling, retries, and event-driven automation.

[<img src="https://img.shields.io/github/stars/PrefectHQ/prefect?style=social" alt="GitHub Repo stars">](https://github.com/PrefectHQ/prefect)

**Key Features:**

*   **Easy Workflow Definition:** Quickly convert Python scripts into orchestrated workflows using intuitive decorators.
*   **Resilient Execution:** Built-in features for retries, dependencies, and complex branching logic to handle unexpected changes.
*   **Scheduling and Automation:** Schedule workflows and trigger them based on events for automated data processing.
*   **Monitoring and Observability:** Track workflow activity through a self-hosted Prefect server or the managed Prefect Cloud dashboard.
*   **Scalability:**  Designed to handle complex data pipelines and scale with your growing needs.
*   **Integration:** Integrates with your existing tools and infrastructure.

## Getting Started

Prefect requires Python 3.9+.

**Installation:**

```bash
pip install -U prefect
```

or

```bash
uv add prefect
```

**Example:** Create a simple workflow to fetch GitHub stars for a repository:

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

**Running the Workflow:**

1.  **Start the Prefect Server:**
    ```bash
    prefect server start
    ```
    Access the UI at http://localhost:4200.

2.  **Deploy and Schedule (Optional):** To run the workflow on a schedule:

    ```python
    if __name__ == "__main__":
        github_stars.serve(
            name="first-deployment",
            cron="* * * * *",
            parameters={"repos": ["PrefectHQ/prefect"]}
        )
    ```

You can also run deployments from the UI or CLI, and trigger them based on [events](https://docs.prefect.io/latest/automate/?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

**Next Steps:** Explore the [Prefect Documentation](https://docs.prefect.io/v3/get-started/index?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) for more information on:

*   Deploying flows to production.
*   Adding error handling and retries.
*   Integrating with your existing tools.
*   Setting up team collaboration features.

## Prefect Cloud

Prefect Cloud provides a managed workflow orchestration solution for modern data teams. Automate over 200 million data tasks monthly and empower your team to increase productivity, reduce errors, and cut costs.  Learn more at [Prefect Cloud](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For interacting with Prefect Cloud or remote Prefect servers, use the [prefect-client](https://pypi.org/project/prefect-client/).  This is a lighter-weight option ideal for ephemeral environments.

## Connect & Contribute

Join the Prefect community of 25,000+ practitioners!

### Community Resources

*   🌐 [Explore the Documentation](https://docs.prefect.io) - Comprehensive guides and API references
*   💬 [Join the Slack Community](https://prefect.io/slack) - Connect with thousands of practitioners
*   🤝 [Contribute to Prefect](https://docs.prefect.io/contribute/) - Help shape the future of the project
*   🔌 [Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations) - Extend Prefect's capabilities
*   📋 [Tail the Dev Log](https://dev-log.prefect.io/) - Prefect's open source development blog

Your contributions, questions, and ideas are always welcome!