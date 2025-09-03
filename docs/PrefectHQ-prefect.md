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

# Prefect: The Python-Native Workflow Orchestration Framework

Prefect is a powerful, Python-based workflow orchestration framework designed to build, run, and monitor data pipelines with ease. [Learn more about Prefect on GitHub](https://github.com/PrefectHQ/prefect).

**Key Features:**

*   **Simplified Workflow Creation:** Turn your Python scripts into production-ready workflows with just a few lines of code using `@flow` and `@task` decorators.
*   **Resilient Pipelines:** Build robust data pipelines that can handle retries, dependencies, and complex branching logic, ensuring data integrity and reliability.
*   **Scheduling & Automation:** Schedule your workflows to run automatically, integrate event-driven triggers and create automated data pipelines.
*   **Real-time Monitoring:** Track workflow activity and gain deep insights into your data pipelines through the self-hosted Prefect server or the managed Prefect Cloud dashboard.
*   **Comprehensive Error Handling:** Built-in features for error handling and retries that make your data pipelines more reliable.
*   **Integration with Existing Tools:** Works seamlessly with your existing tools and infrastructure.

## Getting Started

To install Prefect, ensure you have Python 3.9+ and run the following command:

```bash
pip install -U prefect
```

or

```bash
uv add prefect
```

Follow the quickstart guide to write, run, and orchestrate your first workflow.  You'll be automating and monitoring data pipelines in minutes!

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

Start a Prefect server to monitor your workflow:

```bash
prefect server start
```

Then, access the UI at http://localhost:4200.

To schedule your workflow, create a deployment:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Prefect Cloud

Prefect Cloud provides a managed workflow orchestration platform for the modern data enterprise, automating over 200 million data tasks monthly.

Explore the power of Prefect Cloud:

*   **Increased Engineering Productivity:** Streamline your data workflow development and operations.
*   **Reduced Pipeline Errors:** Improve data quality and pipeline reliability.
*   **Cut Data Workflow Compute Costs:** Optimize resource utilization and reduce expenses.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss) and [try it for yourself](https://app.prefect.cloud).

## prefect-client

For communication with Prefect Cloud or a remote Prefect server, check out the lightweight [prefect-client](https://pypi.org/project/prefect-client/).

## Connect & Contribute

Join the Prefect community and contribute to the future of data orchestration!

### Community Resources

*   🌐 **[Explore the Documentation](https://docs.prefect.io)** - Comprehensive guides and API references
*   💬 **[Join the Slack Community](https://prefect.io/slack)** - Connect with thousands of practitioners
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)** - Help shape the future of the project
*   🔌 **[Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)** - Extend Prefect's capabilities
*   📋 **[Tail the Dev Log](https://dev-log.prefect.io/)** - Prefect's open source development blog

We welcome your contributions, questions, and ideas.