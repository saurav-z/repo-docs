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

# Prefect: Build, Observe, and Operate Data Pipelines in Python

Prefect is a powerful and flexible workflow orchestration framework that simplifies the process of building and managing data pipelines. Visit the [original repository](https://github.com/PrefectHQ/prefect) to learn more.

**Key Features:**

*   **Simplified Workflow Creation:** Transform Python scripts into production-ready workflows with just a few lines of code using `@flow` and `@task` decorators.
*   **Resilient Pipelines:** Build workflows that handle retries, dependencies, and complex branching logic to gracefully recover from unexpected issues.
*   **Scheduling & Automation:** Easily schedule and automate your data pipelines with features like cron-based scheduling and event-driven triggers.
*   **Monitoring & Observability:** Track workflow activity and monitor pipeline performance using the self-hosted Prefect server or the managed Prefect Cloud dashboard.
*   **Deployment Capabilities:** Deploy workflows to production environments effortlessly.
*   **Integration with Existing Tools:** Integrates seamlessly with various tools and services.
*   **Error Handling & Retries:** Implement robust error handling and retries within your workflows.
*   **Team Collaboration Features:** Leverage team collaboration features for improved data workflow management.

## Getting Started

Prefect requires Python 3.9+ and can be installed using pip or uv:

```bash
pip install -U prefect
```

```bash
uv add prefect
```

Here's a simple example to get you started, fetching GitHub stars:

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

Start a Prefect server to visualize your workflow:
```bash
prefect server start
```

Access the UI at `http://localhost:4200`.

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

Prefect Cloud provides a managed workflow orchestration platform for modern data teams. Automating millions of data tasks monthly, Prefect Cloud helps organizations improve engineering productivity, reduce errors, and lower compute costs.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) and [try it](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For interacting with Prefect Cloud or a remote server, use the lightweight [prefect-client](https://pypi.org/project/prefect-client/). Ideal for ephemeral environments.

## Connect & Contribute

Join the Prefect community of over 25,000 practitioners and contribute to its growth!

**Community Resources:**

*   🌐 [Explore the Documentation](https://docs.prefect.io)
*   💬 [Join the Slack Community](https://prefect.io/slack)
*   🤝 [Contribute to Prefect](https://docs.prefect.io/contribute/)
*   🔌 [Create or support a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)
*   📋 [Tail the Dev Log](https://dev-log.prefect.io/)

**Stay Informed:**

*   📥 [Subscribe to our Newsletter](https://prefect.io/newsletter)
*   📣 [Twitter/X](https://x.com/PrefectIO)
*   📺 [YouTube](https://www.youtube.com/@PrefectIO)
*   📱 [LinkedIn](https://www.linkedin.com/company/prefect)

Your contributions are invaluable!