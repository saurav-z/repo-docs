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

# Prefect: Orchestrate Data Pipelines with Ease 🚀

Prefect is a powerful workflow orchestration framework that simplifies building, running, and monitoring data pipelines in Python.  Elevate your scripts into robust, production-ready workflows with Prefect, and build resilient, dynamic data pipelines that adapt to changing conditions.  [Visit the original repository](https://github.com/PrefectHQ/prefect).

## Key Features

*   **Simplified Workflow Creation:** Easily turn Python scripts into automated workflows with decorators.
*   **Resilient Execution:** Built-in support for retries, dependencies, and error handling ensures pipeline reliability.
*   **Scheduling and Automation:** Schedule workflows and trigger them based on events for automated data processing.
*   **Monitoring and Observability:** Track workflow activity and performance through the self-hosted Prefect Server or Prefect Cloud.
*   **Dynamic Pipelines:** Build pipelines that react to the world around them and recover from unexpected changes.
*   **Prefect Cloud Integration:** Leverage a managed service for advanced orchestration, monitoring, and collaboration.

## Getting Started

### Installation

Install Prefect using pip or uv:

```bash
pip install -U prefect
```

or

```bash
uv add prefect
```

### Example: Fetch GitHub Stars

Here's a simple example of a Prefect workflow that fetches the number of GitHub stars from a repository:

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

### Running and Deploying Your Workflow

1.  **Start the Prefect Server:**

    ```bash
    prefect server start
    ```

    Then, open the UI at http://localhost:4200.

2.  **Deploying a Workflow:**  Turn your workflow into a deployment and schedule it:

    ```python
    if __name__ == "__main__":
        github_stars.serve(
            name="first-deployment",
            cron="* * * * *",
            parameters={"repos": ["PrefectHQ/prefect"]}
        )
    ```

    Your workflow will now run on a schedule, or you can trigger it manually via the UI or CLI. Deployments can also respond to events.

    Learn more about deployments, error handling, and integrations in the [Prefect documentation](https://docs.prefect.io/v3/get-started/index?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## Prefect Cloud: Managed Workflow Orchestration

Prefect Cloud provides a fully managed workflow orchestration platform for modern data teams.  It automates millions of data tasks monthly, helping organizations increase engineering productivity, reduce pipeline errors, and lower compute costs. Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or [sign up](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) to try it.

## prefect-client

For client-side functionality when communicating with Prefect Cloud or a remote Prefect server, explore [prefect-client](https://pypi.org/project/prefect-client/), a lighter-weight SDK option, well-suited for ephemeral environments.

## Connect & Contribute

Join the vibrant Prefect community of over 25,000 practitioners!

### Community Resources

*   🌐 **[Documentation](https://docs.prefect.io)**: Comprehensive guides and API references.
*   💬 **[Slack](https://prefect.io/slack)**: Connect with other practitioners.
*   🤝 **[Contribute](https://docs.prefect.io/contribute/)**: Help shape the project's future.
*   🔌 **[Integrations](https://docs.prefect.io/contribute/contribute-integrations)**: Extend Prefect's capabilities.
*   📋 **[Dev Log](https://dev-log.prefect.io/)**: Prefect's open source development blog.

### Stay Informed

*   📥 **[Newsletter](https://prefect.io/newsletter)**: Get the latest Prefect news and updates.
*   📣 **[Twitter/X](https://x.com/PrefectIO)**: Latest updates and announcements.
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)**: Video tutorials and webinars.
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)**: Professional networking and company news.

Your contributions and feedback are critical to Prefect's success!