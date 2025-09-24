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

**Prefect simplifies data pipeline creation and management, turning your scripts into reliable, observable, and production-ready workflows.**  [See the original repo](https://github.com/PrefectHQ/prefect)

Key Features:

*   **Python-Native:** Build workflows directly in Python using intuitive decorators.
*   **Reliable Execution:**  Built-in features for retries, dependencies, and error handling to ensure robust pipelines.
*   **Dynamic Workflows:** React to changing data and external events with flexible control flow.
*   **Scheduling & Automation:** Schedule and automate your workflows with ease.
*   **Monitoring & Observability:** Track workflow activity via a self-hosted server or Prefect Cloud for real-time insights.

## Getting Started

Prefect requires Python 3.9+.

### Installation

Install the latest version of Prefect using pip:

```bash
pip install -U prefect
```
OR
```bash
uv add prefect
```

### Example: Fetching GitHub Stars

Here's a simple example demonstrating how to use Prefect to orchestrate a workflow that retrieves the number of GitHub stars for a repository:

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

### Running the Example

1.  **Start the Prefect server:**

    ```bash
    prefect server start
    ```

2.  **View the UI:** Open the Prefect UI at http://localhost:4200 to monitor your workflow's execution.

3.  **Deploy and Schedule:**  Turn your workflow into a deployment, and schedule it to run regularly.

    ```python
    if __name__ == "__main__":
        github_stars.serve(
            name="first-deployment",
            cron="* * * * *",
            parameters={"repos": ["PrefectHQ/prefect"]}
        )
    ```

## Prefect Cloud

Prefect Cloud offers a managed solution for workflow orchestration, designed for modern data enterprises. Automate millions of data tasks monthly with Prefect Cloud to increase engineering productivity, reduce pipeline errors, and cut workflow costs.

*   **Key Benefits:**
    *   Simplified data pipeline management.
    *   Enhanced monitoring and observability.
    *   Reduced operational overhead.
    *   Scalable and reliable workflow execution.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or sign up to [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

The `prefect-client` library is a lighter-weight option for interacting with Prefect Cloud or a remote Prefect server, ideal for ephemeral environments.

## Connect & Contribute

Join the Prefect community of over 25,000 practitioners.  Contribute, ask questions, and help shape the future of data workflow orchestration!

### Community Resources

*   🌐 **[Explore the Documentation](https://docs.prefect.io)** - Comprehensive guides and API references
*   💬 **[Join the Slack Community](https://prefect.io/slack)** - Connect with thousands of practitioners
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)** - Help shape the future of the project
*   🔌 **[Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)** - Extend Prefect's capabilities
*   📋 **[Tail the Dev Log](https://dev-log.prefect.io/)** - Prefect's open-source development blog
*   📥 **[Subscribe to our Newsletter](https://prefect.io/newsletter)** - Get the latest Prefect news and updates
*   📣 **[Twitter/X](https://x.com/PrefectIO)** - Latest updates and announcements
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)** - Video tutorials and webinars
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)** - Professional networking and company news