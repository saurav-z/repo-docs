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

# Prefect: The Python Workflow Orchestration Framework

Prefect is a powerful, open-source workflow orchestration framework, empowering data teams to build robust and reliable data pipelines in Python. Automate your data tasks with ease! ([See the original repo](https://github.com/PrefectHQ/prefect))

**Key Features:**

*   **Simplified Workflow Creation:** Turn your Python scripts into production-ready workflows with just a few lines of code using `@flow` and `@task` decorators.
*   **Resilient Pipelines:** Build dynamic data pipelines that react to changes and recover from unexpected failures.
*   **Scheduling and Automation:** Schedule workflows, set up retries, manage dependencies, and implement event-based triggers.
*   **Monitoring and Observability:** Track workflow activity with a self-hosted Prefect server or the managed Prefect Cloud dashboard.
*   **Prefect Cloud Integration:** Seamlessly integrate with Prefect Cloud for enterprise-grade workflow orchestration, monitoring, and management.
*   **Retry Mechanisms:** Easily implement retries to handle transient failures gracefully, ensuring pipeline robustness.
*   **Integrations:** Integrate with existing tools to extend Prefect's capabilities and fit your specific needs.

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

# Run the flow
if __name__ == "__main__":
    github_stars(["PrefectHQ/Prefect"])
```

**Running a Prefect Server:**

```bash
prefect server start
```

Access the UI at http://localhost:4200.

**Scheduling Deployments:**

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Prefect Cloud

Prefect Cloud provides a robust, managed platform for workflow orchestration, offering increased engineering productivity, reduced errors, and lower compute costs.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) and try it out [here](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For interacting with Prefect Cloud or a remote Prefect server, explore the lightweight [prefect-client](https://pypi.org/project/prefect-client/).

## Connect & Contribute

Join the Prefect community to collaborate, innovate, and improve data workflows.

### Community Resources
🌐 **[Explore the Documentation](https://docs.prefect.io)** - Comprehensive guides and API references  
💬 **[Join the Slack Community](https://prefect.io/slack)** - Connect with thousands of practitioners  
🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)** - Help shape the future of the project  
 🔌 **[Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)** - Extend Prefect's capabilities

### Stay Informed
📥 **[Subscribe to our Newsletter](https://prefect.io/newsletter)** - Get the latest Prefect news and updates  
📣 **[Twitter/X](https://x.com/PrefectIO)** - Latest updates and announcements  
📺 **[YouTube](https://www.youtube.com/@PrefectIO)** - Video tutorials and webinars  
📱 **[LinkedIn](https://www.linkedin.com/company/prefect)** - Professional networking and company news