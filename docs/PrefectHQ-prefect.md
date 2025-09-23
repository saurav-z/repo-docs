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

# Prefect: The Python-Native Dataflow Automation Framework

**Prefect is a powerful workflow orchestration framework designed for building, running, and monitoring data pipelines in Python.** Easily transform your Python scripts into robust, production-ready workflows.

[Check out the original repository](https://github.com/PrefectHQ/prefect)

## Key Features

*   **Simplified Workflow Creation:** Transform scripts into workflows with just a few lines of code using `@flow` and `@task` decorators.
*   **Built-in Resilience:** Automatically handles retries, dependencies, and complex branching logic for robust pipelines.
*   **Scheduling & Automation:** Schedule workflows and trigger events.
*   **Monitoring & Observability:** Track workflow activity with Prefect server (self-hosted) or Prefect Cloud.
*   **Flexibility:** Easily integrate with your existing tools and infrastructure.
*   **Scalability:** Designed to scale with your growing data needs.
*   **Deployment Options**: Deploy workflows to various environments.

## Getting Started

Prefect requires Python 3.9+.

**1. Installation:**
```bash
pip install -U prefect
```
or
```bash
uv add prefect
```

**2. Example Workflow:**

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

**3. Run and Monitor:**

```bash
prefect server start
```

Open your browser and go to http://localhost:4200 to see your workflow runs.

**4. Deployment and Scheduling:**

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Prefect Cloud

Prefect Cloud is a managed workflow orchestration platform that offers enhanced features, monitoring, and team collaboration capabilities. Automate over 200 million data tasks monthly and increase engineering productivity.

*   **Enhanced Monitoring & Alerting:**  Get real-time insights and proactive alerts.
*   **Team Collaboration:** Streamline workflow management for your entire team.
*   **Scalability & Reliability:** Leverage a robust platform for your most critical pipelines.
*   **Reduce Pipeline Errors & Costs**: Optimize workflows for efficiency.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) and sign up to [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For interacting with Prefect Cloud or remote servers, the `prefect-client` provides a lighter-weight SDK option.

[Check out prefect-client](https://pypi.org/project/prefect-client/)

## Connect & Contribute

Join the Prefect community of over 25,000 data practitioners.

### Community Resources

*   🌐 [Explore the Documentation](https://docs.prefect.io) - Comprehensive guides and API references
*   💬 [Join the Slack Community](https://prefect.io/slack) - Connect with thousands of practitioners
*   🤝 [Contribute to Prefect](https://docs.prefect.io/contribute/) - Help shape the future of the project
*   🔌 [Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations) - Extend Prefect's capabilities
*   📋 [Tail the Dev Log](https://dev-log.prefect.io/) - Prefect's open source development blog

### Stay Informed

*   📥 [Subscribe to our Newsletter](https://prefect.io/newsletter) - Get the latest Prefect news and updates
*   📣 [Twitter/X](https://x.com/PrefectIO) - Latest updates and announcements
*   📺 [YouTube](https://www.youtube.com/@PrefectIO) - Video tutorials and webinars
*   📱 [LinkedIn](https://www.linkedin.com/company/prefect) - Professional networking and company news