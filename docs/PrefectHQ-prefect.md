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

# Prefect: The Python Workflow Orchestrator for Data Pipelines

Prefect is a powerful, open-source workflow orchestration platform built for data scientists and engineers to build, run, and monitor complex data pipelines with ease.  [See the original repo](https://github.com/PrefectHQ/prefect).

**Key Features:**

*   ✅ **Simplified Workflow Creation:** Easily convert Python scripts into production-ready workflows with decorators.
*   ✅ **Robust Error Handling:** Built-in retry mechanisms and dependency management to ensure pipeline resilience.
*   ✅ **Dynamic Data Pipelines:**  Create workflows that react to changing data and external events.
*   ✅ **Scheduling and Automation:** Schedule and automate your workflows with built-in tools.
*   ✅ **Real-time Monitoring:** Track workflow activity and performance through a self-hosted server or Prefect Cloud.
*   ✅ **Scalable Infrastructure:** Designed to scale with your data needs, from small projects to enterprise-level deployments.
*   ✅ **Caching for Efficiency:** Implement caching strategies to avoid redundant computations and optimize workflow execution.
*   ✅ **Extensive Integrations:** Connect Prefect with your existing tools and services through a wide array of integrations.

## Getting Started

Prefect requires Python 3.9 or higher.

**Installation:**

```bash
pip install -U prefect
```

or

```bash
uv add prefect
```

**Example:** Create a simple workflow to fetch GitHub stars:

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

**Run the Prefect Server:**

```bash
prefect server start
```

Then access the UI at `http://localhost:4200` to monitor your workflow.

**Deploy and Schedule:**

Turn your workflow into a deployment and schedule it to run every minute:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

**Next Steps:**

*   [Deploying flows to production environments](https://docs.prefect.io/v3/deploy?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   [Adding error handling and retries](https://docs.prefect.io/v3/develop/write-tasks#retries?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   [Integrating with your existing tools](https://docs.prefect.io/integrations/integrations?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   [Setting up team collaboration features](https://docs.prefect.io/v3/manage/cloud/manage-users/manage-teams#manage-teams?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)

## Prefect Cloud

Prefect Cloud provides workflow orchestration for modern data enterprises. It automates over 200 million data tasks monthly and helps organizations increase engineering productivity, reduce pipeline errors, and cut compute costs.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or sign up to [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For client-side functionality, especially in ephemeral environments, explore the [prefect-client](https://pypi.org/project/prefect-client/).

## Connect & Contribute

Join the Prefect community of over 25,000 practitioners!

**Community Resources:**

*   🌐 [Explore the Documentation](https://docs.prefect.io)
*   💬 [Join the Slack Community](https://prefect.io/slack)
*   🤝 [Contribute to Prefect](https://docs.prefect.io/contribute/)
*   🔌 [Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)
*   📋 [Tail the Dev Log](https://dev-log.prefect.io/)

**Stay Informed:**

*   📥 [Subscribe to our Newsletter](https://prefect.io/newsletter)
*   📣 [Twitter/X](https://x.com/PrefectIO)
*   📺 [YouTube](https://www.youtube.com/@PrefectIO)
*   📱 [LinkedIn](https://www.linkedin.com/company/prefect)

Your contributions and feedback are highly valued.