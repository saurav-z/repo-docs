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

# Prefect: Build, Observe, and Orchestrate Data Workflows in Python

Prefect is a powerful and flexible workflow orchestration framework that empowers data teams to build, monitor, and manage robust data pipelines with ease.  [Check out the source code on GitHub](https://github.com/PrefectHQ/prefect).

**Key Features:**

*   **Simplified Workflow Creation:**  Transform your Python scripts into production-ready workflows with minimal code using intuitive decorators.
*   **Resilient Data Pipelines:**  Build workflows that automatically handle retries, dependencies, and complex branching logic to ensure reliability.
*   **Scheduling and Automation:** Easily schedule your workflows to run automatically or trigger them based on events.
*   **Real-time Monitoring:**  Track and monitor your workflow activity through a self-hosted Prefect server or the managed Prefect Cloud dashboard.
*   **Extensible & Integrations:** Integrate with your existing tools and expand functionality with custom integrations.

## Getting Started

Get up and running quickly with Prefect!

**1. Installation:**

Prefect requires Python 3.9+. Install the latest version using pip or uv:

```bash
pip install -U prefect
```
```bash
uv add prefect
```

**2. Example Workflow:**

Create a simple workflow to get you started:

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

**3. Run Your Workflow:**

*   **Start the Prefect server:** `prefect server start` and open the UI at http://localhost:4200.
*   **Deploy and schedule:**

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

Explore the [Prefect documentation](https://docs.prefect.io/v3/get-started/index) to discover more:

*   [Deploying flows to production environments](https://docs.prefect.io/v3/deploy)
*   [Adding error handling and retries](https://docs.prefect.io/v3/develop/write-tasks#retries)
*   [Integrating with your existing tools](https://docs.prefect.io/integrations/integrations)
*   [Setting up team collaboration features](https://docs.prefect.io/v3/manage/cloud/manage-users/manage-teams#manage-teams)

## Prefect Cloud

Prefect Cloud offers a managed workflow orchestration platform for modern data enterprises. Automate your data tasks and reduce errors to save time and money. Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss) or [try it](https://app.prefect.cloud) today.

## prefect-client

For interacting with Prefect Cloud or a remote server, explore the [prefect-client](https://pypi.org/project/prefect-client/). It's a lighter-weight option for client-side access.

## Connect & Contribute

Join the thriving Prefect community of over 25,000 practitioners!

### Community Resources

*   🌐 **[Explore the Documentation](https://docs.prefect.io)** - Guides and API references
*   💬 **[Join the Slack Community](https://prefect.io/slack)** - Connect with practitioners
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)** - Help shape the project
*   🔌 **[Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)** - Extend Prefect's capabilities
*   📋 **[Tail the Dev Log](https://dev-log.prefect.io/)** - Prefect's open source development blog

### Stay Informed

*   📥 **[Subscribe to our Newsletter](https://prefect.io/newsletter)** - Get the latest news
*   📣 **[Twitter/X](https://x.com/PrefectIO)** - Latest updates and announcements
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)** - Video tutorials and webinars
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)** - Company news

Your contributions, questions, and ideas make Prefect better.