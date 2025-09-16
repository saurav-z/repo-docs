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

# Prefect: The Python-Based Workflow Orchestration Tool for Data Pipelines

Prefect is a powerful and flexible workflow orchestration framework that empowers data teams to build, run, and monitor robust data pipelines in Python.  [Learn more about Prefect on GitHub](https://github.com/PrefectHQ/prefect).

**Key Features:**

*   **Simplified Workflow Creation:** Easily turn Python scripts into production-ready workflows using intuitive decorators.
*   **Resilient Pipelines:** Build pipelines that can handle retries, dependencies, and complex branching logic, ensuring data integrity.
*   **Scheduling & Automation:** Schedule workflows, trigger them based on events, and automate your data processes with ease.
*   **Monitoring & Observability:** Track workflow activity and monitor performance with a self-hosted server or the managed Prefect Cloud dashboard.
*   **Built-in Features:** Leverage built-in capabilities like caching, retries, and event-based automations to streamline your pipelines.

## Getting Started

Prefect requires Python 3.9+ to get started. Install Prefect using pip or uv:

```bash
pip install -U prefect
```

or

```bash
uv add prefect
```

Here's a basic example showing how to use Prefect to fetch GitHub stars:

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

Start the Prefect server and view your workflow's progress:

```bash
prefect server start
```

Navigate to http://localhost:4200 to explore the Prefect UI.  To schedule the workflow:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

Explore the documentation for information on deploying flows to production, adding error handling, integrating with tools, and setting up team collaboration features.

## Prefect Cloud

Prefect Cloud offers workflow orchestration for modern data enterprises. Automate over 200 million tasks monthly and increase engineering productivity.

Explore Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss) and sign up to [try it for yourself](https://app.prefect.cloud).

## prefect-client

For interacting with Prefect Cloud or a remote server, consider the [prefect-client](https://pypi.org/project/prefect-client/), a lighter-weight package for client-side functionality.

## Connect & Contribute

Join the Prefect community and connect with over 25,000 data professionals:

### Community Resources

*   🌐 **[Explore the Documentation](https://docs.prefect.io)** - Comprehensive guides and API references
*   💬 **[Join the Slack Community](https://prefect.io/slack)** - Connect with thousands of practitioners
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)** - Help shape the future of the project
*   🔌 **[Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)** - Extend Prefect's capabilities
*   📋 **[Tail the Dev Log](https://dev-log.prefect.io/)** - Prefect's open source development blog

### Stay Informed

*   📥 **[Subscribe to our Newsletter](https://prefect.io/newsletter)** - Get the latest Prefect news and updates
*   📣 **[Twitter/X](https://x.com/PrefectIO)** - Latest updates and announcements
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)** - Video tutorials and webinars
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)** - Professional networking and company news

Your contributions are valuable to the Prefect community!