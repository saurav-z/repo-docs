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

# Prefect: Build, Observe, and Run Data Workflows in Python

**Prefect is the leading open-source workflow orchestration framework for building, observing, and managing data pipelines with ease.** Elevate your Python scripts into reliable, production-ready workflows with built-in features for scheduling, retries, and more.

[<img src="https://img.shields.io/github/stars/PrefectHQ/prefect?style=social" alt="Stars">](https://github.com/PrefectHQ/prefect)

**Key Features:**

*   **Workflow Orchestration:** Transform your Python scripts into robust, production-ready workflows.
*   **Scheduling and Automation:** Easily schedule and automate your data pipelines.
*   **Resilience:** Build workflows that automatically handle retries, dependencies, and error handling.
*   **Observability:** Monitor workflow activity through a self-hosted Prefect server or Prefect Cloud.
*   **Event-Driven Automation:** Trigger workflows based on events, increasing data pipeline responsiveness.
*   **Simplified Deployment:** Deploy workflows with just a few lines of code.
*   **Dynamic Pipelines:** Build pipelines that react to the world around them and recover from unexpected changes.

## Getting Started

Prefect requires Python 3.9+.  Install the latest version of Prefect using pip:

```bash
pip install -U prefect
```

Or with uv:

```bash
uv add prefect
```

Here's a simple example of how to use Prefect to fetch the number of GitHub stars for a repository:

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

Start a Prefect server to monitor your workflow at http://localhost:4200:

```bash
prefect server start
```

To run your workflow on a schedule, use the serve command:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

**[Explore the Prefect documentation](https://docs.prefect.io/v3/get-started/index?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) to learn more about:**

*   Deploying flows to production environments
*   Adding error handling and retries
*   Integrating with your existing tools
*   Setting up team collaboration features

## Prefect Cloud

Prefect Cloud provides enterprise-grade workflow orchestration for modern data teams. Automate millions of data tasks monthly with a platform that helps you increase engineering productivity, reduce pipeline errors, and cut costs.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or [sign up](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) to try it for yourself.

## prefect-client

If you need to communicate with Prefect Cloud or a remote Prefect server, check out the [prefect-client](https://pypi.org/project/prefect-client/). It provides lighter-weight client-side functionality for ephemeral execution environments.

## Connect & Contribute

Join a vibrant community of 25,000+ data practitioners using Prefect.  Prefect thrives on collaboration, technical innovation, and continuous improvement.

### Community Resources

🌐 [Explore the Documentation](https://docs.prefect.io) - Comprehensive guides and API references  
💬 [Join the Slack Community](https://prefect.io/slack) - Connect with thousands of practitioners  
🤝 [Contribute to Prefect](https://docs.prefect.io/contribute/) - Help shape the future of the project  
 🔌 [Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations) - Extend Prefect's capabilities   
📋 [Tail the Dev Log](https://dev-log.prefect.io/) - Prefect's open source development blog

### Stay Informed

📥 [Subscribe to our Newsletter](https://prefect.io/newsletter) - Get the latest Prefect news and updates  
📣 [Twitter/X](https://x.com/PrefectIO) - Latest updates and announcements  
📺 [YouTube](https://www.youtube.com/@PrefectIO) - Video tutorials and webinars  
📱 [LinkedIn](https://www.linkedin.com/company/prefect) - Professional networking and company news

Your contributions, questions, and feedback are essential.  Report bugs, suggest features, or help improve the documentation – your input is invaluable.

[Back to Top](#prefect)