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


# Prefect: Build, Observe, and React to Your Data Workflows

Prefect is a powerful, open-source workflow orchestration framework designed to build, observe, and react to your data pipelines in Python, offering robust features and a user-friendly experience.  [Visit the original repository](https://github.com/PrefectHQ/prefect).

**Key Features:**

*   **Simplified Workflow Creation:** Easily transform Python scripts into production-ready workflows with a few lines of code.
*   **Resilient Pipelines:** Build data pipelines that automatically handle retries, dependencies, and complex branching logic, ensuring data integrity.
*   **Automated Scheduling:** Schedule your workflows with ease, enabling automated and repeatable data processes.
*   **Real-time Monitoring and Observability:** Track workflow activity and performance using a self-hosted Prefect server or the managed Prefect Cloud dashboard.
*   **Event-Driven Automation:**  Trigger workflows in response to events, enabling dynamic and reactive data pipelines.
*   **Caching:** Optimize performance and reduce compute costs with smart caching capabilities.

## Getting Started

Prefect requires Python 3.9+.  Install the latest version using:

```bash
pip install -U prefect
```

or

```bash
uv add prefect
```

Here's a simple example that demonstrates how to use Prefect to orchestrate a workflow:

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

Start the Prefect server to monitor your workflows:

```bash
prefect server start
```

Access the UI at http://localhost:4200.

Deploy the workflow and schedule it to run every minute:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

For more information, explore the [documentation](https://docs.prefect.io/v3/get-started/index?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none), which covers topics such as:

*   Deploying flows to production environments
*   Adding error handling and retries
*   Integrating with existing tools
*   Setting up team collaboration features

## Prefect Cloud

Prefect Cloud provides enterprise-grade workflow orchestration, empowering organizations to streamline data pipelines, increase engineering productivity, and reduce operational costs.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) and sign up to [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For interacting with Prefect Cloud or remote servers, utilize the [prefect-client](https://pypi.org/project/prefect-client/), a lighter-weight client-side option for ephemeral execution environments.

## Connect & Contribute

Join the vibrant Prefect community and collaborate on solving data challenges.

### Community Resources

*   🌐 **[Documentation](https://docs.prefect.io)**: Comprehensive guides and API references
*   💬 **[Slack Community](https://prefect.io/slack)**: Connect with practitioners
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)**: Help shape the project
*   🔌 **[Prefect Integrations](https://docs.prefect.io/contribute/contribute-integrations)**:  Extend Prefect's capabilities

### Stay Informed

*   📥 **[Newsletter](https://prefect.io/newsletter)**: Get the latest news and updates
*   📣 **[Twitter/X](https://x.com/PrefectIO)**: Stay up-to-date
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)**: Video tutorials and webinars
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)**: Professional networking and company news