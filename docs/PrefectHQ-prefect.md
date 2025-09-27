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


# Prefect: The Workflow Orchestration Framework for Data Pipelines

**Prefect simplifies data pipeline creation and management, empowering data teams to build resilient and dynamic workflows with ease.**  [Go to the original repository](https://github.com/PrefectHQ/prefect)

**Key Features:**

*   **Simplified Workflow Creation:** Transforms Python scripts into production-ready workflows with just a few lines of code.
*   **Resilient Pipelines:**  Built-in features for retries, dependencies, and error handling to ensure workflow reliability.
*   **Scheduling & Automation:** Schedule workflows, trigger them based on events, and automate complex data processes.
*   **Real-Time Monitoring:**  Track workflow activity and performance through the self-hosted Prefect server or Prefect Cloud.
*   **Event-Driven Architecture:** React to changes in your data and environment with event-based automations.
*   **Flexible Deployment:** Deploy workflows to various environments with options for scheduling and monitoring.
*   **Caching and Dependency Management:** Improve performance and reliability with caching and dependency features.

## Getting Started

Prefect requires Python 3.9+ to run.

**Installation:**

```bash
pip install -U prefect
```

or

```bash
uv add prefect
```

**Example:**

Create a Python file using `flow` and `task` decorators. This example shows how to fetch the number of GitHub stars from a repository:

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

**Running the example:**

1.  Start the Prefect server:
    ```bash
    prefect server start
    ```

2.  Open the UI at `http://localhost:4200` to view the workflow execution.

3.  Turn the script into a deployment and schedule to run every minute.  Change the last line of your script to:

    ```python
    if __name__ == "__main__":
        github_stars.serve(
            name="first-deployment",
            cron="* * * * *",
            parameters={"repos": ["PrefectHQ/prefect"]}
        )
    ```

**Next Steps:**

*   [Deploying flows to production environments](https://docs.prefect.io/v3/deploy)
*   [Adding error handling and retries](https://docs.prefect.io/v3/develop/write-tasks#retries)
*   [Integrating with your existing tools](https://docs.prefect.io/integrations/integrations)
*   [Setting up team collaboration features](https://docs.prefect.io/v3/manage/cloud/manage-users/manage-teams#manage-teams)

## Prefect Cloud

Prefect Cloud provides workflow orchestration for modern data enterprises, automating over 200 million tasks monthly.  It helps organizations increase engineering productivity, reduce pipeline errors, and optimize costs.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss) or sign up to [try it for yourself](https://app.prefect.cloud).

## prefect-client

For communicating with Prefect Cloud or a remote Prefect server, check out our
[prefect-client](https://pypi.org/project/prefect-client/). It's ideal for client-side functionality in ephemeral environments.

## Connect & Contribute

Join the thriving Prefect community with over 25,000 practitioners!

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