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

# Prefect: Build Robust Data Pipelines with Python

**Prefect is a powerful, open-source workflow orchestration framework that simplifies the creation, monitoring, and management of data pipelines.**

*   **Simplified Workflow Creation:** Easily turn Python scripts into production-ready workflows with just a few lines of code using `@flow` and `@task` decorators.
*   **Resilient Execution:** Benefit from built-in features like retries, caching, scheduling, and event-driven automation to build pipelines that handle unexpected issues.
*   **Comprehensive Monitoring:** Track workflow activity with a self-hosted Prefect server or the managed Prefect Cloud dashboard.
*   **Scalable & Dynamic:** Build data pipelines that react to their environment and recover from failures.

## Getting Started

### Installation

Ensure you have Python 3.9+ installed.  Install Prefect using pip or uv:

```bash
pip install -U prefect
```

or

```bash
uv add prefect
```

### Example

Here's a quick example to get you started:

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

### Running the Example

1.  **Start the Prefect Server:**

    ```bash
    prefect server start
    ```

2.  **Run the Python Script:** Execute the Python script above.

3.  **Access the UI:** Open http://localhost:4200 in your browser to monitor your workflow.

### Deployment and Scheduling

Turn your script into a scheduled deployment:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

Now your workflow will run on a schedule. You can also run deployments manually from the UI or CLI and trigger them in response to events.

## Prefect Cloud

Prefect Cloud provides enterprise-grade workflow orchestration. Automate over 200 million data tasks monthly and empower your team.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or [try it now](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

The [prefect-client](https://pypi.org/project/prefect-client/) is available for interacting with Prefect Cloud or a remote server, ideal for use in ephemeral environments.

## Connect and Contribute

Join the vibrant Prefect community.

### Community Resources

*   🌐 [Explore the Documentation](https://docs.prefect.io)
*   💬 [Join the Slack Community](https://prefect.io/slack)
*   🤝 [Contribute to Prefect](https://docs.prefect.io/contribute/)
*   🔌 [Integrations](https://docs.prefect.io/contribute/contribute-integrations)

### Stay Informed

*   📥 [Subscribe to our Newsletter](https://prefect.io/newsletter)
*   📣 [Twitter/X](https://x.com/PrefectIO)
*   📺 [YouTube](https://www.youtube.com/@PrefectIO)
*   📱 [LinkedIn](https://www.linkedin.com/company/prefect)

Your contributions are welcome!  [Visit the original repository](https://github.com/PrefectHQ/prefect) to learn more and contribute.