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

Prefect is a powerful, open-source workflow orchestration framework for building, running, and monitoring data pipelines in Python, allowing you to transform scripts into production-ready workflows with ease.  [Check out the original repo](https://github.com/PrefectHQ/prefect).

## Key Features

*   **Simple Workflow Creation:** Easily convert Python scripts into robust, production-ready workflows using intuitive decorators.
*   **Scheduling and Automation:** Schedule workflows to run automatically at specific times or intervals, or trigger them in response to events.
*   **Resilient Pipelines:** Implement automatic retries, dependency management, and complex branching logic for reliable data pipelines.
*   **Monitoring and Observability:** Track workflow activity and performance with a self-hosted Prefect server or a managed Prefect Cloud dashboard.
*   **Deployment & Scaling:** Deploy workflows to various environments and scale them as your data needs grow.
*   **Integration:** Leverage existing tools with Prefect integrations.

## Getting Started

### Installation

Prefect requires Python 3.9+. Install Prefect using pip:

```bash
pip install -U prefect
```

or using `uv`:

```bash
uv add prefect
```

### Example: Fetching GitHub Stars

Here's a simple example of how to use Prefect to fetch the number of GitHub stars from a repository:

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

1.  **Start the Prefect Server:**

    ```bash
    prefect server start
    ```

2.  **Access the UI:**  Open the Prefect UI in your browser at `http://localhost:4200` to monitor your workflow runs.

3.  **Deploying & Scheduling:** To schedule your workflow, turn it into a deployment:

    ```python
    if __name__ == "__main__":
        github_stars.serve(
            name="first-deployment",
            cron="* * * * *",
            parameters={"repos": ["PrefectHQ/prefect"]}
        )
    ```

## Prefect Cloud

Prefect Cloud is a managed workflow orchestration platform for the modern data enterprise. Automate your data tasks, increase engineering productivity, and cut compute costs with Prefect Cloud.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss) or [sign up to try it](https://app.prefect.cloud/).

## prefect-client

For client-side access in ephemeral execution environments, explore the lightweight [prefect-client](https://pypi.org/project/prefect-client/).

## Connect & Contribute

Join the Prefect community of over 25,000 practitioners to collaborate on data solutions.  Your contributions are welcome!

### Community Resources

*   🌐 [Explore the Documentation](https://docs.prefect.io)
*   💬 [Join the Slack Community](https://prefect.io/slack)
*   🤝 [Contribute to Prefect](https://docs.prefect.io/contribute/)
*   🔌 [Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)
*   📋 [Tail the Dev Log](https://dev-log.prefect.io/)

### Stay Informed

*   📥 [Subscribe to our Newsletter](https://prefect.io/newsletter)
*   📣 [Twitter/X](https://x.com/PrefectIO)
*   📺 [YouTube](https://www.youtube.com/@PrefectIO)
*   📱 [LinkedIn](https://www.linkedin.com/company/prefect)