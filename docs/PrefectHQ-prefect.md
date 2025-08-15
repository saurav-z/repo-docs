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

# Prefect: Build and Orchestrate Data Pipelines with Ease

Prefect is an open-source workflow orchestration framework designed to transform your Python scripts into robust, production-ready data pipelines.  [Visit the original repository](https://github.com/PrefectHQ/prefect).

## Key Features

*   **Simplified Workflow Creation:** Easily define workflows using Python code with `@flow` and `@task` decorators.
*   **Scheduling and Automation:** Schedule your workflows to run at specific times or trigger them based on events.
*   **Resilience and Reliability:**  Built-in features for retries, error handling, and dependency management ensure your pipelines are robust.
*   **Monitoring and Observability:** Track workflow activity through a self-hosted server or the managed Prefect Cloud dashboard.
*   **Deployment Options:** Deploy workflows to various environments, including local machines, cloud platforms, and Kubernetes.
*   **Integration with Existing Tools:** Integrate Prefect with your existing data tools and services.

## Getting Started

Prefect requires Python 3.9+.

**Installation:**

Install the latest version of Prefect using pip:

```bash
pip install -U prefect
```

or using `uv`:

```bash
uv add prefect
```

**Example:**

Here's a simple example of a Prefect workflow that fetches the number of GitHub stars from a repository:

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

**Running Your Workflow:**

1.  **Start the Prefect Server:**
    ```bash
    prefect server start
    ```

2.  **Open the UI:** Access the Prefect UI at http://localhost:4200 to monitor your workflow.

3.  **Deployment and Scheduling:**  Transform your script into a deployment and schedule it to run every minute by changing the last line:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Prefect Cloud

Prefect Cloud offers a managed workflow orchestration platform for modern data teams, automating over 200 million data tasks monthly.  It empowers organizations to improve engineering productivity, reduce pipeline errors, and optimize compute costs.

*   **Managed Orchestration:**  Offloads the operational overhead of running and maintaining your orchestration infrastructure.
*   **Enhanced Monitoring and Alerting:** Provides advanced dashboards, monitoring tools, and alerting capabilities.
*   **Collaboration Features:** Enables seamless collaboration and workflow management for teams.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or sign up to [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For interacting with Prefect Cloud or a remote Prefect server, explore the [prefect-client](https://pypi.org/project/prefect-client/), a lightweight option for client-side SDK functionality, ideal for ephemeral environments.

## Connect & Contribute

Join a thriving community of over 25,000 practitioners who solve data challenges with Prefect.

### Community Resources

*   🌐 **[Explore the Documentation](https://docs.prefect.io)** - Comprehensive guides and API references
*   💬 **[Join the Slack Community](https://prefect.io/slack)** - Connect with thousands of practitioners
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)** - Help shape the future of the project
*   🔌 **[Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)** - Extend Prefect's capabilities
*   📋 **[Tail the Dev Log](https://dev-log.prefect.io/)** - Prefect's open source development blog

Stay informed with updates through our:

*   📥 **[Newsletter](https://prefect.io/newsletter)**
*   📣 **[Twitter/X](https://x.com/PrefectIO)**
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)**
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)**

Your contributions are highly valued!