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

# Prefect: The Python Workflow Orchestration Framework for Data Pipelines

Prefect is a powerful Python-based workflow orchestration framework that allows you to build, monitor, and manage complex data pipelines with ease.  [Visit the original repository](https://github.com/PrefectHQ/prefect).

**Key Features:**

*   **Simplified Workflow Creation:**  Turn your scripts into production-ready workflows with just a few lines of code using the `@flow` and `@task` decorators.
*   **Resilient Pipelines:**  Build workflows that can handle retries, dependencies, and complex branching logic.
*   **Scheduling and Automation:** Easily schedule your workflows to run on a regular basis or trigger them based on events.
*   **Monitoring and Observability:** Track workflow activity through a self-hosted Prefect server or the managed Prefect Cloud.
*   **Flexible Deployment:** Deploy workflows to various environments.
*   **Error Handling and Retries:** Built-in capabilities for robust error management.
*   **Integration with Existing Tools:** Seamlessly integrate Prefect with your existing data stack.

## Getting Started

Prefect requires Python 3.9+.

**Installation:**

```bash
pip install -U prefect
```

```bash
uv add prefect
```

**Example:**

Create a simple workflow to fetch GitHub stars:

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

**Run a Prefect Server:**

```bash
prefect server start
```

Access the UI at http://localhost:4200 to monitor your workflow.

**Deploy and Schedule:**

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Prefect Cloud

Prefect Cloud offers a managed workflow orchestration service designed for modern data enterprises.  It streamlines workflow management, enhances engineering productivity, reduces pipeline errors, and minimizes data workflow compute costs.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or [sign up for a free trial](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For client-side functionality in ephemeral execution environments, explore the lightweight [prefect-client](https://pypi.org/project/prefect-client/).

## Connect & Contribute

Join the Prefect community of over 25,000 practitioners and contribute to the future of data orchestration.

**Community Resources:**

*   🌐 **[Documentation](https://docs.prefect.io)**: Comprehensive guides and API references.
*   💬 **[Slack Community](https://prefect.io/slack)**: Connect with practitioners.
*   🤝 **[Contribute](https://docs.prefect.io/contribute/)**:  Shape the project.
*   🔌 **[Integrations](https://docs.prefect.io/contribute/contribute-integrations)**: Support or create new integrations.
*   📋 **[Dev Log](https://dev-log.prefect.io/)**: Prefect's open source development blog.

**Stay Informed:**

*   📥 **[Newsletter](https://prefect.io/newsletter)**: Get Prefect updates.
*   📣 **[Twitter/X](https://x.com/PrefectIO)**: Latest announcements.
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)**: Video tutorials and webinars.
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)**: Company news.