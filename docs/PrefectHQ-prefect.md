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

# Prefect: Build Resilient Data Pipelines in Python

**Prefect is a powerful workflow orchestration framework that transforms your Python scripts into production-ready data pipelines.**  Automate, monitor, and manage your data workflows with ease.  [Visit the Prefect repository on GitHub](https://github.com/PrefectHQ/prefect).

**Key Features:**

*   **Python-Native:** Define workflows using familiar Python syntax.
*   **Resilient Execution:**  Handles retries, dependencies, and complex branching logic automatically.
*   **Scheduling & Automation:** Schedule workflows and trigger them based on events.
*   **Monitoring & Observability:** Track workflow activity with a self-hosted server or Prefect Cloud.
*   **Caching & State Management:** Optimize performance with built-in caching and state management.
*   **Event-Driven Workflows:**  React to changes in your data or environment.
*   **Seamless Integration:**  Integrates with your existing tools and systems.

## Getting Started

Prefect requires Python 3.9+.

**Installation:**

Install the latest version of Prefect using pip or uv:

```bash
pip install -U prefect
```

```bash
uv add prefect
```

**Example:**

Create a simple Python script to define a Prefect flow and task:

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

**Run and Monitor:**

1.  Start the Prefect server:

    ```bash
    prefect server start
    ```

2.  Open the UI: [http://localhost:4200](http://localhost:4200) to visualize your workflow runs.

3.  Deploy and Schedule: Convert the flow to a deployment:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Prefect Cloud

Prefect Cloud offers enterprise-grade workflow orchestration.  It simplifies data pipeline management, enhances engineering productivity, and reduces errors.  Used by leading organizations, Prefect Cloud automates millions of data tasks monthly.

*   **Learn more:** [Prefect Cloud vs. OSS](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   **Try it:** [Sign up for Prefect Cloud](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)

## prefect-client

For client-side functionality, especially in ephemeral environments, explore the lightweight [prefect-client](https://pypi.org/project/prefect-client/).

## Connect & Contribute

Join the Prefect community and help shape the future of data orchestration!

### Community Resources

*   🌐 **[Documentation](https://docs.prefect.io):** Comprehensive guides and API references.
*   💬 **[Slack Community](https://prefect.io/slack):** Connect with fellow practitioners.
*   🤝 **[Contribute](https://docs.prefect.io/contribute/):** Help improve Prefect.
*   🔌 **[Integrations](https://docs.prefect.io/contribute/contribute-integrations):** Support or create new integrations.

### Stay Informed

*   📥 **[Newsletter](https://prefect.io/newsletter):** Get the latest updates.
*   📣 **[Twitter/X](https://x.com/PrefectIO):** Latest updates and announcements
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO):** Video tutorials and webinars.
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect):** Company and professional news.

Your contributions are highly valued!