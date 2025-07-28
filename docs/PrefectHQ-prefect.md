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

# Prefect: Build Robust Data Pipelines in Python

Prefect is a powerful, open-source workflow orchestration platform that simplifies building and managing data pipelines, making it easy to turn Python scripts into reliable, production-ready workflows.

**[Explore the Prefect Repository on GitHub](https://github.com/PrefectHQ/prefect)**

## Key Features

*   **Simplified Workflow Creation:** Easily transform Python scripts into production-ready workflows using intuitive `@flow` and `@task` decorators.
*   **Resilient Pipelines:** Build workflows that automatically handle retries, dependencies, and complex branching logic to gracefully recover from failures.
*   **Scheduling & Automation:** Schedule your workflows to run automatically, trigger them in response to events, or manually run them as needed.
*   **Monitoring & Observability:** Track workflow activity with a self-hosted Prefect server or the managed Prefect Cloud dashboard.
*   **Event-Driven Architecture:** React to changes in your data and infrastructure with event-based automations.
*   **Flexible Deployment:** Deploy workflows to various environments, scaling your pipelines to meet your needs.

## Getting Started

Prefect requires Python 3.9+.

**Installation:**

```bash
pip install -U prefect
```
or
```bash
uv add prefect
```

**Example:**

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

**Run the workflow:**

1.  Start a Prefect server:
    ```bash
    prefect server start
    ```
2.  Open the UI: http://localhost:4200

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

Prefect Cloud provides enterprise-grade workflow orchestration. Automate over 200 million data tasks monthly and reduce pipeline errors.

*   **Learn more about Prefect Cloud:** [Prefect Cloud vs OSS](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   **Try it for yourself:** [Sign up for Prefect Cloud](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)

## prefect-client

For interacting with Prefect Cloud or a remote Prefect server, use the lightweight `prefect-client`.

*   **Explore prefect-client:** [Prefect Client on PyPI](https://pypi.org/project/prefect-client/)

## Connect & Contribute

Join the thriving Prefect community of 25,000+ practitioners.

### Community Resources

*   🌐 **Documentation:** [Explore the Documentation](https://docs.prefect.io)
*   💬 **Slack:** [Join the Slack Community](https://prefect.io/slack)
*   🤝 **Contribute:** [Contribute to Prefect](https://docs.prefect.io/contribute/)
*   🔌 **Integrations:** [Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)

### Stay Informed

*   📥 **Newsletter:** [Subscribe to our Newsletter](https://prefect.io/newsletter)
*   📣 **Twitter/X:** [Latest updates and announcements](https://x.com/PrefectIO)
*   📺 **YouTube:** [Video tutorials and webinars](https://www.youtube.com/@PrefectIO)
*   📱 **LinkedIn:** [Professional networking and company news](https://www.linkedin.com/company/prefect)

Your contributions are valuable to the Prefect community!