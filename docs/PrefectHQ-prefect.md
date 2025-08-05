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

# Prefect: The Python-Native Workflow Orchestration Framework

Prefect is a powerful, open-source workflow orchestration framework built for Python, enabling data teams to build, run, and monitor robust data pipelines. ([View on GitHub](https://github.com/PrefectHQ/prefect))

## Key Features

*   **Simplified Workflow Creation:** Easily transform Python scripts into production-ready workflows with just a few lines of code using `@flow` and `@task` decorators.
*   **Resilient Pipelines:** Build dynamic pipelines that handle retries, dependencies, and complex logic, ensuring reliability.
*   **Scheduling & Automation:** Automate workflows with scheduling capabilities, event-based triggers, and more.
*   **Real-time Monitoring:** Track workflow activity and gain visibility with the Prefect server or Prefect Cloud.
*   **Scalability & Flexibility:** Design pipelines that scale to meet growing data needs and adapt to changing environments.

## Getting Started

### Installation

Install Prefect using pip or uv:

```bash
pip install -U prefect
```

```bash
uv add prefect
```

### Example Workflow

Here's a simple example to fetch GitHub stars:

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

if __name__ == "__main__":
    github_stars(["PrefectHQ/Prefect"])
```

### Running a Workflow

1.  **Start the Prefect server:**

```bash
prefect server start
```

2.  **Open the UI:**  Access the UI at `http://localhost:4200` to monitor your workflow.
3.  **Deployment & Scheduling:** Deploy and schedule your workflow using deployments:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Prefect Cloud

Prefect Cloud offers a managed workflow orchestration solution.  It provides advanced features for observability, collaboration, and enterprise-grade scalability.

*   **Workflow Orchestration:** Automated management of over 200 million data tasks monthly.
*   **Enhanced Productivity:** Increases engineering efficiency.
*   **Reduced Errors:** Minimizes pipeline failures.
*   **Cost Optimization:** Lowers data workflow compute costs.

Explore Prefect Cloud: [https://www.prefect.io/cloud-vs-oss](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
Try it for yourself: [https://app.prefect.cloud](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)

## prefect-client

For interacting with Prefect Cloud or a remote Prefect server, explore the [prefect-client](https://pypi.org/project/prefect-client/). It's a lightweight option for client-side functionality, perfect for ephemeral environments.

## Connect & Contribute

Join a thriving community of over 25,000 data professionals using Prefect. Contribute and collaborate to shape the future of data orchestration.

### Community Resources

*   🌐 **Documentation:** [https://docs.prefect.io](https://docs.prefect.io)
*   💬 **Slack:** [https://prefect.io/slack](https://prefect.io/slack)
*   🤝 **Contribute:** [https://docs.prefect.io/contribute/](https://docs.prefect.io/contribute/)
*   🔌 **Integrations:** [https://docs.prefect.io/contribute/contribute-integrations](https://docs.prefect.io/contribute/contribute-integrations)

### Stay Informed

*   📥 **Newsletter:** [https://prefect.io/newsletter](https://prefect.io/newsletter)
*   📣 **Twitter/X:** [https://x.com/PrefectIO](https://x.com/PrefectIO)
*   📺 **YouTube:** [https://www.youtube.com/@PrefectIO](https://www.youtube.com/@PrefectIO)
*   📱 **LinkedIn:** [https://www.linkedin.com/company/prefect](https://www.linkedin.com/company/prefect)