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

# Prefect: Build, Observe, and Manage Data Workflows in Python

Prefect is a powerful and user-friendly workflow orchestration framework that simplifies the creation, monitoring, and management of data pipelines in Python.  **[Learn more about Prefect on GitHub](https://github.com/PrefectHQ/prefect)**.

## Key Features

*   **Simplified Workflow Definition:** Easily transform Python scripts into robust workflows using intuitive `@flow` and `@task` decorators.
*   **Resilient Execution:** Benefit from built-in features like retries, dependencies, and caching to handle errors and ensure reliable pipeline execution.
*   **Scheduling and Automation:** Schedule workflows and automate their execution with flexible scheduling options.
*   **Observability and Monitoring:** Track workflow activity and monitor performance with a self-hosted Prefect server or the managed Prefect Cloud dashboard.
*   **Event-Driven Workflows:** Trigger workflows based on events, enabling real-time data pipeline automation.
*   **Integration:** Seamlessly integrate with a wide range of tools and services to build end-to-end data solutions.

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

Here's a simple example of a Prefect flow that fetches GitHub stars:

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

1.  Start the Prefect server: `prefect server start`
2.  Open the UI in your browser:  `http://localhost:4200`
3.  Run the workflow manually or turn it into a deployment:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Prefect Cloud

Prefect Cloud offers a fully managed workflow orchestration platform for modern data teams. It allows you to automate over 200 million data tasks monthly, increasing engineering productivity, reducing errors, and cutting workflow costs.  Used by companies such as Progressive Insurance and Cash App.

*   **Learn More:** [Prefect Cloud vs. OSS](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   **Try it Free:** [Prefect Cloud](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)

## prefect-client

The `prefect-client` is a lighter-weight client for interacting with Prefect Cloud or a remote Prefect server, ideal for ephemeral environments.

*   **Learn More:** [prefect-client on PyPI](https://pypi.org/project/prefect-client/)

## Connect & Contribute

Join the Prefect community of over 25,000 practitioners to solve data challenges!

### Community Resources

*   🌐 **[Documentation](https://docs.prefect.io)**: Comprehensive guides and API references
*   💬 **[Slack Community](https://prefect.io/slack)**: Connect with practitioners.
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)**: Help shape the project's future.
*   🔌 **[Create a Prefect Integration](https://docs.prefect.io/contribute/contribute-integrations)**: Extend Prefect's capabilities.

### Stay Informed

*   📥 **[Newsletter](https://prefect.io/newsletter)**: Get the latest Prefect news and updates.
*   📣 **[Twitter/X](https://x.com/PrefectIO)**: Latest updates and announcements
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)**: Video tutorials and webinars
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)**: Professional networking and company news

Your contributions, questions, and ideas are welcome!
```
Key improvements and SEO considerations:

*   **Clear Title and Hook:** The main heading is more descriptive, and the first sentence acts as a compelling hook, highlighting the key benefit.
*   **Keyword Optimization:** Includes relevant keywords like "workflow orchestration," "data pipelines," and "Python."
*   **Bulleted Features:** Uses bullet points to make key features easily scannable.
*   **Clear Structure:** Uses headings and subheadings to organize information and improve readability.
*   **Calls to Action:** Includes clear calls to action (e.g., "Learn more," "Try it Free") to encourage engagement.
*   **Community Emphasis:** Highlights the community and provides links to relevant resources.
*   **Concise Summary:**  Provides a more concise summary of the project.
*   **GitHub Link:** Added a direct link back to the original repo at the start.