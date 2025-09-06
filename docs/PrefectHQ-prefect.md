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

# Prefect: Orchestrate Your Python Workflows with Ease

Prefect is a powerful and flexible workflow orchestration framework that empowers you to build, monitor, and manage data pipelines in Python. **Transform your Python scripts into robust, production-ready workflows with Prefect's intuitive features.**

**Key Features:**

*   **Simplified Workflow Creation:** Easily define workflows using Python decorators (`@flow`, `@task`).
*   **Scheduling & Automation:** Schedule your workflows to run automatically and trigger them based on events.
*   **Resilient Execution:** Built-in retry mechanisms, error handling, and dependency management for robust workflows.
*   **Observability & Monitoring:** Track workflow runs, visualize execution, and receive alerts through a UI.
*   **Scalability:** Supports scaling workflows to handle complex data processes.
*   **Deployment Flexibility:** Deploy workflows to various environments, including your own infrastructure or Prefect Cloud.
*   **Integration:** Integrates with existing tools and services.

**[View the original repository on GitHub](https://github.com/PrefectHQ/prefect)**

## Getting Started

Prefect requires Python 3.9+.  Install the latest version:

```bash
pip install -U prefect
```

Or, use `uv`:

```bash
uv add prefect
```

Here's a basic example to get you started. This script fetches the number of GitHub stars for a repository:

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

Start a Prefect server to monitor your workflow (UI available at http://localhost:4200):

```bash
prefect server start
```

To schedule your workflow:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Prefect Cloud

Prefect Cloud offers a managed workflow orchestration solution for modern data teams. Automate tasks, reduce pipeline errors, and optimize costs.

Learn more: [Prefect Cloud vs. OSS](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) | [Try Prefect Cloud](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)

## prefect-client

The `prefect-client` package provides client-side functionality for interacting with Prefect Cloud or a remote Prefect server, ideal for ephemeral environments.  Check it out on [PyPI](https://pypi.org/project/prefect-client/).

## Connect & Contribute

Join the Prefect community of over 25,000 practitioners!

### Community Resources

*   🌐 **[Documentation](https://docs.prefect.io)**
*   💬 **[Slack Community](https://prefect.io/slack)**
*   🤝 **[Contribute](https://docs.prefect.io/contribute/)**
*   🔌 **[Create Integrations](https://docs.prefect.io/contribute/contribute-integrations)**
*   📋 **[Dev Log](https://dev-log.prefect.io/)**

### Stay Informed

*   📥 **[Newsletter](https://prefect.io/newsletter)**
*   📣 **[Twitter/X](https://x.com/PrefectIO)**
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)**
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)**
```
Key improvements and why:

*   **SEO-Friendly Title & Description:** Includes key terms like "workflow orchestration," "data pipelines," and "Python" to improve searchability.
*   **Concise Hook:** Starts with a compelling one-sentence summary.
*   **Clear Headings:** Organizes the README for better readability.
*   **Bulleted Key Features:** Highlights the main advantages of Prefect.
*   **Clear Call to Action:** Guides users on how to get started.
*   **Focus on Benefits:** Highlights the benefits users will get from using Prefect
*   **Simplified "Getting Started" section:**  Concise instructions for installation and a runnable example, and links to the documentation.
*   **Prefect Cloud Section:**  Provides a concise overview of the Cloud offering.
*   **Community Section:** Easy links for users to find useful community resources.
*   **Markdown Formatting:** Uses Markdown for better visual presentation.
*   **Concise and Focused:** Removes unnecessary jargon and focuses on the core value proposition.
*   **Includes Link Back to Original Repo** to preserve the original resource.