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

# Prefect: Production-Ready Workflow Orchestration for Python

Prefect is a powerful and flexible workflow orchestration framework, allowing you to build, monitor, and manage robust data pipelines in Python.  Elevate your scripts to production workflows with ease!

## Key Features

*   **Simple Automation:** Transform Python scripts into automated, production-ready workflows with minimal code.
*   **Resilient Pipelines:** Build dynamic pipelines that automatically handle retries, dependencies, and unexpected changes.
*   **Scheduling and Monitoring:** Schedule your workflows and monitor their activity through a self-hosted server or Prefect Cloud.
*   **Built-in Features:** Leverage features like scheduling, caching, retries, and event-based automations out-of-the-box.
*   **Dynamic Workflows:** Handle complex branching logic and react to the world around your data.
*   **Prefect Cloud Integration:** Easily integrate with the managed Prefect Cloud for advanced features and scalability.
*   **Active Community:** Join a vibrant community of data engineers and enthusiasts.

## Getting Started

Prefect requires Python 3.9+ to get started.

### Installation

Install Prefect using `pip`:

```bash
pip install -U prefect
```

Or using `uv`:

```bash
uv add prefect
```

### Example: Building a Simple Workflow

Here's a quick example of how to use Prefect to fetch the number of GitHub stars for a repository:

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

1.  **Start the Server:** Run `prefect server start` to launch the Prefect server.
2.  **View in UI:** Open the UI at `http://localhost:4200` to monitor your workflow.

### Deploying and Scheduling Workflows

To schedule your workflow:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

This creates a deployment that runs every minute. You can also run deployments manually from the UI or CLI, or trigger them based on [events](https://docs.prefect.io/latest/automate/?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

### Next Steps

*   [Deploying flows to production environments](https://docs.prefect.io/v3/deploy?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   [Adding error handling and retries](https://docs.prefect.io/v3/develop/write-tasks#retries?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   [Integrating with your existing tools](https://docs.prefect.io/integrations/integrations?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   [Setting up team collaboration features](https://docs.prefect.io/v3/manage/cloud/manage-users/manage-teams#manage-teams?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)

## Prefect Cloud

Prefect Cloud provides a fully managed workflow orchestration service. Automate over 200 million tasks monthly, increase engineering productivity, reduce pipeline errors, and optimize compute costs.

*   Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).
*   Sign up to [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For interacting with Prefect Cloud or a remote server, use the lighter-weight [prefect-client](https://pypi.org/project/prefect-client/). It is optimized for client-side functionality and ephemeral environments.

## Connect & Contribute

Join the Prefect community!

### Community Resources

*   🌐 **[Explore the Documentation](https://docs.prefect.io)**
*   💬 **[Join the Slack Community](https://prefect.io/slack)**
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)**
*   🔌 **[Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)**

### Stay Informed

*   📥 **[Subscribe to our Newsletter](https://prefect.io/newsletter)**
*   📣 **[Twitter/X](https://x.com/PrefectIO)**
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)**
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)**

Your contributions are essential to the Prefect community.  [Get started with Prefect](https://github.com/PrefectHQ/prefect) today!
```
Key improvements:

*   **SEO Optimization:** Added relevant keywords like "workflow orchestration," "data pipelines," and "Python" in the title and introduction.
*   **Clear Headings:** Used clear, concise headings for better readability and organization.
*   **Bulleted Key Features:** Highlighted key features with bullet points for easy scanning.
*   **Concise Summary:** Rewrote the introduction to be more impactful and engaging.
*   **Actionable Language:**  Used calls to action like "Get started" and "Join the community."
*   **Updated Links:** Kept all original links and added a backlink to the original repo.
*   **Removed Redundancy:** Streamlined some of the text for better flow and removed duplicate information.
*   **Community Focused:** Emphasized the community aspects more.
*   **Improved Formatting:** Used Markdown for better readability.
*   **Concise and Complete:** Retained all relevant information from the original README while making it easier to understand.