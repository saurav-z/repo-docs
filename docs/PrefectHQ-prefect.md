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

Prefect is a powerful, open-source workflow orchestration platform built for data pipelines in Python, enabling you to build robust, dynamic data workflows with ease. [Learn more about Prefect](https://github.com/PrefectHQ/prefect).

**Key Features:**

*   **Simplified Workflow Creation:** Easily convert Python scripts into production-ready workflows with just a few lines of code.
*   **Resilient Pipelines:** Build dynamic data pipelines that react to changing conditions and automatically recover from errors.
*   **Scheduling & Automation:** Schedule workflows, set up retries, and trigger workflows based on events.
*   **Observability:** Track workflow activity and monitor performance using a self-hosted Prefect server or Prefect Cloud.
*   **Built-in Features:** Benefit from built-in features like caching, retries, and complex branching logic.

## Getting Started

Prefect requires Python 3.9+ and is easily installed using pip or `uv`:

```bash
pip install -U prefect
```

```bash
uv add prefect
```

To see Prefect in action, create a Python file using the `flow` and `task` decorators to orchestrate a workflow.  Here's an example to fetch GitHub stars:

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

Start a Prefect server locally and view the UI:

```bash
prefect server start
```

Access the UI at http://localhost:4200.  To schedule your workflow, turn it into a deployment:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

Your workflow will now run on a schedule!  You can also run workflows manually from the UI or CLI, and trigger them based on events.

**Next Steps:**  Explore the [Prefect documentation](https://docs.prefect.io/v3/get-started/index?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) for:

*   Deploying workflows to production environments
*   Implementing error handling and retries
*   Integrating with existing tools
*   Setting up team collaboration

## Prefect Cloud

Prefect Cloud offers a managed workflow orchestration solution for modern data teams, automating over 200 million data tasks monthly. It enables organizations to enhance engineering productivity, reduce pipeline errors, and optimize data workflow costs.

Read more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or [sign up for a free trial](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For interacting with Prefect Cloud or a remote Prefect server, use the [prefect-client](https://pypi.org/project/prefect-client/), a lightweight client-side SDK.  This is ideal for ephemeral execution environments.

## Connect & Contribute

Join a thriving community of over 25,000 data professionals! Prefect's community thrives on collaboration and innovation.

### Community Resources

*   🌐 **[Explore the Documentation](https://docs.prefect.io)**: Comprehensive guides and API references
*   💬 **[Join the Slack Community](https://prefect.io/slack)**: Connect with thousands of practitioners
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)**: Help shape the future of the project
*   🔌 **[Integrations](https://docs.prefect.io/contribute/contribute-integrations)**: Support or create new Prefect integrations

### Stay Informed

*   📥 **[Subscribe to our Newsletter](https://prefect.io/newsletter)**: Get the latest Prefect news and updates
*   📣 **[Twitter/X](https://x.com/PrefectIO)**: Latest updates and announcements
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)**: Video tutorials and webinars
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)**: Professional networking and company news

Your contributions are invaluable.  Whether you're reporting issues, suggesting features, or improving the documentation, your input helps make Prefect better!
```
Key improvements and SEO considerations:

*   **Clear, Concise Title:**  Uses "Prefect" and incorporates key phrase "workflow orchestration" for searchability.
*   **SEO-Optimized Hook:** Starts with a strong, benefit-driven one-sentence summary including primary keywords.
*   **Headings:**  Uses clear headings to structure the content (e.g., "Key Features," "Getting Started," "Prefect Cloud") making it easy to scan.
*   **Bulleted Key Features:**  Uses bullet points to highlight core benefits, making the information easy to digest.
*   **Keywords Throughout:**  Naturally incorporates relevant keywords like "workflow orchestration," "data pipelines," "Python," "automation," "scheduling," "monitoring," and "Prefect Cloud."
*   **Links to Key Resources:** Provides clear, direct links to the documentation, quickstart guide, and Prefect Cloud.
*   **Call to Action:** Encourages users to explore the documentation, join the community, and try Prefect Cloud.
*   **Clean Formatting:** Improved readability with Markdown formatting, including bold text, code blocks, and lists.
*   **GitHub Link:** Provides a direct link back to the original repository.