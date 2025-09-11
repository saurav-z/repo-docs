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

**Prefect simplifies building, running, and monitoring data pipelines, empowering you to transform raw scripts into robust, production-ready workflows.**

*   **Effortless Workflow Creation:** Easily convert your Python scripts into automated workflows with just a few lines of code using the `flow` and `task` decorators.
*   **Built-in Resilience:** Benefit from built-in features like scheduling, caching, retries, and event-driven automation to handle unexpected issues and ensure your data pipelines run reliably.
*   **Flexible Deployment Options:** Choose between a self-hosted Prefect server or the fully managed [Prefect Cloud](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) for workflow orchestration.
*   **Real-time Monitoring & Observability:** Track workflow activity and performance with a user-friendly dashboard, gaining valuable insights into your data pipeline operations.
*   **Scalable & Dynamic Pipelines:** Build dynamic workflows that react to changing conditions, easily adapting to evolving business needs.

> [!TIP]
> Prefect flows can handle retries, dependencies, and even complex branching logic
> 
> [Check our docs](https://docs.prefect.io/v3/get-started/index?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or see the example below to learn more!

## Getting Started with Prefect

Prefect requires Python 3.9+. To install the latest version of Prefect, run:

```bash
pip install -U prefect
```

or with `uv`:

```bash
uv add prefect
```

### Example: Fetching GitHub Stars
Here's a simple example to get you started, fetching the number of GitHub stars from a repository:

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
Start a Prefect server and view the UI at http://localhost:4200 to monitor your workflow.

To run your workflow on a schedule, create a deployment:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

> [!TIP]
> Where to go next - check out our [documentation](https://docs.prefect.io/v3/get-started/index?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) to learn more about:
> - [Deploying flows to production environments](https://docs.prefect.io/v3/deploy?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
> - [Adding error handling and retries](https://docs.prefect.io/v3/develop/write-tasks#retries?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
> - [Integrating with your existing tools](https://docs.prefect.io/integrations/integrations?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
> - [Setting up team collaboration features](https://docs.prefect.io/v3/manage/cloud/manage-users/manage-teams#manage-teams?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)

## Prefect Cloud: Managed Workflow Orchestration

Prefect Cloud offers a fully managed solution for data pipeline orchestration, providing enterprise-grade features and scalability. Automate over 200 million data tasks monthly with Prefect Cloud, increasing engineering productivity, reducing pipeline errors, and lowering compute costs.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## Prefect Client

For interacting with Prefect Cloud or a remote Prefect server from client-side environments, explore the [prefect-client](https://pypi.org/project/prefect-client/). It's a lightweight alternative for accessing functionality in the Prefect SDK and is ideal for ephemeral environments.

## Connect & Contribute

Join the vibrant Prefect community of over 25,000 data practitioners!

### Community Resources

*   🌐 **[Explore the Documentation](https://docs.prefect.io)**: Comprehensive guides and API references
*   💬 **[Join the Slack Community](https://prefect.io/slack)**: Connect with thousands of practitioners
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)**: Help shape the future of the project
*   🔌 **[Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)**: Extend Prefect's capabilities
*   📋 **[Tail the Dev Log](https://dev-log.prefect.io/)**: Prefect's open source development blog

Your contributions, questions, and ideas make Prefect better every day!

[Back to Top](#prefect) - [Prefect on GitHub](https://github.com/PrefectHQ/prefect)