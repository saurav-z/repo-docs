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

# Prefect: Build and Orchestrate Data Workflows with Ease

Prefect is a powerful, open-source workflow orchestration platform that simplifies building, running, and monitoring data pipelines in Python.  Visit the [Prefect GitHub Repository](https://github.com/PrefectHQ/prefect) for more details.

Key Features:

*   **Simplified Workflow Definition:**  Use Python code to define data pipelines with the `@flow` and `@task` decorators for easy orchestration.
*   **Resilient Data Pipelines:**  Build workflows that automatically handle retries, dependencies, and even complex branching logic to ensure reliable execution.
*   **Scheduling & Automation:** Schedule workflows with cron expressions and trigger them based on events.
*   **Monitoring & Observability:** Track workflow activity with a self-hosted Prefect server or with Prefect Cloud for a centralized view of all workflows.
*   **Scalability:** Easily scale your data pipelines to meet growing data demands.
*   **Integration:** Works seamlessly with existing tools, databases, and APIs.

## Getting Started

To get started with Prefect, follow these steps:

1.  **Installation:** Ensure you have Python 3.9+ and install Prefect using pip or uv:

    ```bash
    pip install -U prefect
    ```

    or

    ```bash
    uv add prefect
    ```

2.  **Create a Workflow:**  Write a Python script using Prefect's `@flow` and `@task` decorators to define your data pipeline.  Here's a simple example that fetches the number of GitHub stars from a repository:

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

3.  **Run and Monitor:**  Start a Prefect server to monitor your workflow.  Then, run your script and see the results in the UI:

    ```bash
    prefect server start
    ```

    You can then deploy your workflow to run on a schedule:

    ```python
    if __name__ == "__main__":
        github_stars.serve(
            name="first-deployment",
            cron="* * * * *",
            parameters={"repos": ["PrefectHQ/prefect"]}
        )
    ```

    You can also run your workflow manually from the UI or CLI, and you can even run deployments in response to events.

## Prefect Cloud

Prefect Cloud provides workflow orchestration for modern data enterprises. Automating over 200 million data tasks monthly, Prefect empowers organizations to increase engineering productivity, reduce pipeline errors, and cut data workflow compute costs.

Read more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or sign up to [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

If your use case is geared towards communicating with Prefect Cloud or a remote Prefect server, check out our
[prefect-client](https://pypi.org/project/prefect-client/). It is a lighter-weight option for accessing client-side functionality in the Prefect SDK and is ideal for use in ephemeral execution environments.

## Connect & Contribute

Join the thriving Prefect community!  Whether you're a seasoned data engineer or just starting, we welcome your contributions.

### Community Resources

*   🌐 **[Explore the Documentation](https://docs.prefect.io)** - Comprehensive guides and API references
*   💬 **[Join the Slack Community](https://prefect.io/slack)** - Connect with thousands of practitioners
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)** - Help shape the future of the project
*   🔌 **[Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)** - Extend Prefect's capabilities

### Stay Informed

*   📥 **[Subscribe to our Newsletter](https://prefect.io/newsletter)** - Get the latest Prefect news and updates
*   📣 **[Twitter/X](https://x.com/PrefectIO)** - Latest updates and announcements
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)** - Video tutorials and webinars
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)** - Professional networking and company news