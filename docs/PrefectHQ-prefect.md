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

# Prefect: The Dataflow Automation Framework

**Prefect is a powerful Python-based workflow orchestration framework that simplifies the creation, deployment, and monitoring of data pipelines, helping you build robust, reliable, and observable workflows.**

*   **Simplified Workflow Creation:** Build data pipelines quickly and easily with Python decorators.
*   **Resilient Pipelines:**  Automatically handles retries, dependencies, and complex branching logic for robust execution.
*   **Scheduling & Automation:** Schedule workflows and trigger events-based automations with ease.
*   **Monitoring & Observability:** Track workflow activity with a self-hosted Prefect server or via the managed Prefect Cloud dashboard.
*   **Extensible and Integratable:**  Integrates seamlessly with your existing tools and supports numerous integrations.

## Getting Started

To get started with Prefect, you'll need Python 3.9+ installed.

1.  **Install Prefect:**

    ```bash
    pip install -U prefect
    ```
    or
    ```bash
    uv add prefect
    ```

2.  **Create and Run a Simple Workflow:**  Here's a basic example that fetches the number of GitHub stars for a repository:

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

    # Run the flow
    if __name__ == "__main__":
        github_stars(["PrefectHQ/Prefect"])
    ```

3.  **Start the Prefect Server:**

    ```bash
    prefect server start
    ```
    Open the UI at `http://localhost:4200` to monitor your workflow.

4.  **Deploy and Schedule:**  To run your workflow on a schedule, turn it into a deployment:

    ```python
    if __name__ == "__main__":
        github_stars.serve(
            name="first-deployment",
            cron="* * * * *",
            parameters={"repos": ["PrefectHQ/prefect"]}
        )
    ```

    This will schedule the workflow to run every minute. You can manage and run deployments from the UI or CLI.

>   **Next Steps:** Explore the [Prefect documentation](https://docs.prefect.io/v3/get-started/index) to learn more about deploying flows, error handling, integrations, and team collaboration features.

## Prefect Cloud

Prefect Cloud is a managed workflow orchestration platform designed for modern data teams, automating over 200 million data tasks monthly.  It helps organizations increase engineering productivity, reduce pipeline errors, and optimize compute costs.  Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss) or sign up to [try it for yourself](https://app.prefect.cloud).

## prefect-client

For interacting with Prefect Cloud or a remote Prefect server, explore the [prefect-client](https://pypi.org/project/prefect-client/), a lightweight option optimized for ephemeral environments.

## Connect and Contribute

Join the Prefect community, a thriving group of over 25,000 practitioners!

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

Your contributions, questions, and ideas are essential to the Prefect community.

---

[Back to the top](https://github.com/PrefectHQ/prefect)