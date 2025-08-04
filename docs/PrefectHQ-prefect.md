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

Prefect is a powerful and flexible Python workflow orchestration framework designed to simplify data pipeline creation, management, and monitoring.  [Get started with Prefect on GitHub](https://github.com/PrefectHQ/prefect).

**Key Features:**

*   **Simplified Workflow Definition:** Easily turn Python scripts into production-ready workflows using intuitive `@flow` and `@task` decorators.
*   **Resilient and Dynamic Pipelines:** Build pipelines that automatically handle retries, dependencies, and complex branching logic, adapting to changing conditions.
*   **Scheduling and Automation:** Schedule workflows to run at specific times or trigger them based on events.
*   **Observability and Monitoring:** Track workflow activity and monitor performance with a self-hosted Prefect server or the managed Prefect Cloud.
*   **Caching and Error Handling:** Built-in features for caching results and handling errors, improving pipeline reliability.
*   **Cloud Integration:** Seamlessly integrates with Prefect Cloud for advanced features and management.

## Getting Started

Prefect requires Python 3.9+. To install the latest version of Prefect, run one of the following commands:

```bash
pip install -U prefect
```

```bash
uv add prefect
```

Then create and run a Python file that uses Prefect `flow` and `task` decorators to orchestrate and observe your workflow - in this case, a simple script that fetches the number of GitHub stars from a repository:

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

Fire up a Prefect server and open the UI at http://localhost:4200 to see what happened:

```bash
prefect server start
```

To run your workflow on a schedule, turn it into a deployment and schedule it to run every minute by changing the last line of your script to the following:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

You now have a process running locally that is looking for scheduled deployments!
Additionally you can run your workflow manually from the UI or CLI. You can even run deployments in response to [events](https://docs.prefect.io/latest/automate/?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

> [!TIP]
> Where to go next - check out our [documentation](https://docs.prefect.io/v3/get-started/index?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) to learn more about:
> - [Deploying flows to production environments](https://docs.prefect.io/v3/deploy?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
> - [Adding error handling and retries](https://docs.prefect.io/v3/develop/write-tasks#retries?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
> - [Integrating with your existing tools](https://docs.prefect.io/integrations/integrations?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
> - [Setting up team collaboration features](https://docs.prefect.io/v3/manage/cloud/manage-users/manage-teams#manage-teams?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)


## Prefect Cloud

Prefect Cloud provides robust workflow orchestration for modern data teams. By automating millions of data tasks monthly, Prefect Cloud helps organizations increase engineering productivity, reduce pipeline errors, and minimize workflow compute costs.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) and sign up to [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For communication with Prefect Cloud or a remote Prefect server, explore our [prefect-client](https://pypi.org/project/prefect-client/), a lightweight option tailored for client-side functionality within the Prefect SDK, ideal for ephemeral execution environments.

## Connect & Contribute

Join the vibrant Prefect community, composed of over 25,000 data professionals, who are using Prefect to solve complex data challenges. Prefect thrives on collaboration, technical innovation, and continuous improvement.

### Community Resources

🌐 **[Explore the Documentation](https://docs.prefect.io)** - Comprehensive guides and API references  
💬 **[Join the Slack Community](https://prefect.io/slack)** - Connect with thousands of practitioners  
🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)** - Help shape the future of the project  
 🔌 **[Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)** - Extend Prefect's capabilities

### Stay Informed

📥 **[Subscribe to our Newsletter](https://prefect.io/newsletter)** - Get the latest Prefect news and updates  
📣 **[Twitter/X](https://x.com/PrefectIO)** - Latest updates and announcements  
📺 **[YouTube](https://www.youtube.com/@PrefectIO)** - Video tutorials and webinars  
📱 **[LinkedIn](https://www.linkedin.com/company/prefect)** - Professional networking and company news