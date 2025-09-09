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

**Prefect is a powerful, open-source workflow orchestration tool that helps data teams build, run, and monitor reliable data pipelines with ease.**

[<img src="https://img.shields.io/github/stars/PrefectHQ/prefect?style=social" alt="GitHub stars">](https://github.com/PrefectHQ/prefect)

Prefect allows you to transform your Python scripts into production-ready workflows in minutes.  This results in:

*   **Simplified Workflow Creation:**  Easily define workflows using Python decorators for tasks and flows.
*   **Robust Error Handling:** Built-in features for retries, dependencies, and intelligent error handling.
*   **Dynamic Workflows:** Build pipelines that react to changes in your data or environment.
*   **Scheduling & Automation:** Schedule workflows with cron expressions, or trigger them based on events.
*   **Monitoring & Observability:** Track workflow activity through the Prefect Server or Prefect Cloud dashboards.
*   **Integration:** Integrates with your existing tools and services for seamless data pipeline management.

Prefect streamlines the process of automating data workflows with features like scheduling, caching, retries, and event-based automations, improving data pipeline reliability and efficiency.

**Key Features:**

*   **Python-Native:** Build workflows using familiar Python syntax.
*   **Flexible Deployment:** Run workflows locally, in your infrastructure, or in Prefect Cloud.
*   **Reliable Execution:** Built-in retries, caching, and error handling.
*   **Observability:** Centralized monitoring and logging with a user-friendly UI.
*   **Scalable:** Designed to handle complex, data-intensive workflows.
*   **Event-Driven:** Trigger workflows based on events and external triggers.

## Getting Started

Prefect requires Python 3.9+.  Get started by running one of these commands:

```bash
pip install -U prefect
```

```bash
uv add prefect
```

The following code shows how to create and run a Python file using Prefect `flow` and `task` decorators.  This example retrieves the number of GitHub stars from a repository:

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

To observe what happens, start the Prefect server and open the UI at http://localhost:4200:

```bash
prefect server start
```

Schedule your workflow as a deployment and set it to run every minute using the following code:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

You can run your workflow manually from the UI or CLI and have it run in response to [events](https://docs.prefect.io/latest/automate/?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

**Learn More:**

*   [Deploying flows to production environments](https://docs.prefect.io/v3/deploy?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   [Adding error handling and retries](https://docs.prefect.io/v3/develop/write-tasks#retries?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   [Integrating with your existing tools](https://docs.prefect.io/integrations/integrations?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   [Setting up team collaboration features](https://docs.prefect.io/v3/manage/cloud/manage-users/manage-teams#manage-teams?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)

## Prefect Cloud

[Prefect Cloud](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) provides workflow orchestration for the modern data enterprise. It automates over 200 million data tasks monthly, empowering organizations to increase engineering productivity, reduce pipeline errors, and cut data workflow compute costs.

Sign up to [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For communication with Prefect Cloud or a remote Prefect server, explore the lightweight [prefect-client](https://pypi.org/project/prefect-client/).  This is ideal for ephemeral execution environments.

## Connect & Contribute

Join a vibrant community of over 25,000 practitioners who solve data challenges with Prefect. Prefect's community thrives on collaboration, technical innovation, and continuous improvement.

### Community Resources

🌐 **[Explore the Documentation](https://docs.prefect.io)** - Comprehensive guides and API references  
💬 **[Join the Slack Community](https://prefect.io/slack)** - Connect with thousands of practitioners  
🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)** - Help shape the future of the project  
 🔌 **[Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)** - Extend Prefect's capabilities   
📋 **[Tail the Dev Log](https://dev-log.prefect.io/)** - Prefect's open source development blog

### Stay Informed

📥 **[Subscribe to our Newsletter](https://prefect.io/newsletter)** - Get the latest Prefect news and updates  
📣 **[Twitter/X](https://x.com/PrefectIO)** - Latest updates and announcements  
📺 **[YouTube](https://www.youtube.com/@PrefectIO)** - Video tutorials and webinars  
📱 **[LinkedIn](https://www.linkedin.com/company/prefect)** - Professional networking and company news

Your contributions, questions, and ideas make Prefect better every day!

[Back to Top](#prefect)  |  [Original Repository](https://github.com/PrefectHQ/prefect)
```

Key improvements and SEO considerations:

*   **Clear, Concise Title:** "Prefect: The Python Workflow Orchestration Framework" is optimized for search.
*   **One-Sentence Hook:**  Immediately grabs attention and states the core benefit.
*   **Strategic Keyword Use:**  Uses terms like "workflow orchestration," "data pipelines," "Python," and "automation" naturally throughout.
*   **Feature Bullets:**  Highlights key benefits for quick scanning.
*   **Structured Headings:** Organizes content for readability and SEO.
*   **Call to Action:** Includes a call to action to get started.
*   **Links:** Properly uses internal and external links for SEO and user navigation.
*   **Concise Language:**  Removes unnecessary words.
*   **Social Proof:** Includes GitHub star badge.
*   **Community Emphasis:**  Promotes the community and encourages contribution.
*   **Back to Top and Original Repo links:** Added for easy navigation.