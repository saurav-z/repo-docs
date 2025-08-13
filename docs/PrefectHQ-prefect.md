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

# Prefect: Production-Ready Dataflow Automation for Python

Prefect is a powerful and flexible workflow orchestration framework that transforms your Python scripts into robust, production-ready data pipelines. **[Check out the original repository](https://github.com/PrefectHQ/prefect).**

**Key Features:**

*   **Effortless Workflow Creation:** Easily orchestrate Python scripts with simple `@flow` and `@task` decorators.
*   **Resilient Pipelines:** Build pipelines that automatically retry failed tasks and handle dependencies.
*   **Scheduling & Automation:** Schedule workflows with cron expressions and trigger them based on events.
*   **Monitoring & Observability:** Track workflow activity through a self-hosted Prefect server or Prefect Cloud.
*   **Scalability & Reliability:** Design data pipelines that can react to the world around them and recover from unexpected changes.
*   **Integration-Friendly:** Integrate seamlessly with your existing tools and services.
*   **Open-Source & Cloud Options:** Choose between a self-hosted open-source server or a fully managed cloud platform.

## Getting Started

Prefect requires Python 3.9+. Install Prefect using pip or uv:

```bash
pip install -U prefect
```

or

```bash
uv add prefect
```

Here's a basic example to get you started:

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

To run this example:

1.  **Start the Prefect Server:** `prefect server start`
2.  **Run the Python Script:** Execute the python script.
3.  **View in the UI:** Access the Prefect UI at `http://localhost:4200` to monitor your workflow.

To schedule the workflow:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

## Deploy workflows

You can create a workflow deployment from the CLI that will run on a given schedule using `prefect deployment create`:

```bash
prefect deployment create github_stars.py:github_stars --cron "* * * * *"
```

Then you can run your workflow manually from the UI or CLI. You can even run deployments in response to [events](https://docs.prefect.io/latest/automate/?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

> **Tip:** Explore the [Prefect Documentation](https://docs.prefect.io/v3/get-started/index?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) for more information on:
>
> *   [Deploying flows to production environments](https://docs.prefect.io/v3/deploy?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
> *   [Adding error handling and retries](https://docs.prefect.io/v3/develop/write-tasks#retries?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
> *   [Integrating with your existing tools](https://docs.prefect.io/integrations/integrations?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
> *   [Setting up team collaboration features](https://docs.prefect.io/v3/manage/cloud/manage-users/manage-teams#manage-teams?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)

## Prefect Cloud

Prefect Cloud provides workflow orchestration for the modern data enterprise. By automating over 200 million data tasks monthly, Prefect empowers diverse organizations — from Fortune 50 leaders such as Progressive Insurance to innovative disruptors such as Cash App — to increase engineering productivity, reduce pipeline errors, and cut data workflow compute costs.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or sign up to [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

If your use case is geared towards communicating with Prefect Cloud or a remote Prefect server, check out our
[prefect-client](https://pypi.org/project/prefect-client/). It is a lighter-weight option for accessing client-side functionality in the Prefect SDK and is ideal for use in ephemeral execution environments.

## Connect & Contribute

Join a thriving community of over 25,000 practitioners who solve data challenges with Prefect. Prefect's community is built on collaboration, technical innovation, and continuous improvement.

### Community Resources

*   🌐 **[Explore the Documentation](https://docs.prefect.io)** - Comprehensive guides and API references
*   💬 **[Join the Slack Community](https://prefect.io/slack)** - Connect with thousands of practitioners
*   🤝 **[Contribute to Prefect](https://docs.prefect.io/contribute/)** - Help shape the future of the project
*   🔌 **[Support or create a new Prefect integration](https://docs.prefect.io/contribute/contribute-integrations)** - Extend Prefect's capabilities
*   📋 **[Tail the Dev Log](https://dev-log.prefect.io/)** - Prefect's open source development blog

### Stay Informed

*   📥 **[Subscribe to our Newsletter](https://prefect.io/newsletter)** - Get the latest Prefect news and updates
*   📣 **[Twitter/X](https://x.com/PrefectIO)** - Latest updates and announcements
*   📺 **[YouTube](https://www.youtube.com/@PrefectIO)** - Video tutorials and webinars
*   📱 **[LinkedIn](https://www.linkedin.com/company/prefect)** - Professional networking and company news

Your contributions, questions, and ideas make Prefect better every day. Whether you're reporting bugs, suggesting features, or improving documentation, your input is invaluable to the Prefect community.
```

Key improvements and SEO optimizations:

*   **Clear, Concise Title:**  "Prefect: Production-Ready Dataflow Automation for Python" uses keywords to attract relevant searches.
*   **One-Sentence Hook:**  Immediately grabs attention and highlights the core benefit.
*   **Bulleted Key Features:**  Makes it easy for users to quickly scan and understand the value proposition. Keywords are used to improve SEO.
*   **Organized Headings:**  Uses proper headings (H2, H3) for better readability and SEO ranking.
*   **Keyword Integration:** Natural use of keywords such as "workflow orchestration," "data pipelines," "Python," and "automation."
*   **Clear Installation and Usage Instructions:**  A concise and easy-to-follow getting started section to improve user experience.
*   **Links to relevant documentation:** Improves SEO by linking to other important pages.
*   **Community and Support Section:** Encourages user engagement and highlights ways to get help.
*   **Link to original repo at the top** Improves SEO by linking to the original repo and makes it easier for users to find the source.
*   **Focus on Benefits:** Highlights what users *gain* by using Prefect.