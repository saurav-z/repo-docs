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

**Automate, monitor, and manage your data pipelines with ease using Prefect, the leading Python workflow orchestration framework.**

Prefect empowers data teams to build robust and reliable data pipelines.  Transform your scripts into production-ready workflows with features like scheduling, retries, caching, and event-driven automation, all with just a few lines of code.

Key features include:

*   **Simplified Workflow Creation:** Easily define workflows using Pythonic decorators for `flow` and `task`.
*   **Resilient Pipelines:**  Built-in support for retries, dependencies, and dynamic branching to handle errors gracefully.
*   **Scheduling and Automation:** Schedule workflows to run at specific times or trigger them based on events.
*   **Monitoring and Observability:** Track workflow activity via a self-hosted Prefect server or the managed Prefect Cloud dashboard.
*   **Flexible Deployment:** Deploy workflows to various environments, from local machines to cloud platforms.
*   **Built-in Integrations:** Integrate with a vast array of data tools and services.

## Getting Started

Prefect requires Python 3.9+.

### Installation

Install Prefect using pip:

```bash
pip install -U prefect
```

Or, using `uv`:

```bash
uv add prefect
```

### Example: Fetching GitHub Stars

Here's a simple example demonstrating how to use Prefect:

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

### Running the Example

1.  **Start the Prefect Server:**
    ```bash
    prefect server start
    ```

2.  **Run the Python Script:**  Execute the provided Python code.

3.  **View in the UI:**  Access the Prefect UI at [http://localhost:4200](http://localhost:4200) to monitor your workflow.

### Scheduling with Deployments

To schedule the workflow, modify the last line of the script:

```python
if __name__ == "__main__":
    github_stars.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={"repos": ["PrefectHQ/prefect"]}
    )
```

Now, the workflow will run every minute.  You can also run deployments manually from the UI or CLI, and trigger them via [events](https://docs.prefect.io/latest/automate/?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

**Learn More:** Explore the [Prefect Documentation](https://docs.prefect.io/v3/get-started/index?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) for details on:

*   [Deploying Flows](https://docs.prefect.io/v3/deploy?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   [Error Handling and Retries](https://docs.prefect.io/v3/develop/write-tasks#retries?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   [Integrations](https://docs.prefect.io/integrations/integrations?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)
*   [Team Collaboration](https://docs.prefect.io/v3/manage/cloud/manage-users/manage-teams#manage-teams?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none)

## Prefect Cloud

Prefect Cloud provides enterprise-grade workflow orchestration. It automates over 200 million data tasks monthly for organizations like Progressive Insurance and Cash App, improving productivity and reducing pipeline errors.

Learn more about Prefect Cloud [here](https://www.prefect.io/cloud-vs-oss?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none) or [try it for yourself](https://app.prefect.cloud?utm_source=oss&utm_medium=oss&utm_campaign=oss_gh_repo&utm_term=none&utm_content=none).

## prefect-client

For client-side functionality, especially in ephemeral environments, check out our [prefect-client](https://pypi.org/project/prefect-client/).

## Connect & Contribute

Join the vibrant Prefect community of 25,000+ practitioners!

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

We value your contributions!  Report bugs, suggest features, and help improve Prefect.

[Back to the original repo](https://github.com/PrefectHQ/prefect)
```

Key improvements and SEO considerations:

*   **Clear Headline:**  Uses the target keyword ("Python workflow orchestration") and places it prominently.
*   **Concise Hook:**  A strong opening sentence highlighting the core value proposition.
*   **Bulleted Key Features:** Uses bullet points for readability and to emphasize core benefits.
*   **Keyword Optimization:**  Naturally incorporates relevant keywords like "workflow orchestration," "data pipelines," and "Python."
*   **Action-Oriented Language:** Uses verbs like "Automate," "monitor," and "manage."
*   **Clear Structure:**  Uses headings and subheadings for easy navigation and readability.
*   **Complete and Concise:**  Offers a good overview of the project and what it does in a focused manner.
*   **Links to important resources:** Documentation, cloud and community resources are all clearly linked
*   **Back to original repo:** A link back to the original repository is provided for easy access.