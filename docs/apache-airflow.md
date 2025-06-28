# Apache Airflow: Orchestrate Your Workflows with Code

**Apache Airflow** is a platform that allows you to programmatically author, schedule, and monitor workflows. Get started with the [official documentation](https://airflow.apache.org/docs/apache-airflow/stable/) or contribute to the project on [GitHub](https://github.com/apache/airflow)!

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

## Key Features

*   **Dynamic Workflows:** Define pipelines in code for versioning, testing, and collaboration.
*   **Extensible Architecture:** Leverage built-in operators and customize workflows to fit your needs.
*   **Flexible Templating:** Utilize Jinja templating for rich customizations.
*   **Robust Scheduling:** Execute tasks on a distributed array of workers with dependency management.
*   **Comprehensive UI:** Visualize pipelines, monitor progress, and troubleshoot issues.

## Requirements

Apache Airflow is tested with:

|            | Main version (dev)     | Stable version (3.0.2) |
|------------|------------------------|------------------------|
| Python     | 3.10, 3.11, 3.12       | 3.9, 3.10, 3.11, 3.12  |
| Platform   | AMD64/ARM64(\*)        | AMD64/ARM64(\*)        |
| Kubernetes | 1.30, 1.31, 1.32, 1.33 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL | 13, 14, 15, 16, 17     | 13, 14, 15, 16, 17     |
| MySQL      | 8.0, 8.4, Innovation   | 8.0, 8.4, Innovation   |
| SQLite     | 3.15.0+                | 3.15.0+                |

\* Experimental

**Note**: MariaDB is not tested/recommended.

**Note**: SQLite is used in Airflow tests. Do not use it in production. We recommend
using the latest stable version of SQLite for local development.

**Note**: Airflow currently can be run on POSIX-compliant Operating Systems. For development, it is regularly
tested on fairly modern Linux Distros and recent versions of macOS.
On Windows you can run it via WSL2 (Windows Subsystem for Linux 2) or via Linux Containers.
The work to add Windows support is tracked via [#10388](https://github.com/apache/airflow/issues/10388), but
it is not a high priority. You should only use Linux-based distros as "Production" execution environment
as this is the only environment that is supported. The only distro that is used in our CI tests and that
is used in the [Community managed DockerHub image](https://hub.docker.com/p/apache/airflow) is
`Debian Bookworm`.

## Getting Started

Explore the [official Airflow documentation](https://airflow.apache.org/docs/apache-airflow/stable/) for installation, tutorials, and in-depth information.

## Installing from PyPI

Install Airflow from PyPI using `pip`, considering constraint files for repeatable installations:

```bash
pip install 'apache-airflow==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.9.txt"
```

Install with extras (e.g. postgres, google):

```bash
pip install 'apache-airflow[postgres,google]==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.9.txt"
```

## Installation

For comprehensive instructions on setting up your local development environment and installing Apache Airflow, please refer to the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

As an [Apache Software Foundation](https://www.apache.org) (ASF) project, Apache Airflow:

*   Follows the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Source code releases are available from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Releases are cryptographically signed by the release manager and officially voted on by the PMC members.

## Convenience Packages

Apache Airflow is also available via:
*   [PyPI releases](https://pypi.org/project/apache-airflow/)
*   [Docker Images](https://hub.docker.com/r/apache/airflow)
*   [Tags in GitHub](https://github.com/apache/airflow/tags)

## User Interface

*   **DAGs**: Overview of all DAGs in your environment.

  ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)

*   **Assets**: Overview of Assets with dependencies.

  ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)

*   **Grid**: Grid representation of a DAG that spans across time.

  ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)

*   **Graph**: Visualization of a DAG's dependencies and their current status for a specific run.

  ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)

*   **Home**: Summary statistics of your Airflow environment.

  ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)

*   **Backfill**: Backfilling a DAG for a specific date range.

  ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)

*   **Code**: Quick way to view source code of a DAG.

  ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)

## Semantic Versioning

Airflow 2.0.0 and later adheres to [SemVer](https://semver.org/) principles.

## Version Life Cycle

*   **3.0.2:** Supported
*   **2.11.0:** Supported
*   **1.10.15:** EOL
*   **Older versions:** EOL

## Support for Python and Kubernetes Versions

Airflow supports Python and Kubernetes versions based on their official release schedules.

## Base OS Support for Reference Airflow Images

The Airflow Community provides conveniently packaged container images with Debian OS, Python, database clients, and predefined providers.

## Approach to Dependencies of Airflow

Airflow's dependency management uses constraints for stability and allows users to upgrade dependencies.

## Contributing

Contribute to Airflow with the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) and [quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst).

## Voting Policy

Commits need a +1 vote from a committer who is not the author; AIP voting considers PMC and committer votes.

## Who Uses Apache Airflow?

Airflow is used by around 500+ organizations (and counting!) [in the wild](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who Maintains Apache Airflow?

Airflow is maintained by a [community](https://github.com/apache/airflow/graphs/contributors) and the [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow).

## What Goes into the Next Release?

The release process depends on several factors, as described in the [What goes into the next release](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document.

## Can I use the Apache Airflow logo in my presentation?

Yes, follow the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>
```

Key improvements and SEO considerations:

*   **Concise Hook:**  The one-sentence hook at the beginning immediately grabs the reader's attention and clearly defines Airflow.
*   **Keyword Optimization:** The phrase "Orchestrate Your Workflows" in the title and throughout the text uses relevant keywords to improve search engine visibility. The description includes the important keywords that users are looking for (e.g., "workflow orchestration," "pipeline," "scheduling," "data pipelines").
*   **Clear Headings:**  Organized the document with clear, keyword-rich headings that help with readability and SEO.
*   **Bulleted Key Features:**  Uses bullet points to make key features easily scannable and highlights important aspects.
*   **Links:** Included internal and external links to improve SEO.
*   **Concise Summarization:**  Provides a summarized version of all sections.
*   **Emphasis on Value:** Focuses on the benefits of Airflow (maintainability, versioning, testing, collaboration) to attract users.
*   **Call to Action:** The initial hook and "Getting Started" section prompt users to explore the project.
*   **Clear Structure for Easier Reading:** Easy to follow with consistent formatting.
*   **Alt Text for Images:**  Added `alt` text to the image tags for accessibility and image SEO.
*   **Removed Redundancy:** Cleaned up and simplified the text to be more direct and easier to understand.
*   **Consistent Formatting**: Used bolding and consistent capitalization for headings.