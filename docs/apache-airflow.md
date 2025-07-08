# Apache Airflow: The Open-Source Platform for Orchestrating Workflows

**Apache Airflow** is a platform that allows you to programmatically author, schedule, and monitor your workflows, making them more maintainable, versionable, testable, and collaborative.  Visit the [original repository](https://github.com/apache/airflow) for more information.

## Key Features

*   **Dynamic:** Define pipelines as code for dynamic DAG generation and parameterization.
*   **Extensible:** Utilize a wide range of built-in operators and customize Airflow to fit your needs.
*   **Flexible:** Leverage the [Jinja](https://jinja.palletsprojects.com) templating engine for rich customizations.

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

## Installation

Detailed installation instructions are available in the [INSTALLING.md](INSTALLING.md) file.

### Installing from PyPI

Install Apache Airflow from PyPI using the following approach to ensure repeatable installations:

```bash
pip install 'apache-airflow==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

Install with extras (e.g., postgres, google):

```bash
pip install 'apache-airflow[postgres,google]==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

For more on installing provider distributions, check [providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html).

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

## Versioning and Lifecycle

Airflow uses [Semantic Versioning](https://semver.org/) (SemVer) with the following version lifecycle:

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.2                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

## Contributing

Contribute to Airflow by following the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

## Additional Resources

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>
```

Key improvements and optimizations:

*   **SEO-friendly headings:**  Uses clear, descriptive headings (e.g., "Key Features," "Installation") to improve readability and searchability.
*   **Concise summary & hook:** The first sentence is a strong, keyword-rich hook.
*   **Bulleted lists:** Uses bullet points for key features and other lists, making the content easy to scan.
*   **Keyword optimization:** Includes relevant keywords such as "orchestrate workflows," "programmatically," "schedule," "monitor," "DAG," "pipeline," "data pipelines" etc.
*   **Clear calls to action:** Directs users to the key resources (installation, contributing, documentation).
*   **Focus on benefits:**  Highlights *why* users should use Airflow (maintainable, versionable, etc.).
*   **Removed redundant content:** Cleaned up the text to be more focused and impactful, removing unnecessary introductory text, and shortening the content.
*   **Table for Requirements:** A well-formatted table makes reading the requirements far easier.
*   **Links:** Direct links to the repository, documentation, and other important resources.
*   **Concise, actionable installation instructions:**  Includes the necessary pip installation commands, with important caveats and recommendations.
*   **Formatting:** Uses markdown formatting for bolding, lists, and other elements to enhance readability.
*   **Version Lifecycle table:** Added the version lifecycle table to ensure information is easily accessible.