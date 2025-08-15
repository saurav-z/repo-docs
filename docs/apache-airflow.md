# Apache Airflow: Orchestrate Your Workflows with Code

**Apache Airflow is a powerful platform for programmatically authoring, scheduling, and monitoring workflows.**  Define your workflows as code for enhanced maintainability, versioning, and collaboration.  [Explore the original repository](https://github.com/apache/airflow).

## Key Features

*   **Dynamic Workflows:** Define pipelines in code, enabling dynamic DAG generation and parameterization.
*   **Extensible Architecture:** Leverage a wide range of built-in operators and extend Airflow to meet your unique needs.
*   **Flexible Templating:** Customize workflows with ease using the powerful [**Jinja**](https://jinja.palletsprojects.com) templating engine.

## Installing Apache Airflow

To use Apache Airflow, you can install it with the following methods:

### Installing from PyPI

Install Apache Airflow with `pip install apache-airflow` or install it with extras such as google using `pip install 'apache-airflow[postgres,google]'`. For a more repeatable installation, use constraint files found in the `constraints-main` and `constraints-2-0` branches.

### Installation

For comprehensive instructions on setting up your local development environment and installing Apache Airflow, please refer to the [INSTALLING.md](INSTALLING.md) file.

### Official source code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project.  Our official source code releases:

- Follow the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
- Can be downloaded from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
- Are cryptographically signed by the release manager
- Are officially voted on by the PMC members during the
  [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

Following the ASF rules, the source packages released must be sufficient for a user to build and test the
release provided they have access to the appropriate platform and tools.

## User Interface

Airflow offers a rich user interface for visualizing and managing your workflows:

*   **DAGs:** Overview of all DAGs in your environment.
    ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)

*   **Assets:** Overview of Assets with dependencies.
    ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)

*   **Grid:** Grid representation of a DAG that spans across time.
    ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)

*   **Graph:** Visualization of a DAG's dependencies and their current status for a specific run.
    ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)

*   **Home:** Summary statistics of your Airflow environment.
    ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)

*   **Backfill:** Backfilling a DAG for a specific date range.
    ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)

*   **Code:** Quick way to view source code of a DAG.
    ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)

## Support for Python and Kubernetes versions

*   Airflow supports various Python and Kubernetes versions. Check out the latest stable version.

## Version Life Cycle

Apache Airflow version life cycle:

<!-- This table is automatically updated by pre-commit scripts/ci/pre_commit/supported_versions.py -->
<!-- Beginning of auto-generated table -->

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.4                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

<!-- End of auto-generated table -->

## Contributing

Contribute to Apache Airflow by reviewing the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) and [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst).

## Voting Policy

* Commits need a +1 vote from a committer who is not the author
* When we do AIP voting, both PMC member's and committer's `+1s` are considered a binding vote.

## Who uses Apache Airflow?

Find out which organizations are using Apache Airflow [in the wild](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who maintains Apache Airflow?

The [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow) and the [community](https://github.com/apache/airflow/graphs/contributors) are responsible for reviewing and merging PRs.

## Links

-   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
-   [Chat](https://s.apache.org/airflow-slack)
-   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>