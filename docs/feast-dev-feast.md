# Feast: The Open Source Feature Store for Machine Learning

**Feast** is the leading open-source feature store, designed to streamline and accelerate the machine learning lifecycle by providing a single source of truth for features.  [Explore the Feast Repository on GitHub](https://github.com/feast-dev/feast).

[![PyPI - Downloads](https://img.shields.io/pypi/dm/feast)](https://pypi.org/project/feast/)
[![GitHub contributors](https://img.shields.io/github/contributors/feast-dev/feast)](https://github.com/feast-dev/feast/graphs/contributors)
[![unit-tests](https://github.com/feast-dev/feast/actions/workflows/unit_tests.yml/badge.svg?branch=master&event=pull_request)](https://github.com/feast-dev/feast/actions/workflows/unit_tests.yml)
[![integration-tests-and-build](https://github.com/feast-dev/feast/actions/workflows/master_only.yml/badge.svg?branch=master&event=push)](https://github.com/feast-dev/feast/actions/workflows/master_only.yml)
[![linter](https://github.com/feast-dev/feast/actions/workflows/linter.yml/badge.svg?branch=master&event=push)](https://github.com/feast-dev/feast/actions/workflows/linter.yml)
[![Docs Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://docs.feast.dev/)
[![Python API](https://img.shields.io/badge/docs-latest-brightgreen.svg)](http://rtd.feast.dev/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/feast-dev/feast/blob/master/LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/feast-dev/feast.svg?style=flat&sort=semver&color=blue)](https://github.com/feast-dev/feast/releases)

## Key Features of Feast

*   **Unified Feature Access:**  Provides consistent access to features for both training and serving, managing offline and online stores, and a feature server for low-latency retrieval.
*   **Data Leakage Prevention:** Generates point-in-time correct feature sets to avoid data leakage, ensuring model accuracy and reliability.
*   **Infrastructure Decoupling:** Abstract feature storage and retrieval, enabling portability across training/serving, batch/real-time models, and various data infrastructures.
*   **Scalable Feature Management:** Efficiently manages and serves large volumes of features, essential for production-scale machine learning.
*   **Feature Engineering Support:** Offers on-demand and streaming transformations to create and transform features.

## How Feast Works

Feast simplifies the machine learning workflow by providing a centralized feature store:

1.  **Install Feast:** `pip install feast`
2.  **Create a Feature Repository:** `feast init my_feature_repo`
3.  **Define and Apply Features:**  Register feature definitions and set up your feature store using `feast apply`.
4.  **Explore Data (Experimental):** Access the web UI ( `feast ui` ).
5.  **Build Training Datasets:** Easily create training datasets using the Python SDK.
6.  **Materialize Features:** Load feature values into your online store using `feast materialize-incremental $CURRENT_TIME`.
7.  **Read Online Features:** Retrieve low-latency feature values for real-time predictions using the Python SDK.

## 📐 Architecture

[![](docs/assets/feast_marchitecture.png)](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws)
*Click on the image to see the Feast architecture*

## 📦 Functionality and Roadmap

Feast is continuously evolving with new features and improvements. The current roadmap includes:

*   **Natural Language Processing:** Vector Search (Alpha release)
*   **Data Sources:** Snowflake, Redshift, BigQuery, Parquet, Azure Synapse, Azure SQL, Hive, Postgres, Spark, Couchbase, Kafka/Kinesis and more!
*   **Offline Stores:** Snowflake, Redshift, BigQuery, Azure Synapse, Azure SQL, Hive, Postgres, Trino, Spark, Couchbase, In-memory/Pandas, Custom offline store support and more!
*   **Online Stores:** Snowflake, DynamoDB, Redis, Datastore, Bigtable, SQLite, Dragonfly, IKV, Azure Cache for Redis, Postgres, Cassandra/AstraDB, ScyllaDB, Couchbase, Custom online store support and more!
*   **Feature Engineering:** On-demand Transformations (On Read & Write), Streaming Transformations (Alpha release) and Batch transformation (In progress)
*   **Streaming:** Custom streaming ingestion job support, Push based streaming data ingestion to online/offline store
*   **Deployments:** AWS Lambda (Alpha release), Kubernetes
*   **Feature Serving:** Python Client, Python/Java/Go/Offline feature servers, Feast Operator, Registry Server
*   **Data Quality Management:** Data profiling and validation (Great Expectations)
*   **Feature Discovery and Governance:** Python SDK, CLI, Model-centric feature tracking, Amundsen & DataHub integration, Feast Web UI, Feast Lineage Explorer

## 🎓 Important Resources

*   [Official Documentation](https://docs.feast.dev/)
*   [Quickstart](https://docs.feast.dev/getting-started/quickstart)
*   [Tutorials](https://docs.feast.dev/tutorials/tutorials-overview)
*   [Examples](https://github.com/feast-dev/feast/tree/master/examples)
*   [Running Feast with Snowflake/GCP/AWS](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws)
*   [Change Log](https://github.com/feast-dev/feast/blob/master/CHANGELOG.md)

## 👋 Contributing

Feast thrives on community contributions! Review the:

*   [Contribution Process for Feast](https://docs.feast.dev/project/contributing)
*   [Development Guide for Feast](https://docs.feast.dev/project/development-guide)
*   [Development Guide for the Main Feast Repository](./CONTRIBUTING.md)

## 🌟 GitHub Star History

<p align="center">
<a href="https://star-history.com/#feast-dev/feast&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=feast-dev/feast&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=feast-dev/feast&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=feast-dev/feast&type=Date" />
 </picture>
</a>
</p>

## ✨ Contributors

Feast is made possible by these amazing contributors:

<a href="https://github.com/feast-dev/feast/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=feast-dev/feast" />
</a>