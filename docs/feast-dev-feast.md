<p align="center">
    <a href="https://feast.dev/">
      <img src="docs/assets/feast_logo.png" width="550" alt="Feast Feature Store Logo">
    </a>
</p>
<br />

[![PyPI - Downloads](https://img.shields.io/pypi/dm/feast)](https://pypi.org/project/feast/)
[![GitHub contributors](https://img.shields.io/github/contributors/feast-dev/feast)](https://github.com/feast-dev/feast/graphs/contributors)
[![unit-tests](https://github.com/feast-dev/feast/actions/workflows/unit_tests.yml/badge.svg?branch=master&event=pull_request)](https://github.com/feast-dev/feast/actions/workflows/unit_tests.yml)
[![integration-tests-and-build](https://github.com/feast-dev/feast/actions/workflows/master_only.yml/badge.svg?branch=master&event=push)](https://github.com/feast-dev/feast/actions/workflows/master_only.yml)
[![linter](https://github.com/feast-dev/feast/actions/workflows/linter.yml/badge.svg?branch=master&event=push)](https://github.com/feast-dev/feast/actions/workflows/linter.yml)
[![Docs Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://docs.feast.dev/)
[![Python API](https://img.shields.io/badge/docs-latest-brightgreen.svg)](http://rtd.feast.dev/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/feast-dev/feast/blob/master/LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/feast-dev/feast.svg?style=flat&sort=semver&color=blue)](https://github.com/feast-dev/feast/releases)

## **Feast: The Open Source Feature Store**

**Feast** is an open-source feature store that helps you build and manage machine learning features for both training and real-time serving, streamlining your ML workflow.

Join us on Slack!
👋👋👋 [Come say hi on Slack!](https://communityinviter.com/apps/feastopensource/feast-the-open-source-feature-store)

[Check out our DeepWiki!](https://deepwiki.com/feast-dev/feast)

<a href="https://trendshift.io/repositories/8046" target="_blank"><img src="https://trendshift.io/api/badge/repositories/8046" alt="feast-dev%2Ffeast | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

## Key Features

*   **Consistent Feature Availability:**  Manage offline and online stores for features, ensuring features are available for both model training and low-latency, real-time prediction.
*   **Data Leakage Prevention:** Generate point-in-time correct feature sets to ensure data consistency.
*   **Decoupled Infrastructure:**  Abstract feature storage and retrieval for model portability across training, serving, and different data infrastructures.
*   **Feature Engineering Capabilities:**  Streamline and automate feature engineering.
*   **Integration with Snowflake, AWS, and more:** Ready-to-use integrations with major cloud providers and data storage platforms.

For more in-depth information, see the [Feast Documentation](https://docs.feast.dev/).

## 📐 Architecture

![](docs/assets/feast_marchitecture.png)

Discover how to run Feast on Snowflake, GCP, or AWS: [Feast Snowflake/GCP/AWS Guide](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws).

## 🐣 Getting Started

### 1. Install Feast
```commandline
pip install feast
```

### 2. Create a feature repository
```commandline
feast init my_feature_repo
cd my_feature_repo/feature_repo
```

### 3. Register your feature definitions and set up your feature store
```commandline
feast apply
```

### 4. Explore your data in the web UI (experimental)

![Web UI](ui/sample.png)
```commandline
feast ui
```

### 5. Build a training dataset
```python
from feast import FeatureStore
import pandas as pd
from datetime import datetime

entity_df = pd.DataFrame.from_dict({
    "driver_id": [1001, 1002, 1003, 1004],
    "event_timestamp": [
        datetime(2021, 4, 12, 10, 59, 42),
        datetime(2021, 4, 12, 8,  12, 10),
        datetime(2021, 4, 12, 16, 40, 26),
        datetime(2021, 4, 12, 15, 1 , 12)
    ]
})

store = FeatureStore(repo_path=".")

training_df = store.get_historical_features(
    entity_df=entity_df,
    features = [
        'driver_hourly_stats:conv_rate',
        'driver_hourly_stats:acc_rate',
        'driver_hourly_stats:avg_daily_trips'
    ],
).to_df()

print(training_df.head())

# Train model
# model = ml.fit(training_df)
```
```commandline
            event_timestamp  driver_id  conv_rate  acc_rate  avg_daily_trips
0 2021-04-12 08:12:10+00:00       1002   0.713465  0.597095              531
1 2021-04-12 10:59:42+00:00       1001   0.072752  0.044344               11
2 2021-04-12 15:01:12+00:00       1004   0.658182  0.079150              220
3 2021-04-12 16:40:26+00:00       1003   0.162092  0.309035              959

```

### 6. Load feature values into your online store
```commandline
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
feast materialize-incremental $CURRENT_TIME
```

```commandline
Materializing feature view driver_hourly_stats from 2021-04-14 to 2021-04-15 done!
```

### 7. Read online features at low latency
```python
from pprint import pprint
from feast import FeatureStore

store = FeatureStore(repo_path=".")

feature_vector = store.get_online_features(
    features=[
        'driver_hourly_stats:conv_rate',
        'driver_hourly_stats:acc_rate',
        'driver_hourly_stats:avg_daily_trips'
    ],
    entity_rows=[{"driver_id": 1001}]
).to_dict()

pprint(feature_vector)

# Make prediction
# model.predict(feature_vector)
```
```json
{
    "driver_id": [1001],
    "driver_hourly_stats__conv_rate": [0.49274],
    "driver_hourly_stats__acc_rate": [0.92743],
    "driver_hourly_stats__avg_daily_trips": [72]
}
```

## 📦 Functionality and Roadmap

Explore the functionality that contributors are planning to develop for Feast:

*   We welcome contributions to all items on the roadmap!

*   **Natural Language Processing**
    *   [x] Vector Search (Alpha release. See [RFC](https://docs.google.com/document/d/18IWzLEA9i2lDWnbfbwXnMCg3StlqaLVI-uRpQjr_Vos/edit#heading=h.9gaqqtox9jg6))
    *   [ ] [Enhanced Feature Server and SDK for native support for NLP](https://github.com/feast-dev/feast/issues/4964)
*   **Data Sources**
    *   [x] [Snowflake source](https://docs.feast.dev/reference/data-sources/snowflake)
    *   [x] [Redshift source](https://docs.feast.dev/reference/data-sources/redshift)
    *   [x] [BigQuery source](https://docs.feast.dev/reference/data-sources/bigquery)
    *   [x] [Parquet file source](https://docs.feast.dev/reference/data-sources/file)
    *   [x] [Azure Synapse + Azure SQL source (contrib plugin)](https://docs.feast.dev/reference/data-sources/mssql)
    *   [x] [Hive (community plugin)](https://github.com/baineng/feast-hive)
    *   [x] [Postgres (contrib plugin)](https://docs.feast.dev/reference/data-sources/postgres)
    *   [x] [Spark (contrib plugin)](https://docs.feast.dev/reference/data-sources/spark)
    *   [x] [Couchbase (contrib plugin)](https://docs.feast.dev/reference/data-sources/couchbase)
    *   [x] Kafka / Kinesis sources (via [push support into the online store](https://docs.feast.dev/reference/data-sources/push))
*   **Offline Stores**
    *   [x] [Snowflake](https://docs.feast.dev/reference/offline-stores/snowflake)
    *   [x] [Redshift](https://docs.feast.dev/reference/offline-stores/redshift)
    *   [x] [BigQuery](https://docs.feast.dev/reference/offline-stores/bigquery)
    *   [x] [Azure Synapse + Azure SQL (contrib plugin)](https://docs.feast.dev/reference/offline-stores/mssql.md)
    *   [x] [Hive (community plugin)](https://github.com/baineng/feast-hive)
    *   [x] [Postgres (contrib plugin)](https://docs.feast.dev/reference/offline-stores/postgres)
    *   [x] [Trino (contrib plugin)](https://github.com/Shopify/feast-trino)
    *   [x] [Spark (contrib plugin)](https://docs.feast.dev/reference/offline-stores/spark)
    *   [x] [Couchbase (contrib plugin)](https://docs.feast.dev/reference/offline-stores/couchbase)
    *   [x] [In-memory / Pandas](https://docs.feast.dev/reference/offline-stores/file)
    *   [x] [Custom offline store support](https://docs.feast.dev/how-to-guides/customizing-feast/adding-a-new-offline-store)
*   **Online Stores**
    *   [x] [Snowflake](https://docs.feast.dev/reference/online-stores/snowflake)
    *   [x] [DynamoDB](https://docs.feast.dev/reference/online-stores/dynamodb)
    *   [x] [Redis](https://docs.feast.dev/reference/online-stores/redis)
    *   [x] [Datastore](https://docs.feast.dev/reference/online-stores/datastore)
    *   [x] [Bigtable](https://docs.feast.dev/reference/online-stores/bigtable)
    *   [x] [SQLite](https://docs.feast.dev/reference/online-stores/sqlite)
    *   [x] [Dragonfly](https://docs.feast.dev/reference/online-stores/dragonfly)
    *   [x] [IKV - Inlined Key Value Store](https://docs.feast.dev/reference/online-stores/ikv)
    *   [x] [Azure Cache for Redis (community plugin)](https://github.com/Azure/feast-azure)
    *   [x] [Postgres (contrib plugin)](https://docs.feast.dev/reference/online-stores/postgres)
    *   [x] [Cassandra / AstraDB (contrib plugin)](https://docs.feast.dev/reference/online-stores/cassandra)
    *   [x] [ScyllaDB (contrib plugin)](https://docs.feast.dev/reference/online-stores/scylladb)
    *   [x] [Couchbase (contrib plugin)](https://docs.feast.dev/reference/online-stores/couchbase)
    *   [x] [Custom online store support](https://docs.feast.dev/how-to-guides/customizing-feast/adding-support-for-a-new-online-store)
*   **Feature Engineering**
    *   [x] On-demand Transformations (On Read) (Beta release. See [RFC](https://docs.google.com/document/d/1lgfIw0Drc65LpaxbUu49RCeJgMew547meSJttnUqz7c/edit#))
    *   [x] Streaming Transformations (Alpha release. See [RFC](https://docs.google.com/document/d/1UzEyETHUaGpn0ap4G82DHluiCj7zEbrQLkJJkKSv4e8/edit))
    *   [ ] Batch transformation (In progress. See [RFC](https://docs.google.com/document/d/1964OkzuBljifDvkV-0fakp2uaijnVzdwWNGdz7Vz50A/edit))
    *   [x] On-demand Transformations (On Write) (Beta release. See [GitHub Issue](https://github.com/feast-dev/feast/issues/4376))
*   **Streaming**
    *   [x] [Custom streaming ingestion job support](https://docs.feast.dev/how-to-guides/customizing-feast/creating-a-custom-provider)
    *   [x] [Push based streaming data ingestion to online store](https://docs.feast.dev/reference/data-sources/push)
    *   [x] [Push based streaming data ingestion to offline store](https://docs.feast.dev/reference/data-sources/push)
*   **Deployments**
    *   [x] AWS Lambda (Alpha release. See [RFC](https://docs.google.com/document/d/1eZWKWzfBif66LDN32IajpaG-j82LSHCCOzY6R7Ax7MI/edit))
    *   [x] Kubernetes (See [guide](https://docs.feast.dev/how-to-guides/running-feast-in-production))
*   **Feature Serving**
    *   [x] Python Client
    *   [x] [Python feature server](https://docs.feast.dev/reference/feature-servers/python-feature-server)
    *   [x] [Feast Operator (alpha)](https://github.com/feast-dev/feast/blob/master/infra/feast-operator/README.md)
    *   [x] [Java feature server (alpha)](https://github.com/feast-dev/feast/blob/master/infra/charts/feast/README.md)
    *   [x] [Go feature server (alpha)](https://docs.feast.dev/reference/feature-servers/go-feature-server)
    *   [x] [Offline Feature Server (alpha)](https://docs.feast.dev/reference/feature-servers/offline-feature-server)
    *   [x] [Registry server (alpha)](https://github.com/feast-dev/feast/blob/master/docs/reference/feature-servers/registry-server.md)
*   **Data Quality Management (See [RFC](https://docs.google.com/document/d/110F72d4NTv80p35wDSONxhhPBqWRwbZXG4f9mNEMd98/edit))**
    *   [x] Data profiling and validation (Great Expectations)
*   **Feature Discovery and Governance**
    *   [x] Python SDK for browsing feature registry
    *   [x] CLI for browsing feature registry
    *   [x] Model-centric feature tracking (feature services)
    *   [x] Amundsen integration (see [Feast extractor](https://github.com/amundsen-io/amundsen/blob/main/databuilder/databuilder/extractor/feast_extractor.py))
    *   [x] DataHub integration (see [DataHub Feast docs](https://datahubproject.io/docs/generated/ingestion/sources/feast/))
    *   [x] Feast Web UI (Beta release. See [docs](https://docs.feast.dev/reference/alpha-web-ui))
    *   [ ] Feast Lineage Explorer

## 🎓 Important Resources

*   **[Quickstart](https://docs.feast.dev/getting-started/quickstart)**
*   **[Tutorials](https://docs.feast.dev/tutorials/tutorials-overview)**
*   **[Examples](https://github.com/feast-dev/feast/tree/master/examples)**
*   **[Running Feast with Snowflake/GCP/AWS](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws)**
*   **[Change Log](https://github.com/feast-dev/feast/blob/master/CHANGELOG.md)**

Explore the official documentation: [Feast Documentation](https://docs.feast.dev/)

## 👋 Contributing

Feast is a community-driven project, and contributions are highly encouraged. Please review the [Feast Contribution Process](https://docs.feast.dev/project/contributing) and [Development Guide](https://docs.feast.dev/project/development-guide) to get started. Refer to the [Main Feast Repository Development Guide](./CONTRIBUTING.md) for repository-specific guidelines.

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

A big thank you to all of our amazing contributors!

<a href="https://github.com/feast-dev/feast/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=feast-dev/feast" alt="Contributors" />
</a>

[Back to Top](#feast-the-open-source-feature-store)
```
Key improvements and SEO considerations:

*   **Clear, Concise Title:**  The H1 heading uses the project's name and a keyword-rich description ("Open Source Feature Store").
*   **Concise Introduction:** The hook sentence directly tells the user what Feast does.
*   **Keyword Optimization:**  Uses terms like "feature store," "machine learning," "training," and "real-time serving" to target relevant search queries.
*   **Bulleted Feature List:**  Highlights key benefits in a reader-friendly format.
*   **Clear Section Headings:** Uses H2 and H3 for better readability and SEO structure.
*   **Calls to Action:**  Encourages users to explore the documentation, join the community, and contribute.
*   **Contextual Links:** Provides links back to key resources and the original repository.
*   **Contributor Section:**  Showcases contributors, which is good for community engagement and can improve search ranking.
*   **Added alt text to images:** Improves accessibility and SEO.
*   **Back to Top link:** Easy navigation.