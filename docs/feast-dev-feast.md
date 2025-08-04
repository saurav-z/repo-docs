<p align="center">
    <a href="https://feast.dev/">
      <img src="docs/assets/feast_logo.png" width="550" alt="Feast Logo">
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

## 🎉 Feast: Your Open-Source Feature Store for Machine Learning

Feast (**Fea**ture **St**ore) is the open-source feature store that simplifies the journey from raw data to production-ready features, accelerating machine learning pipelines.  Discover how Feast can help you build and manage features effectively.

### Key Features:

*   **Consistent Feature Availability:** Manage both offline and online stores to ensure features are consistently available for both model training and real-time predictions.
*   **Data Leakage Prevention:** Generate point-in-time correct feature sets to avoid data leakage, allowing data scientists to focus on model building.
*   **Decoupled Infrastructure:** Abstract feature storage from retrieval, enabling seamless transitions between training and serving, batch and real-time models, and different data infrastructure systems.
*   **Feature Serving:** Easily serve pre-computed features online for low-latency predictions.
*   **Comprehensive Data Source Support:** Integrate with a wide variety of data sources, including Snowflake, BigQuery, Redshift, and more.
*   **Extensible and Customizable:** Benefit from a growing ecosystem of community plugins and the ability to customize your feature store.

For more details and in-depth information about Feast, visit our [documentation](https://docs.feast.dev/).

## 📐 Architecture Overview
<img src="docs/assets/feast_marchitecture.png" alt="Feast Architecture Diagram">

### Deployment Options

Want to run the full Feast on Snowflake/GCP/AWS? Click [here](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws).

## 🚀 Getting Started

### 1.  Install Feast
```bash
pip install feast
```

### 2.  Create a Feature Repository
```bash
feast init my_feature_repo
cd my_feature_repo/feature_repo
```

### 3.  Register Feature Definitions and Set Up Your Feature Store
```bash
feast apply
```

### 4.  Explore Your Data (Experimental) in the Web UI
```bash
feast ui
```
![Web UI](ui/sample.png)

### 5. Build a Training Dataset
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
```
            event_timestamp  driver_id  conv_rate  acc_rate  avg_daily_trips
0 2021-04-12 08:12:10+00:00       1002   0.713465  0.597095              531
1 2021-04-12 10:59:42+00:00       1001   0.072752  0.044344               11
2 2021-04-12 15:01:12+00:00       1004   0.658182  0.079150              220
3 2021-04-12 16:40:26+00:00       1003   0.162092  0.309035              959

```

### 6. Load Feature Values into Your Online Store
```bash
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
feast materialize-incremental $CURRENT_TIME
```

```
Materializing feature view driver_hourly_stats from 2021-04-14 to 2021-04-15 done!
```

### 7. Read Online Features at Low Latency
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

## 🗓️ Functionality and Roadmap

Explore the future of Feast:

*   **Natural Language Processing**
    *   [x] Vector Search (Alpha release)
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
    *   [x] On-demand Transformations (On Read) (Beta release)
    *   [x] Streaming Transformations (Alpha release)
    *   [ ] Batch transformation (In progress)
    *   [x] On-demand Transformations (On Write) (Beta release)
*   **Streaming**
    *   [x] [Custom streaming ingestion job support](https://docs.feast.dev/how-to-guides/customizing-feast/creating-a-custom-provider)
    *   [x] [Push based streaming data ingestion to online store](https://docs.feast.dev/reference/data-sources/push)
    *   [x] [Push based streaming data ingestion to offline store](https://docs.feast.dev/reference/data-sources/push)
*   **Deployments**
    *   [x] AWS Lambda (Alpha release)
    *   [x] Kubernetes (See [guide](https://docs.feast.dev/how-to-guides/running-feast-in-production))
*   **Feature Serving**
    *   [x] Python Client
    *   [x] [Python feature server](https://docs.feast.dev/reference/feature-servers/python-feature-server)
    *   [x] [Feast Operator (alpha)](https://github.com/feast-dev/feast/blob/master/infra/feast-operator/README.md)
    *   [x] [Java feature server (alpha)](https://github.com/feast-dev/feast/blob/master/infra/charts/feast/README.md)
    *   [x] [Go feature server (alpha)](https://docs.feast.dev/reference/feature-servers/go-feature-server)
    *   [x] [Offline Feature Server (alpha)](https://docs.feast.dev/reference/feature-servers/offline-feature-server)
    *   [x] [Registry server (alpha)](https://github.com/feast-dev/feast/blob/master/docs/reference/feature-servers/registry-server.md)
*   **Data Quality Management**
    *   [x] Data profiling and validation (Great Expectations)
*   **Feature Discovery and Governance**
    *   [x] Python SDK for browsing feature registry
    *   [x] CLI for browsing feature registry
    *   [x] Model-centric feature tracking (feature services)
    *   [x] Amundsen integration (see [Feast extractor](https://github.com/amundsen-io/amundsen/blob/main/databuilder/databuilder/extractor/feast_extractor.py))
    *   [x] DataHub integration (see [DataHub Feast docs](https://datahubproject.io/docs/generated/ingestion/sources/feast/))
    *   [x] Feast Web UI (Beta release)
    *   [ ] Feast Lineage Explorer

## 📚 Important Resources

Find everything you need to get started:

*   [Documentation](https://docs.feast.dev/)
    *   [Quickstart](https://docs.feast.dev/getting-started/quickstart)
    *   [Tutorials](https://docs.feast.dev/tutorials/tutorials-overview)
    *   [Examples](https://github.com/feast-dev/feast/tree/master/examples)
    *   [Running Feast with Snowflake/GCP/AWS](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws)
    *   [Change Log](https://github.com/feast-dev/feast/blob/master/CHANGELOG.md)

## 👋 Contributing

Join the Feast community! Review the contribution guidelines:

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

A big thanks to everyone who contributes!

<a href="https://github.com/feast-dev/feast/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=feast-dev/feast" alt="Contributors"/>
</a>
```

Key improvements and explanations:

*   **SEO Optimization:** Included relevant keywords in the headings and body, like "feature store," "machine learning," and related terms.  The introductory sentence is a concise and compelling hook.
*   **Clear Headings:** Used clear, concise headings to structure the document and improve readability and scanning.
*   **Bulleted Key Features:**  Presented the key features in bulleted lists for easy consumption and better SEO.
*   **Concise Summarization:** Streamlined the text while retaining the key information.  Removed redundant phrases.
*   **Emphasis on Benefits:**  Highlighted the benefits of using Feast.
*   **Call to Action (Implied):** The clear structure encourages users to explore the documentation and get started.
*   **Improved Formatting:** Used more standard Markdown formatting (e.g., `###` for sub-headings) for better readability and rendering across different platforms.
*   **Alt text for images:** Added `alt` text to images, crucial for accessibility and SEO.
*   **Added contributor images with a useful title and alt text**: Added image with `alt` text
*   **Removed irrelevant links**: Removing links to external sources that do not have to be in the README file
*   **Link back to the original repo**: Included a link at the very top, as requested.  Added a link to see all contributors.
*   **Added table of contents (implied)**: Headings create an implied table of contents that is easy to scan.
*   **Included examples**: Examples are present to help the user get started
*   **Corrected errors in Markdown**: Cleaned up and corrected the Markdown.
*   **Removed unnecessary code:** Removed unneeded code.