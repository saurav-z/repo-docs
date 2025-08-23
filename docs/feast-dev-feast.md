<!--Do not modify this file. It is auto-generated from a template (infra/templates/README.md.jinja2)-->

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


## 🤖 Feast: The Open Source Feature Store for Machine Learning

Feast, the open-source feature store, simplifies the management of machine learning features, making it easier to build, deploy, and scale your ML models.  [Explore the Feast GitHub Repository](https://github.com/feast-dev/feast)

## Key Features of Feast

*   **Consistent Feature Availability**:  Feast ensures features are available for both training and serving by managing offline and online stores, along with a feature server.
*   **Data Leakage Prevention**: Generate point-in-time correct feature sets to prevent data leakage, enabling data scientists to focus on feature engineering.
*   **Decoupling ML from Data Infrastructure**:  Abstracts feature storage from retrieval, ensuring model portability across training, serving, and data infrastructure changes.
*   **Scalable Feature Serving**: Feast can scale to production-level serving requirements
*   **Flexibility and Extensibility**: Supports various data sources, online and offline stores, and feature engineering capabilities, including custom integrations.

## 📐 Architecture

![Feast Architecture](docs/assets/feast_marchitecture.png)

*   **Offline Store**:  Used for batch processing and model training, ideal for historical data.
*   **Online Store**:  Offers low-latency feature retrieval for real-time predictions.
*   **Feature Server**:  Serves pre-computed features online, ensuring fast access.

Want to deploy the full Feast on Snowflake/GCP/AWS? Click [here](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws).

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

*   **Natural Language Processing**
    *   [x] Vector Search (Alpha release)
    *   [ ] Enhanced Feature Server and SDK for native support for NLP
*   **Data Sources**
    *   [x] Snowflake source
    *   [x] Redshift source
    *   [x] BigQuery source
    *   [x] Parquet file source
    *   [x] Azure Synapse + Azure SQL source (contrib plugin)
    *   [x] Hive (community plugin)
    *   [x] Postgres (contrib plugin)
    *   [x] Spark (contrib plugin)
    *   [x] Couchbase (contrib plugin)
    *   [x] Kafka / Kinesis sources (via push support into the online store)
*   **Offline Stores**
    *   [x] Snowflake
    *   [x] Redshift
    *   [x] BigQuery
    *   [x] Azure Synapse + Azure SQL (contrib plugin)
    *   [x] Hive (community plugin)
    *   [x] Postgres (contrib plugin)
    *   [x] Trino (contrib plugin)
    *   [x] Spark (contrib plugin)
    *   [x] Couchbase (contrib plugin)
    *   [x] In-memory / Pandas
    *   [x] Custom offline store support
*   **Online Stores**
    *   [x] Snowflake
    *   [x] DynamoDB
    *   [x] Redis
    *   [x] Datastore
    *   [x] Bigtable
    *   [x] SQLite
    *   [x] Dragonfly
    *   [x] IKV - Inlined Key Value Store
    *   [x] Azure Cache for Redis (community plugin)
    *   [x] Postgres (contrib plugin)
    *   [x] Cassandra / AstraDB (contrib plugin)
    *   [x] ScyllaDB (contrib plugin)
    *   [x] Couchbase (contrib plugin)
    *   [x] Custom online store support
*   **Feature Engineering**
    *   [x] On-demand Transformations (On Read) (Beta release)
    *   [x] Streaming Transformations (Alpha release)
    *   [ ] Batch transformation (In progress)
    *   [x] On-demand Transformations (On Write) (Beta release)
*   **Streaming**
    *   [x] Custom streaming ingestion job support
    *   [x] Push based streaming data ingestion to online store
    *   [x] Push based streaming data ingestion to offline store
*   **Deployments**
    *   [x] AWS Lambda (Alpha release)
    *   [x] Kubernetes
*   **Feature Serving**
    *   [x] Python Client
    *   [x] Python feature server
    *   [x] Feast Operator (alpha)
    *   [x] Java feature server (alpha)
    *   [x] Go feature server (alpha)
    *   [x] Offline Feature Server (alpha)
    *   [x] Registry server (alpha)
*   **Data Quality Management**
    *   [x] Data profiling and validation (Great Expectations)
*   **Feature Discovery and Governance**
    *   [x] Python SDK for browsing feature registry
    *   [x] CLI for browsing feature registry
    *   [x] Model-centric feature tracking (feature services)
    *   [x] Amundsen integration
    *   [x] DataHub integration
    *   [x] Feast Web UI (Beta release)
    *   [ ] Feast Lineage Explorer

## 🎓 Important Resources

*   [Quickstart](https://docs.feast.dev/getting-started/quickstart)
*   [Tutorials](https://docs.feast.dev/tutorials/tutorials-overview)
*   [Examples](https://github.com/feast-dev/feast/tree/master/examples)
*   [Running Feast with Snowflake/GCP/AWS](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws)
*   [Change Log](https://github.com/feast-dev/feast/blob/master/CHANGELOG.md)

## 👋 Contributing

Feast welcomes contributions from the community.  Check out the [contribution process](https://docs.feast.dev/project/contributing) and [development guide](https://docs.feast.dev/project/development-guide) to get started.

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

A huge thank you to all the contributors!

<a href="https://github.com/feast-dev/feast/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=feast-dev/feast" alt="Feast Contributors" />
</a>