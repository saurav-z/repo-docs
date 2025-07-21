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

## **Feast: The Open Source Feature Store for Machine Learning**

Feast is an open-source feature store designed to streamline the machine learning pipeline by providing a centralized and consistent way to manage, serve, and discover features.  [Visit the original repo](https://github.com/feast-dev/feast)

## Key Features

*   **Feature Availability for Training and Serving:** Manage features across offline and online stores for both model training and real-time prediction using an offline store for batch processing and an online store for low-latency retrieval.
*   **Data Leakage Prevention:**  Generate point-in-time correct feature sets, ensuring data scientists focus on feature engineering without worrying about data leakage during model training.
*   **ML/Data Infrastructure Decoupling:** Abstract feature storage from feature retrieval, facilitating portability as you move from training to serving, batch to real-time models, and different data infrastructure systems.
*   **Feature Server:** Serve pre-computed features online.
*   **Feature Registry:** Maintain a single source of truth for feature definitions.
*   **Integration with various data sources and online/offline stores.**

## Architecture

Feast's architecture is designed to be flexible and scalable.

![](docs/assets/feast_marchitecture.png)

## Getting Started

Get started quickly with these simple steps:

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

## Functionality and Roadmap

Feast is continuously evolving with new features and integrations. See the [Feast documentation](https://docs.feast.dev/) for a complete and up-to-date list. The roadmap includes:

*   **Natural Language Processing**
    *   [x] Vector Search (Alpha release)
    *   [ ] Enhanced Feature Server and SDK for native support for NLP
*   **Data Sources**
    *   [x] Snowflake, Redshift, BigQuery, and Parquet
    *   [x] Azure Synapse + Azure SQL (contrib plugin)
    *   [x] Hive, Postgres, Spark, Couchbase (contrib plugins)
    *   [x] Kafka / Kinesis sources (via push support)
*   **Offline Stores**
    *   [x] Snowflake, Redshift, BigQuery
    *   [x] Azure Synapse + Azure SQL, Hive, Postgres, Trino, Spark, Couchbase (contrib plugins)
    *   [x] In-memory / Pandas, Custom offline store support
*   **Online Stores**
    *   [x] Snowflake, DynamoDB, Redis, Datastore, Bigtable, SQLite, Dragonfly, IKV
    *   [x] Azure Cache for Redis, Postgres, Cassandra / AstraDB, ScyllaDB, Couchbase (contrib plugins)
    *   [x] Custom online store support
*   **Feature Engineering**
    *   [x] On-demand Transformations (On Read & Write)
    *   [x] Streaming Transformations (Alpha release)
    *   [ ] Batch transformation (In progress)
*   **Streaming**
    *   [x] Custom streaming ingestion job support
    *   [x] Push based streaming data ingestion to online/offline store
*   **Deployments**
    *   [x] AWS Lambda (Alpha release)
    *   [x] Kubernetes
*   **Feature Serving**
    *   [x] Python Client, Python feature server
    *   [x] Feast Operator (alpha), Java feature server (alpha), Go feature server (alpha), Offline Feature Server (alpha), Registry server (alpha)
*   **Data Quality Management**
    *   [x] Data profiling and validation (Great Expectations)
*   **Feature Discovery and Governance**
    *   [x] Python SDK & CLI for browsing feature registry
    *   [x] Model-centric feature tracking (feature services)
    *   [x] Amundsen, DataHub, Feast Web UI integrations

## Important Resources

*   [Documentation](https://docs.feast.dev/)
    *   [Quickstart](https://docs.feast.dev/getting-started/quickstart)
    *   [Tutorials](https://docs.feast.dev/tutorials/tutorials-overview)
    *   [Examples](https://github.com/feast-dev/feast/tree/master/examples)
    *   [Running Feast with Snowflake/GCP/AWS](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws)
    *   [Change Log](https://github.com/feast-dev/feast/blob/master/CHANGELOG.md)

## Contributing

We welcome contributions!  Check out the following guides:

*   [Contribution Process for Feast](https://docs.feast.dev/project/contributing)
*   [Development Guide for Feast](https://docs.feast.dev/project/development-guide)
*   [Development Guide for the Main Feast Repository](./CONTRIBUTING.md)

## Star History

<p align="center">
<a href="https://star-history.com/#feast-dev/feast&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=feast-dev/feast&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=feast-dev/feast&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=feast-dev/feast&type=Date" />
 </picture>
</a>
</p>

## Contributors

Thank you to the amazing contributors:

<a href="https://github.com/feast-dev/feast/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=feast-dev/feast" />
</a>