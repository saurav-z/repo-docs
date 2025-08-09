<!--Do not modify this file. It is auto-generated from a template (infra/templates/README.md.jinja2)-->

<p align="center">
    <a href="https://feast.dev/">
      <img src="docs/assets/feast_logo.png" width="550" alt="Feast Feature Store Logo">
    </a>
</p>
<br />

[![PyPI - Downloads](https://img.shields.io/pypi/dm/feast)](https://pypi.org/project/feast/)
[![GitHub contributors](https://img.shields.io/github/contributors/feast-dev/feast)](https://github.com/feast-dev/feast/graphs/contributors)
[![Unit Tests](https://github.com/feast-dev/feast/actions/workflows/unit_tests.yml/badge.svg?branch=master&event=pull_request)](https://github.com/feast-dev/feast/actions/workflows/unit_tests.yml)
[![Integration Tests & Build](https://github.com/feast-dev/feast/actions/workflows/master_only.yml/badge.svg?branch=master&event=push)](https://github.com/feast-dev/feast/actions/workflows/master_only.yml)
[![Linter](https://github.com/feast-dev/feast/actions/workflows/linter.yml/badge.svg?branch=master&event=push)](https://github.com/feast-dev/feast/actions/workflows/linter.yml)
[![Docs Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://docs.feast.dev/)
[![Python API](https://img.shields.io/badge/docs-latest-brightgreen.svg)](http://rtd.feast.dev/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/feast-dev/feast/blob/master/LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/feast-dev/feast.svg?style=flat&sort=semver&color=blue)](https://github.com/feast-dev/feast/releases)


## 📢 Simplify Machine Learning with Feast: The Open-Source Feature Store

Feast is a powerful, open-source feature store designed to streamline and optimize the machine learning lifecycle. [Explore the Feast GitHub Repository](https://github.com/feast-dev/feast) to learn more.

**Key Features:**

*   **Consistent Feature Availability:** Manage features consistently across training and serving with offline and online stores, and a feature server.
*   **Data Leakage Prevention:** Generate point-in-time correct feature sets to prevent data leakage and improve model accuracy.
*   **Decoupled Infrastructure:** Abstract feature storage from feature retrieval, ensuring model portability across training, serving, batch, and real-time environments.
*   **Scalability:** Handle large datasets efficiently for both model training and online prediction.
*   **Feature Engineering Support:**  Enables you to create and manage feature pipelines to transform raw data into valuable features.
*   **Integration:** Seamlessly integrates with popular data sources, online stores, and machine learning frameworks.
*   **Extensible:** Offers custom offline and online store support to fit various data infrastructure requirements.

## 📐 Architecture

![](docs/assets/feast_marchitecture.png)

*A minimal Feast deployment architecture*

Want to run the full Feast on Snowflake/GCP/AWS? Click [here](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws).

## 🚀 Getting Started

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

Feast's roadmap includes:

*   **Natural Language Processing**
    *   [x] Vector Search (Alpha)
    *   [ ] Enhanced Feature Server and SDK for native support for NLP
*   **Data Sources**
    *   [x] Snowflake, Redshift, BigQuery, Parquet file, Azure Synapse + Azure SQL, Hive, Postgres, Spark, Couchbase, Kafka / Kinesis sources (via push support)
*   **Offline Stores**
    *   [x] Snowflake, Redshift, BigQuery, Azure Synapse + Azure SQL, Hive, Postgres, Trino, Spark, Couchbase, In-memory / Pandas, Custom offline store support
*   **Online Stores**
    *   [x] Snowflake, DynamoDB, Redis, Datastore, Bigtable, SQLite, Dragonfly, IKV - Inlined Key Value Store, Azure Cache for Redis, Postgres, Cassandra / AstraDB, ScyllaDB, Couchbase, Custom online store support
*   **Feature Engineering**
    *   [x] On-demand Transformations (On Read) (Beta)
    *   [x] Streaming Transformations (Alpha)
    *   [ ] Batch transformation (In progress)
    *   [x] On-demand Transformations (On Write) (Beta)
*   **Streaming**
    *   [x] Custom streaming ingestion job support, Push based streaming data ingestion to online store, Push based streaming data ingestion to offline store
*   **Deployments**
    *   [x] AWS Lambda (Alpha)
    *   [x] Kubernetes
*   **Feature Serving**
    *   [x] Python Client, Python feature server, Feast Operator (alpha), Java feature server (alpha), Go feature server (alpha), Offline Feature Server (alpha), Registry server (alpha)
*   **Data Quality Management (See [RFC](https://docs.google.com/document/d/110F72d4NTv80p35wDSONxhhPBqWRwbZXG4f9mNEMd98/edit))**
    *   [x] Data profiling and validation (Great Expectations)
*   **Feature Discovery and Governance**
    *   [x] Python SDK for browsing feature registry, CLI for browsing feature registry, Model-centric feature tracking (feature services), Amundsen integration, DataHub integration, Feast Web UI (Beta)
    *   [ ] Feast Lineage Explorer

## 🎓 Important Resources

*   [Documentation](https://docs.feast.dev/)
    *   [Quickstart](https://docs.feast.dev/getting-started/quickstart)
    *   [Tutorials](https://docs.feast.dev/tutorials/tutorials-overview)
    *   [Examples](https://github.com/feast-dev/feast/tree/master/examples)
    *   [Running Feast with Snowflake/GCP/AWS](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws)
    *   [Change Log](https://github.com/feast-dev/feast/blob/master/CHANGELOG.md)

## 👋 Contributing

Feast thrives on community contributions!  Explore our guides:

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

A big thank you to our amazing contributors!

<a href="https://github.com/feast-dev/feast/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=feast-dev/feast" alt="Feast Contributors"/>
</a>