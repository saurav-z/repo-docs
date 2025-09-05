<p align="center">
    <a href="https://feast.dev/">
      <img src="https://raw.githubusercontent.com/feast-dev/feast/master/docs/assets/feast_logo.png" width="550" alt="Feast Logo">
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

## Feast: The Open Source Feature Store

**Feast is an open-source feature store that streamlines the management and serving of machine learning features for training and real-time inference.** Simplify your ML workflow and boost model performance with Feast! Visit the [Feast GitHub repository](https://github.com/feast-dev/feast) to get started.

## Key Features

*   **Unified Feature Serving:**  Consistently serve features for both training and real-time prediction, leveraging offline and online stores with a feature server.
*   **Data Leakage Prevention:**  Generate point-in-time correct feature sets to prevent data leakage and ensure model accuracy.
*   **Decoupled Infrastructure:**  Abstract feature storage from retrieval, making your models portable across different data infrastructure systems.

## Architecture Overview

Feast's architecture provides a comprehensive solution for feature management.

![](https://raw.githubusercontent.com/feast-dev/feast/master/docs/assets/feast_marchitecture.png)

For detailed instructions on how to run the full Feast on Snowflake/GCP/AWS, see: [Feast on Snowflake/GCP/AWS](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws).

## Getting Started

Here's how to quickly get started with Feast:

### 1. Install Feast
```bash
pip install feast
```

### 2. Create a Feature Repository
```bash
feast init my_feature_repo
cd my_feature_repo/feature_repo
```

### 3. Register Features and Set Up Your Store
```bash
feast apply
```

### 4. Explore with the Web UI (Experimental)
```bash
feast ui
```
![Web UI](https://raw.githubusercontent.com/feast-dev/feast/master/ui/sample.png)

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
```bash
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

```bash
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

## Functionality and Roadmap

Feast is continuously evolving! Here's a look at the upcoming features:

*   **Natural Language Processing**
    *   \[x] Vector Search (Alpha release)
    *   \[ ] [Enhanced Feature Server and SDK for native support for NLP](https://github.com/feast-dev/feast/issues/4964)
*   **Data Sources**
    *   \[x] Snowflake, Redshift, BigQuery, Parquet file, Azure Synapse + Azure SQL, Hive, Postgres, Spark, Couchbase, Kafka / Kinesis
*   **Offline Stores**
    *   \[x] Snowflake, Redshift, BigQuery, Azure Synapse + Azure SQL, Hive, Postgres, Trino, Spark, Couchbase, In-memory / Pandas, Custom
*   **Online Stores**
    *   \[x] Snowflake, DynamoDB, Redis, Datastore, Bigtable, SQLite, Dragonfly, IKV, Azure Cache for Redis, Postgres, Cassandra / AstraDB, ScyllaDB, Couchbase, Custom
*   **Feature Engineering**
    *   \[x] On-demand Transformations (On Read) (Beta release)
    *   \[x] Streaming Transformations (Alpha release)
    *   \[ ] Batch transformation (In progress)
    *   \[x] On-demand Transformations (On Write) (Beta release)
*   **Streaming**
    *   \[x] Custom streaming ingestion job support
    *   \[x] Push based streaming data ingestion to online store
    *   \[x] Push based streaming data ingestion to offline store
*   **Deployments**
    *   \[x] AWS Lambda (Alpha release)
    *   \[x] Kubernetes
*   **Feature Serving**
    *   \[x] Python Client, Python feature server, Feast Operator, Java feature server, Go feature server, Offline Feature Server, Registry server
*   **Data Quality Management**
    *   \[x] Data profiling and validation (Great Expectations)
*   **Feature Discovery and Governance**
    *   \[x] Python SDK, CLI, Model-centric feature tracking, Amundsen integration, DataHub integration, Feast Web UI, Feast Lineage Explorer

## Important Resources

*   [Documentation](https://docs.feast.dev/)
*   [Quickstart](https://docs.feast.dev/getting-started/quickstart)
*   [Tutorials](https://docs.feast.dev/tutorials/tutorials-overview)
*   [Examples](https://github.com/feast-dev/feast/tree/master/examples)
*   [Running Feast with Snowflake/GCP/AWS](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws)
*   [Change Log](https://github.com/feast-dev/feast/blob/master/CHANGELOG.md)

## Contributing

We welcome contributions! Review the following guides:

*   [Contribution Process](https://docs.feast.dev/project/contributing)
*   [Development Guide](https://docs.feast.dev/project/development-guide)
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

Thank you to our amazing contributors!

<a href="https://github.com/feast-dev/feast/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=feast-dev/feast" alt="Contributors"/>
</a>