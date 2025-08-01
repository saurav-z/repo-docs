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

## Feast: The Open Source Feature Store for Machine Learning

Feast is an open-source feature store that streamlines the process of managing, serving, and discovering features for machine learning, making it easy to scale and productionize your analytic data.  [Explore the Feast GitHub Repository](https://github.com/feast-dev/feast).

## Key Features

*   **Consistent Feature Availability:** Manage features consistently for training and serving through offline and online stores, and a feature server.
*   **Prevent Data Leakage:** Generate point-in-time correct feature sets to ensure training data accuracy and prevent leakage.
*   **Decoupled Infrastructure:** Abstract feature storage from retrieval, making your models portable and flexible across different environments.
*   **Feature Engineering Support:** Enables On-demand and Streaming Transformations.
*   **Broad Support:** Supports various Data Sources and Offline/Online Stores, including Snowflake, Redshift, BigQuery, Redis, DynamoDB, and more.

## Architecture

The architecture diagram visualizes a minimal Feast deployment.

<img src="docs/assets/feast_marchitecture.png" alt="Feast Architecture" width="700">

Want to run the full Feast on Snowflake/GCP/AWS? Click [here](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws).

## Getting Started

### 1. Install Feast

```bash
pip install feast
```

### 2. Create a Feature Repository

```bash
feast init my_feature_repo
cd my_feature_repo/feature_repo
```

### 3. Register Feature Definitions and Set Up

```bash
feast apply
```

### 4. Explore the Web UI (Experimental)

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
```text
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

```text
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

*   **Natural Language Processing**
    *   [x] Vector Search (Alpha release)
    *   [ ] [Enhanced Feature Server and SDK for native support for NLP](https://github.com/feast-dev/feast/issues/4964)
*   **Data Sources**
    *   [x] Snowflake, Redshift, BigQuery, Parquet file, Azure Synapse + Azure SQL, Hive, Postgres, Spark, Couchbase, Kafka / Kinesis (via push)
*   **Offline Stores**
    *   [x] Snowflake, Redshift, BigQuery, Azure Synapse + Azure SQL, Hive, Postgres, Trino, Spark, Couchbase, In-memory / Pandas, Custom offline store support
*   **Online Stores**
    *   [x] Snowflake, DynamoDB, Redis, Datastore, Bigtable, SQLite, Dragonfly, IKV, Azure Cache for Redis, Postgres, Cassandra / AstraDB, ScyllaDB, Couchbase, Custom online store support
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
    *   [x] Python feature server, Feast Operator (alpha), Java feature server (alpha), Go feature server (alpha), Offline Feature Server (alpha), Registry server (alpha)
*   **Data Quality Management (See [RFC](https://docs.google.com/document/d/110F72d4NTv80p35wDSONxhhPBqWRwbZXG4f9mNEMd98/edit))**
    *   [x] Data profiling and validation (Great Expectations)
*   **Feature Discovery and Governance**
    *   [x] Python SDK for browsing feature registry
    *   [x] CLI for browsing feature registry
    *   [x] Model-centric feature tracking (feature services)
    *   [x] Amundsen integration
    *   [x] DataHub integration
    *   [x] Feast Web UI (Beta release)
    *   [ ] Feast Lineage Explorer

## Important Resources

*   [Documentation](https://docs.feast.dev/)
    *   [Quickstart](https://docs.feast.dev/getting-started/quickstart)
    *   [Tutorials](https://docs.feast.dev/tutorials/tutorials-overview)
    *   [Examples](https://github.com/feast-dev/feast/tree/master/examples)
    *   [Running Feast with Snowflake/GCP/AWS](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws)
    *   [Change Log](https://github.com/feast-dev/feast/blob/master/CHANGELOG.md)

## Contributing

Contribute to the Feast project by following our guidelines:
-   [Contribution Process for Feast](https://docs.feast.dev/project/contributing)
-   [Development Guide for Feast](https://docs.feast.dev/project/development-guide)
-   [Development Guide for the Main Feast Repository](./CONTRIBUTING.md)

## GitHub Star History

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

<a href="https://github.com/feast-dev/feast/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=feast-dev/feast" alt="Contributors" />
</a>
```
Key improvements and SEO considerations:

*   **Clear Title and Hook:** The title includes the primary keyword, and the one-sentence hook immediately describes the project's value.
*   **Keywords:** Uses relevant keywords like "feature store," "machine learning," "feature engineering," "online store," and specific database/cloud provider names throughout the document.
*   **Headings & Structure:**  Organized with clear, descriptive headings and subheadings for readability and SEO.  This helps with search engine indexing.
*   **Bullet Points:** The "Key Features" section uses bullet points for easy scanning and highlights key selling points.
*   **Concise Language:**  Uses straightforward language to explain complex concepts.
*   **Internal & External Links:** Includes links to the official documentation, getting started guides, and the GitHub repository itself to enhance the user experience.
*   **Alt Text:**  Added alt text to images for accessibility and SEO.
*   **Actionable examples**: The examples are clear and show how the core functionalities work.
*   **Contributor information**: Includes useful data about contributors.
*   **Roadmap/Functionality:** The "Functionality and Roadmap" section helps show the project's capabilities.
*   **GitHub Star History** Improves social proof and is great for SEO.