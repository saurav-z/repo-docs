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

## Simplify Machine Learning Feature Management with Feast

Feast is an open-source feature store designed to streamline the management and serving of features for machine learning, accelerating your path from data to production.  Access the [original repository here](https://github.com/feast-dev/feast).

## Key Features

*   **Unified Feature Storage:** Manage features consistently across offline and online stores, ensuring reliable data for training and real-time prediction.
*   **Data Leakage Prevention:** Generate point-in-time correct feature sets to avoid data leakage, allowing data scientists to focus on model development.
*   **Infrastructure Decoupling:** Abstract feature storage from retrieval, ensuring model portability and flexibility across various data infrastructures.
*   **Offline Store:** Process historical data for scale-out batch scoring and model training.
*   **Online Store:** Power real-time prediction with low-latency feature access.
*   **Feature Server:** Serve pre-computed features online for fast and efficient retrieval.

## Architecture Overview

![Feast Architecture](https://raw.githubusercontent.com/feast-dev/feast/master/docs/assets/feast_marchitecture.png)

## Getting Started

### 1. Install Feast
```bash
pip install feast
```

### 2. Create a feature repository
```bash
feast init my_feature_repo
cd my_feature_repo/feature_repo
```

### 3. Register your feature definitions and set up your feature store
```bash
feast apply
```

### 4. Explore your data in the web UI (experimental)

![Web UI Sample](https://raw.githubusercontent.com/feast-dev/feast/master/ui/sample.png)
```bash
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
```bash
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

## 💻 Functionality and Roadmap

* **Natural Language Processing**
  * [x] Vector Search
  * [ ] Enhanced Feature Server and SDK for native support for NLP
* **Data Sources**
  * [x] Snowflake, Redshift, BigQuery, Parquet, Azure Synapse + Azure SQL, Hive, Postgres, Spark, Couchbase, Kafka / Kinesis sources
* **Offline Stores**
  * [x] Snowflake, Redshift, BigQuery, Azure Synapse + Azure SQL, Hive, Postgres, Trino, Spark, Couchbase, In-memory / Pandas, Custom offline store support
* **Online Stores**
  * [x] Snowflake, DynamoDB, Redis, Datastore, Bigtable, SQLite, Dragonfly, IKV - Inlined Key Value Store, Azure Cache for Redis, Postgres, Cassandra / AstraDB, ScyllaDB, Couchbase, Custom online store support
* **Feature Engineering**
  * [x] On-demand Transformations (On Read) (Beta release.)
  * [x] Streaming Transformations (Alpha release)
  * [ ] Batch transformation (In progress)
  * [x] On-demand Transformations (On Write) (Beta release)
* **Streaming**
  * [x] Custom streaming ingestion job support
  * [x] Push based streaming data ingestion to online store
  * [x] Push based streaming data ingestion to offline store
* **Deployments**
  * [x] AWS Lambda (Alpha release)
  * [x] Kubernetes
* **Feature Serving**
  * [x] Python Client
  * [x] Python feature server
  * [x] Feast Operator (alpha)
  * [x] Java feature server (alpha)
  * [x] Go feature server (alpha)
  * [x] Offline Feature Server (alpha)
  * [x] Registry server (alpha)
* **Data Quality Management**
  * [x] Data profiling and validation (Great Expectations)
* **Feature Discovery and Governance**
  * [x] Python SDK for browsing feature registry
  * [x] CLI for browsing feature registry
  * [x] Model-centric feature tracking (feature services)
  * [x] Amundsen integration
  * [x] DataHub integration
  * [x] Feast Web UI (Beta release)
  * [ ] Feast Lineage Explorer

## 📚 Important Resources

*   [Documentation](https://docs.feast.dev/)
    *   [Quickstart](https://docs.feast.dev/getting-started/quickstart)
    *   [Tutorials](https://docs.feast.dev/tutorials/tutorials-overview)
    *   [Examples](https://github.com/feast-dev/feast/tree/master/examples)
    *   [Running Feast with Snowflake/GCP/AWS](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws)
    *   [Change Log](https://github.com/feast-dev/feast/blob/master/CHANGELOG.md)

## 🤝 Contributing

Feast is a community project and welcomes contributions.  Explore the [Contribution Process](https://docs.feast.dev/project/contributing) and [Development Guide](https://docs.feast.dev/project/development-guide) for more information.

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

Thank you to all our contributors!

<a href="https://github.com/feast-dev/feast/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=feast-dev/feast" />
</a>
```
Key improvements and SEO considerations:

*   **Clear Title & Hook:**  "Simplify Machine Learning Feature Management with Feast" is a strong title that immediately conveys value. The first sentence acts as a compelling hook.
*   **Keyword Optimization:** The text incorporates relevant keywords like "feature store," "machine learning," "feature management," "online store," and "offline store" throughout the descriptions and headings.
*   **Concise Bullet Points:**  Key features are presented in easy-to-scan bullet points.
*   **Structured Headings:** Uses clear headings (e.g., "Key Features," "Architecture Overview," "Getting Started," "Functionality and Roadmap," etc.) for better readability and SEO.
*   **Call to Action:**  Includes a call to action to install and start using Feast.
*   **Comprehensive Roadmap:** The "Functionality and Roadmap" section provides valuable information about the project's future, and shows the project is actively developed.
*   **Contributor Section:** Thanks contributors, with an automatically generated contributor graph.
*   **Strong URLs:** Makes use of relevant URLs for the Feast docs, and uses descriptive anchor text.
*   **Use of Image Alt Text:** Added `alt` text to images to improve accessibility and SEO.
*   **Improved formatting:** Uses consistent formatting throughout, and fixes the code block issues in the original
*   **Simplified the "getting started" section:** Improved readability by removing irrelevant links, added code block labels.
*   **Added Links:** Added links to important resources like the documentation, quickstart guide, and tutorials.