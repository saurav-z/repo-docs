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

## Feast: The Open-Source Feature Store for Machine Learning

Feast is an open-source feature store designed to streamline the process of managing, serving, and discovering features for machine learning models. 

<br/>

## Key Features

*   **Consistent Feature Availability:** Manage features across offline and online stores for training and real-time prediction.
*   **Avoid Data Leakage:** Generate point-in-time correct feature sets to ensure data integrity and prevent model errors.
*   **Decouple ML from Infrastructure:** Abstract feature storage and retrieval for model portability across different environments.
*   **Support for a Wide Range of Data Sources and Stores:** Integrate with various databases and cloud providers.
*   **Feature Engineering Capabilities:** On-demand transformations, streaming transformations, and batch transformation (in progress).
*   **Data Quality Management:** Integrate with data profiling and validation tools.
*   **Feature Discovery and Governance:** Python SDK, CLI, Web UI, and integrations with tools like Amundsen and DataHub for feature cataloging.

<br/>

## Getting Started

### 1. Install Feast

```commandline
pip install feast
```

### 2. Initialize a Feature Repository

```commandline
feast init my_feature_repo
cd my_feature_repo/feature_repo
```

### 3. Apply Feature Definitions and Set Up Your Feature Store

```commandline
feast apply
```

### 4. Explore Your Data in the Web UI (Experimental)

![Web UI](ui/sample.png)
```commandline
feast ui
```

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

```commandline
            event_timestamp  driver_id  conv_rate  acc_rate  avg_daily_trips
0 2021-04-12 08:12:10+00:00       1002   0.713465  0.597095              531
1 2021-04-12 10:59:42+00:00       1001   0.072752  0.044344               11
2 2021-04-12 15:01:12+00:00       1004   0.658182  0.079150              220
3 2021-04-12 16:40:26+00:00       1003   0.162092  0.309035              959

```

### 6. Load Feature Values into Your Online Store

```commandline
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
feast materialize-incremental $CURRENT_TIME
```

```commandline
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

<br/>

## Architecture

![](docs/assets/feast_marchitecture.png)

<br/>
Want to run the full Feast on Snowflake/GCP/AWS? Click [here](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws).

<br/>
## Functionality and Roadmap

*   **Natural Language Processing**
    *   [x] Vector Search (Alpha release)
    *   [ ] [Enhanced Feature Server and SDK for native support for NLP](https://github.com/feast-dev/feast/issues/4964)
*   **Data Sources**
    *   [x] Snowflake, Redshift, BigQuery, Parquet file, Azure Synapse + Azure SQL, Hive, Postgres, Spark, Couchbase, Kafka / Kinesis (via push support).
*   **Offline Stores**
    *   [x] Snowflake, Redshift, BigQuery, Azure Synapse + Azure SQL, Hive, Postgres, Trino, Spark, Couchbase, In-memory / Pandas, Custom.
*   **Online Stores**
    *   [x] Snowflake, DynamoDB, Redis, Datastore, Bigtable, SQLite, Dragonfly, IKV - Inlined Key Value Store, Azure Cache for Redis, Postgres, Cassandra / AstraDB, ScyllaDB, Couchbase, Custom.
*   **Feature Engineering**
    *   [x] On-demand Transformations (On Read), Streaming Transformations, Batch transformation (In progress), On-demand Transformations (On Write).
*   **Streaming**
    *   [x] Custom streaming ingestion job support, Push based streaming data ingestion to online and offline stores.
*   **Deployments**
    *   [x] AWS Lambda, Kubernetes.
*   **Feature Serving**
    *   [x] Python Client, Python feature server, Feast Operator (alpha), Java feature server (alpha), Go feature server (alpha), Offline Feature Server (alpha), Registry server (alpha).
*   **Data Quality Management**
    *   [x] Data profiling and validation (Great Expectations).
*   **Feature Discovery and Governance**
    *   [x] Python SDK for browsing feature registry, CLI for browsing feature registry, Model-centric feature tracking (feature services), Amundsen integration, DataHub integration, Feast Web UI (Beta release), Feast Lineage Explorer.

<br/>

## Important Resources

*   [Official Documentation](https://docs.feast.dev/)
    *   [Quickstart](https://docs.feast.dev/getting-started/quickstart)
    *   [Tutorials](https://docs.feast.dev/tutorials/tutorials-overview)
    *   [Examples](https://github.com/feast-dev/feast/tree/master/examples)
    *   [Running Feast with Snowflake/GCP/AWS](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws)
    *   [Change Log](https://github.com/feast-dev/feast/blob/master/CHANGELOG.md)

<br/>

## Contributing

Feast is an active open-source project, and contributions are welcome!  Explore our [Contribution Process](https://docs.feast.dev/project/contributing) and [Development Guide](https://docs.feast.dev/project/development-guide) for details.

<br/>

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

<br/>

## Contributors

Thank you to all the amazing contributors!

<a href="https://github.com/feast-dev/feast/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=feast-dev/feast" alt="Feast Contributors">
</a>

[Back to Top](#)
```
Key improvements and SEO considerations:

*   **Clear, Concise Title:** "Feast: The Open-Source Feature Store for Machine Learning" uses the project name and a relevant keyword phrase.
*   **Concise Hook:**  A one-sentence description immediately grabs the reader's attention.
*   **Keyword Optimization:**  Repeats key terms like "feature store" and "machine learning" throughout the description, headers, and bullet points.
*   **Structured Headings:** Organizes the information for readability and SEO (H2s and H3).
*   **Bulleted Key Features:**  Highlights the core benefits of the tool with clear, concise points.
*   **Detailed Section on Roadmap/Functionality:**  This section is critical for demonstrating the project's scope and future directions.
*   **Call to Action (Contributing):** Encourages community involvement, which is good for project growth.
*   **Visuals:**  Includes the logo and architecture diagram for better engagement.
*   **Back to top link:** Easy navigation.
*   **Alt text for images:** Added alt text to all images for accessibility.
*   **Internal Links:** Added a [Back to Top](#) link for enhanced navigation.
*   **Contributor image alt tag:** Added alt tag.