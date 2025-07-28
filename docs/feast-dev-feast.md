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

Feast (**Fea**ture **St**ore) is an open-source feature store that streamlines the management and serving of machine learning features for training and real-time inference, providing a single source of truth for your data.  [Learn more at the original repository](https://github.com/feast-dev/feast).

## Key Features of Feast

*   **Unified Feature Access:** Makes features consistently available for training and serving by managing offline and online stores.
*   **Data Leakage Prevention:** Ensures data scientists focus on feature engineering with point-in-time correct feature sets.
*   **Decoupled Infrastructure:** Provides a single data access layer, ensuring model portability across training and serving environments.

## Architecture Overview

Feast provides a comprehensive feature store solution.
[<img src="docs/assets/feast_marchitecture.png" alt="Feast Architecture" width="700">](https://docs.feast.dev/)

## Getting Started

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

## Roadmap & Functionality

*   **(NLP)** Vector Search
*   **(Data Sources)** Snowflake, Redshift, BigQuery, Parquet, Azure Synapse, Azure SQL, Hive, Postgres, Spark, Couchbase, Kafka / Kinesis
*   **(Offline Stores)** Snowflake, Redshift, BigQuery, Azure Synapse + Azure SQL, Hive, Postgres, Trino, Spark, Couchbase, In-memory / Pandas, Custom Support
*   **(Online Stores)** Snowflake, DynamoDB, Redis, Datastore, Bigtable, SQLite, Dragonfly, IKV, Azure Cache for Redis, Postgres, Cassandra / AstraDB, ScyllaDB, Couchbase, Custom Support
*   **(Feature Engineering)** On-demand Transformations (On Read & Write), Streaming Transformations, Batch Transformation
*   **(Streaming)** Custom Ingestion, Push to Online/Offline Store
*   **(Deployments)** AWS Lambda, Kubernetes
*   **(Feature Serving)** Python Client, Python Feature Server, Feast Operator, Java Feature Server, Go Feature Server, Offline Feature Server, Registry Server
*   **(Data Quality Management)** Data profiling and validation (Great Expectations)
*   **(Feature Discovery and Governance)** Python SDK & CLI, Model-centric tracking, Amundsen & DataHub integration, Feast Web UI, Feast Lineage Explorer

## Important Resources

*   [Documentation](https://docs.feast.dev/)
*   [Quickstart](https://docs.feast.dev/getting-started/quickstart)
*   [Tutorials](https://docs.feast.dev/tutorials/tutorials-overview)
*   [Examples](https://github.com/feast-dev/feast/tree/master/examples)
*   [Running Feast with Snowflake/GCP/AWS](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws)
*   [Change Log](https://github.com/feast-dev/feast/blob/master/CHANGELOG.md)

## Contribute

Feast is a community-driven project, and contributions are welcome; please review the [Contribution Process](https://docs.feast.dev/project/contributing) and the [Development Guide](https://docs.feast.dev/project/development-guide).

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

*   **Clear Title and Hook:**  The title is now a clear, SEO-friendly, and includes the primary keyword "Feature Store".  The opening sentence immediately tells the reader what Feast is and its core benefit.
*   **SEO-Friendly Headings:** Uses clear, concise headings (H2 and H3) to structure the content, making it easier for search engines to understand the page's topic.
*   **Keyword Optimization:**  Includes relevant keywords like "feature store," "machine learning," "real-time inference," and terms related to core functionalities in the headings and throughout the text.
*   **Bulleted Key Features:** Uses bullet points to highlight the most important features, making them easily scannable.
*   **Concise Summaries:** Provides brief descriptions of each feature.
*   **Added Alt Text to Images:** Added descriptive `alt` text to images, improving SEO and accessibility.
*   **Concise and Readable:** The text is rewritten to be more concise, improving readability.
*   **Call to Action:** Encourages users to learn more with a direct link to documentation.
*   **Added "Overview" section** Added an overview to provide an overview of the project.
*   **Contributor Section** Added a contributors section to highlight contributors.
*   **Star History Section** Added a Star History section to show GitHub star history.
*   **Updated Markdown** Corrected the Markdown code and corrected the format.