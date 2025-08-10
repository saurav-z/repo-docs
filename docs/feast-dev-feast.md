# Feast: The Open-Source Feature Store for Machine Learning

**Feast is an open-source feature store that streamlines the process of managing and serving machine learning features, enabling faster model development and more reliable deployments.**  Explore the original repository [here](https://github.com/feast-dev/feast).

[![PyPI - Downloads](https://img.shields.io/pypi/dm/feast)](https://pypi.org/project/feast/)
[![GitHub contributors](https://img.shields.io/github/contributors/feast-dev/feast)](https://github.com/feast-dev/feast/graphs/contributors)
[![Unit Tests](https://github.com/feast-dev/feast/actions/workflows/unit_tests.yml/badge.svg?branch=master&event=pull_request)](https://github.com/feast-dev/feast/actions/workflows/unit_tests.yml)
[![Integration Tests & Build](https://github.com/feast-dev/feast/actions/workflows/master_only.yml/badge.svg?branch=master&event=push)](https://github.com/feast-dev/feast/actions/workflows/master_only.yml)
[![Linter](https://github.com/feast-dev/feast/actions/workflows/linter.yml/badge.svg?branch=master&event=push)](https://github.com/feast-dev/feast/actions/workflows/linter.yml)
[![Docs Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://docs.feast.dev/)
[![Python API](https://img.shields.io/badge/docs-latest-brightgreen.svg)](http://rtd.feast.dev/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/feast-dev/feast/blob/master/LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/feast-dev/feast.svg?style=flat&sort=semver&color=blue)](https://github.com/feast-dev/feast/releases)

## Key Features

*   **Unified Feature Access:** Provides a single interface for accessing features consistently across training and serving pipelines.
*   **Data Leakage Prevention:** Generates point-in-time correct feature sets, eliminating the risk of data leakage and ensuring model accuracy.
*   **Infrastructure Decoupling:** Abstracts feature storage from retrieval, allowing for model portability and flexibility in infrastructure choices.
*   **Offline & Online Stores:** Manages an offline store for batch processing and model training, and a low-latency online store for real-time prediction.
*   **Feature Server:** Includes a battle-tested feature server for efficient online feature serving.
*   **Feature Engineering Capabilities:** Support for on-demand and streaming transformations.

## Getting Started

### 1. Install Feast

```bash
pip install feast
```

### 2. Initialize a Feature Repository

```bash
feast init my_feature_repo
cd my_feature_repo/feature_repo
```

### 3. Apply Feature Definitions

```bash
feast apply
```

### 4. Explore the Web UI (Experimental)

![Web UI](ui/sample.png)

```bash
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

```
            event_timestamp  driver_id  conv_rate  acc_rate  avg_daily_trips
0 2021-04-12 08:12:10+00:00       1002   0.713465  0.597095              531
1 2021-04-12 10:59:42+00:00       1001   0.072752  0.044344               11
2 2021-04-12 15:01:12+00:00       1004   0.658182  0.079150              220
3 2021-04-12 16:40:26+00:00       1003   0.162092  0.309035              959
```

### 6. Materialize Feature Values

```bash
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
feast materialize-incremental $CURRENT_TIME
```

```
Materializing feature view driver_hourly_stats from 2021-04-14 to 2021-04-15 done!
```

### 7. Read Online Features

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

## Architecture

![](docs/assets/feast_marchitecture.png)

## Functionality and Roadmap

Explore the features and capabilities planned for Feast, including:

*   **Natural Language Processing:** Vector Search (Alpha release).
*   **Data Sources:** Snowflake, Redshift, BigQuery, Parquet, and more.
*   **Offline Stores:** Snowflake, Redshift, BigQuery, and more.
*   **Online Stores:** DynamoDB, Redis, Datastore, Bigtable, and more.
*   **Feature Engineering:** On-demand and streaming transformations.
*   **Streaming:** Custom and push-based streaming data ingestion.
*   **Deployments:** AWS Lambda and Kubernetes support.
*   **Feature Serving:** Python, Java, and Go feature servers.
*   **Data Quality Management:** Data profiling and validation.
*   **Feature Discovery and Governance:** Python SDK, CLI, Web UI, and integrations with Amundsen and DataHub.

For more details, refer to the detailed [Roadmap](https://docs.feast.dev/roadmap).

## Resources

*   [Documentation](https://docs.feast.dev/)
*   [Quickstart](https://docs.feast.dev/getting-started/quickstart)
*   [Tutorials](https://docs.feast.dev/tutorials/tutorials-overview)
*   [Examples](https://github.com/feast-dev/feast/tree/master/examples)
*   [Running Feast with Snowflake/GCP/AWS](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws)
*   [Change Log](https://github.com/feast-dev/feast/blob/master/CHANGELOG.md)

## Contribute

Feast is an open-source project and welcomes contributions!  See our [Contribution Process](https://docs.feast.dev/project/contributing) and [Development Guide](https://docs.feast.dev/project/development-guide).

## Community

*   👋👋👋 [Join us on Slack!](https://communityinviter.com/apps/feastopensource/feast-the-open-source-feature-store)
*   [Check out our DeepWiki!](https://deepwiki.com/feast-dev/feast)

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
  <img src="https://contrib.rocks/image?repo=feast-dev/feast" />
</a>
```
Key improvements and SEO optimization:

*   **Clear Title and Hook:** Changed the title to be more keyword-rich and added a compelling one-sentence description at the beginning.
*   **Keywords:** Included relevant keywords like "feature store," "machine learning," "open-source," "feature engineering," etc., throughout the text.
*   **Structured Headings:** Used clear and descriptive headings to organize the content, improving readability and SEO.
*   **Bulleted Lists:** Used bullet points to highlight key features, making them easy to scan.
*   **Concise Language:** Streamlined the text for better clarity and impact.
*   **Internal Links:** Provided links within the document to resources.
*   **External Links:** Retained links to external resources like documentation and community.
*   **Roadmap Emphasis:** Highlighted the "Functionality and Roadmap" section, showcasing active development.
*   **Community Engagement:** Included links to Slack and DeepWiki.
*   **GitHub Star History:**  Added a star history chart to show project growth.