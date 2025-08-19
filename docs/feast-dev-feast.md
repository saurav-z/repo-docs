# Feast: The Open Source Feature Store for Machine Learning

**Feast** is an open-source feature store designed to streamline the management and serving of machine learning features, enabling faster and more reliable model development. [Explore the Feast Repository](https://github.com/feast-dev/feast)

<p align="center">
    <a href="https://feast.dev/">
      <img src="docs/assets/feast_logo.png" width="550">
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

## Why Use Feast?

Feast solves the complexities of managing features for machine learning, making it easier to build, deploy, and maintain ML models.

**Key Benefits:**

*   **Consistent Feature Availability:**  Provides a unified approach to serve features consistently for both model training (offline store) and real-time prediction (online store), along with a feature server.
*   **Data Leakage Prevention:**  Ensures point-in-time correct feature sets, allowing data scientists to focus on feature engineering.
*   **Decoupled Infrastructure:** Abstracts feature storage and retrieval, enabling portability across different data infrastructure systems.

## Architecture

![](docs/assets/feast_marchitecture.png)

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

### 3. Register Feature Definitions and Set Up the Feature Store

```bash
feast apply
```

### 4. Explore Your Data (Experimental Web UI)

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

### 6. Load Feature Values into Online Store

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

## Functionality and Roadmap

Feast is actively evolving with new features and integrations.  Here's what's planned:

*   **Natural Language Processing**
    *   [x] Vector Search (Alpha)
    *   [ ] Enhanced Feature Server and SDK for NLP
*   **(Numerous) Data Sources** - including Snowflake, Redshift, BigQuery, and more.
*   **(Numerous) Offline Stores** - Snowflake, Redshift, BigQuery, and more.
*   **(Numerous) Online Stores** - Snowflake, DynamoDB, Redis, and more.
*   **Feature Engineering**
    *   [x] On-demand Transformations (On Read) (Beta)
    *   [x] Streaming Transformations (Alpha)
    *   [ ] Batch transformation (In progress)
    *   [x] On-demand Transformations (On Write) (Beta)
*   **Streaming**
    *   [x] Custom streaming ingestion job support
    *   [x] Push based streaming data ingestion to online store
    *   [x] Push based streaming data ingestion to offline store
*   **Deployments**
    *   [x] AWS Lambda (Alpha)
    *   [x] Kubernetes
*   **Feature Serving**
    *   [x] Python Client
    *   [x] Python feature server
    *   [x] Feast Operator (alpha)
    *   [x] Java feature server (alpha)
    *   [x] Go feature server (alpha)
    *   [x] Offline Feature Server (alpha)
    *   [x] Registry server (alpha)
*   **Data Quality Management (See [RFC](https://docs.google.com/document/d/110F72d4NTv80p35wDSONxhhPBqWRwbZXG4f9mNEMd98/edit))**
    *   [x] Data profiling and validation (Great Expectations)
*   **Feature Discovery and Governance**
    *   [x] Python SDK for browsing feature registry
    *   [x] CLI for browsing feature registry
    *   [x] Model-centric feature tracking (feature services)
    *   [x] Amundsen integration
    *   [x] DataHub integration
    *   [x] Feast Web UI (Beta)
    *   [ ] Feast Lineage Explorer

## Important Resources

*   [Documentation](https://docs.feast.dev/)
    *   [Quickstart](https://docs.feast.dev/getting-started/quickstart)
    *   [Tutorials](https://docs.feast.dev/tutorials/tutorials-overview)
    *   [Examples](https://github.com/feast-dev/feast/tree/master/examples)
    *   [Running Feast with Snowflake/GCP/AWS](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws)
    *   [Change Log](https://github.com/feast-dev/feast/blob/master/CHANGELOG.md)

## Join the Community

*   👋👋👋 [Come say hi on Slack!](https://communityinviter.com/apps/feastopensource/feast-the-open-source-feature-store)
*   [Check out our DeepWiki!](https://deepwiki.com/feast-dev/feast)

## Contributing

Feast thrives on community contributions.  Check out these guides to get involved:

*   [Contribution Process](https://docs.feast.dev/project/contributing)
*   [Development Guide](https://docs.feast.dev/project/development-guide)
*   [Development Guide for the Main Feast Repository](./CONTRIBUTING.md)

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

A big thank you to all our contributors!

<a href="https://github.com/feast-dev/feast/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=feast-dev/feast" />
</a>
```
Key improvements and SEO optimizations:

*   **Concise Hook:**  Replaced the general overview with a single, strong sentence that grabs attention.
*   **Clear Headings:**  Used clear, keyword-rich headings (e.g., "Why Use Feast?", "Getting Started") to improve readability and SEO.
*   **Bulleted Lists:**  Used bulleted lists for key benefits and features, making them easier to scan and understand.
*   **Keyword Optimization:**  Strategically incorporated relevant keywords like "feature store," "machine learning," and specific feature names throughout the text.
*   **Stronger Call to Action:**  Included clear calls to action, like "Explore the Feast Repository."
*   **Simplified Getting Started:** Streamlined the "Getting Started" section with more concise steps.
*   **Comprehensive Roadmap:** Improved the "Functionality and Roadmap" section to show a more complete overview of what Feast offers
*   **Community Section:** Added a clear Community section with calls to action
*   **GitHub Star History:**  Added the Star History visualization, which improves SEO and user engagement.
*   **Contextual Links:**  Kept all existing links but improved the surrounding text for context, and improved internal links to other sections.