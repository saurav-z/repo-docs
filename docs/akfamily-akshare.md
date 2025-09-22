# AKShare: Your One-Stop Solution for Financial Data in Python

**AKShare is a comprehensive Python library designed to simplify the process of fetching financial data, making it an invaluable tool for data scientists, researchers, and finance professionals.**  [Visit the original repo](https://github.com/akfamily/akshare)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/akshare.svg)](https://pypi.org/project/akshare/)
[![PyPI](https://img.shields.io/pypi/v/akshare.svg)](https://pypi.org/project/akshare/)
[![Downloads](https://pepy.tech/badge/akshare)](https://pepy.tech/project/akshare)
[![Documentation Status](https://readthedocs.org/projects/akshare/badge/?version=latest)](https://akshare.readthedocs.io/?badge=latest)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![akshare](https://img.shields.io/badge/Data%20Science-AKShare-green)](https://github.com/akfamily/akshare)
[![Actions Status](https://github.com/akfamily/akshare/actions/workflows/release_and_deploy.yml/badge.svg)](https://github.com/akfamily/akshare/actions)
[![MIT Licence](https://img.shields.io/badge/license-MIT-blue)](https://github.com/akfamily/akshare/blob/main/LICENSE)
[![](https://img.shields.io/github/forks/jindaxiang/akshare)](https://github.com/akfamily/akshare)
[![](https://img.shields.io/github/stars/jindaxiang/akshare)](https://github.com/akfamily/akshare)
[![](https://img.shields.io/github/issues/jindaxiang/akshare)](https://github.com/akfamily/akshare)
[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://github.com/prettier/prettier)

## Key Features

*   **Simplified Data Access:**  Fetch financial data with a single line of code.
*   **Wide Range of Data Sources:** Access data from numerous sources.
*   **Ease of Use:** Designed for both beginners and experienced users.
*   **Extensible:** Easily integrates with other applications and tools.
*   **Powerful Ecosystem:** Leverages the power of the Python ecosystem.
*   **Comprehensive Documentation:**  Detailed documentation and tutorials available.
*   **Active Community:**  Join a community of developers and researchers.

## Installation

### General

```shell
pip install akshare --upgrade
```

### China (Specific Mirror)

```shell
pip install akshare -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com  --upgrade
```

## Quick Start

### Example: Fetching Historical Stock Data

```python
import akshare as ak

stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20170301", end_date='20231022', adjust="")
print(stock_zh_a_hist_df)
```

### Example: Plotting Stock Data with *mplfinance*

```python
import akshare as ak
import mplfinance as mpf  # Please install mplfinance as follows: pip install mplfinance

stock_us_daily_df = ak.stock_us_daily(symbol="AAPL", adjust="qfq")
stock_us_daily_df = stock_us_daily_df.set_index(["date"])
stock_us_daily_df = stock_us_daily_df["2020-04-01": "2020-04-29"]
mpf.plot(stock_us_daily_df, type="candle", mav=(3, 6, 9), volume=True, show_nontrading=False)
```

## Resources

### Tutorials

1.  [Overview](https://akshare.akfamily.xyz/introduction.html)
2.  [Installation](https://akshare.akfamily.xyz/installation.html)
3.  [Tutorial](https://akshare.akfamily.xyz/tutorial.html)
4.  [Data Dict](https://akshare.akfamily.xyz/data/index.html)
5.  [Subjects](https://akshare.akfamily.xyz/topic/index.html)

## Contribution

[AKShare](https://github.com/akfamily/akshare) thrives on community contributions.  We welcome your help in:

*   Reporting or fixing bugs.
*   Requesting or publishing new interfaces.
*   Writing or improving documentation.
*   Adding test cases.

>   *Note: Code is formatted using [Ruff](https://github.com/astral-sh/ruff).*

## Important Disclaimers

1.  **Research Purpose Only:** All data provided is for academic research purposes.
2.  **Reference Only:** Data is for reference and does not constitute investment advice.
3.  **Data Risk:** Investors should be aware of data risks.
4.  **Open Source Commitment:**  We are committed to providing open-source financial data.
5.  **Data Interface Changes:**  Some data interfaces may be removed due to uncontrollable factors.
6.  **License Compliance:**  Please adhere to the relevant open-source protocol.
7.  **HTTP API:**  Use the [AKTools](https://aktools.readthedocs.io/) for other programming languages.

## Show Your Support

Add the AKShare badge to your project:

```markdown
[![Data: akshare](https://img.shields.io/badge/Data%20Science-AKShare-green)](https://github.com/akfamily/akshare)
```

## Citation

If you use AKShare in your publications, please use the following bibtex:

```markdown
@misc{akshare,
    author = {Albert King and Yaojie Zhang},
    title = {AKShare},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/akfamily/akshare}},
}
```

## Acknowledgements

(List of acknowledgements)

## Backers and Sponsors

<a href="https://www.jetbrains.com/?from=albertandking/akshare" target="_blank">
<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jetbrains.png" alt="JetBrains logo.">
</a>