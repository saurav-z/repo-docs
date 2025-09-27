# AKShare: Your Open-Source Toolkit for Financial Data in Python

**AKShare simplifies access to a wide range of financial data, empowering quantitative researchers, analysts, and data enthusiasts with free and open-source tools.** ([View on GitHub](https://github.com/akfamily/akshare))

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

*   **Comprehensive Data Coverage:** Access a vast collection of financial data, including stock prices, futures, economic indicators, and more.
*   **Easy-to-Use API:** Simplify data retrieval with a clean and intuitive Python API.
*   **Open Source and Free:** Leverage a community-driven project, built for and by data enthusiasts.
*   **Extensible:** Integrate AKShare data seamlessly into your existing workflows and applications.
*   **Well-Documented:** Comprehensive documentation with tutorials and examples to help you get started quickly.
*   **Python Ecosystem Integration:** Leverage the power of the Python ecosystem, including libraries like pandas and matplotlib, for data analysis and visualization.

## Installation

### General

```shell
pip install akshare --upgrade
```

### China (if experiencing issues with the standard install)

```shell
pip install akshare -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com  --upgrade
```

## Usage Examples

### Fetching Stock Data

```python
import akshare as ak

stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20170301", end_date='20231022', adjust="")
print(stock_zh_a_hist_df)
```

### Plotting Stock Data

```python
import akshare as ak
import mplfinance as mpf  # Please install mplfinance: pip install mplfinance

stock_us_daily_df = ak.stock_us_daily(symbol="AAPL", adjust="qfq")
stock_us_daily_df = stock_us_daily_df.set_index(["date"])
stock_us_daily_df = stock_us_daily_df["2020-04-01": "2020-04-29"]
mpf.plot(stock_us_daily_df, type="candle", mav=(3, 6, 9), volume=True, show_nontrading=False)
```

## Tutorials & Documentation

*   [Overview](https://akshare.akfamily.xyz/introduction.html)
*   [Installation](https://akshare.akfamily.xyz/installation.html)
*   [Tutorial](https://akshare.akfamily.xyz/tutorial.html)
*   [Data Dict](https://akshare.akfamily.xyz/data/index.html)
*   [Subjects](https://akshare.akfamily.xyz/topic/index.html)

## Contributing

AKShare is a community-driven project, and contributions are welcome!  Whether you're fixing bugs, adding new features, or improving documentation, your help is valuable.

*   Report or fix bugs
*   Require or publish interface
*   Write or fix documentation
*   Add test cases

> Notice: We use [Ruff](https://github.com/astral-sh/ruff) to format the code

## Citation

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

(List of Acknowledgements from original README)

## Backer and Sponsor

<a href="https://www.jetbrains.com/?from=albertandking/akshare" target="_blank">
<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jetbrains.png" alt="JetBrains logo.">
</a>