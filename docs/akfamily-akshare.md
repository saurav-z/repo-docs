# AKShare: Open-Source Financial Data Library for Python

**AKShare is a powerful Python library that simplifies the process of accessing and analyzing financial data from various sources, empowering data scientists and financial analysts.** [Explore AKShare on GitHub](https://github.com/akfamily/akshare)

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

*   **Comprehensive Data Coverage:** Access a wide range of financial data, including stock prices, market indicators, economic data, and more.
*   **Ease of Use:** Simplify data fetching with intuitive and straightforward Python functions.
*   **Extensible:** Easily integrate AKShare with your existing projects and customize it to your specific needs.
*   **Well-Documented:** Benefit from comprehensive documentation and tutorials to get you started quickly.
*   **Open Source and Free:** Utilize a completely free, open-source library for your financial data analysis needs.

## Installation

### General

```shell
pip install akshare --upgrade
```

### China (If you have issues with the general installation)

```shell
pip install akshare -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com  --upgrade
```

## Usage Examples

### Data Retrieval

```python
import akshare as ak

stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20170301", end_date='20231022', adjust="")
print(stock_zh_a_hist_df)
```

### Data Visualization

```python
import akshare as ak
import mplfinance as mpf  # Please install mplfinance as follows: pip install mplfinance

stock_us_daily_df = ak.stock_us_daily(symbol="AAPL", adjust="qfq")
stock_us_daily_df = stock_us_daily_df.set_index(["date"])
stock_us_daily_df = stock_us_daily_df["2020-04-01": "2020-04-29"]
mpf.plot(stock_us_daily_df, type="candle", mav=(3, 6, 9), volume=True, show_nontrading=False)
```

## Resources & Documentation

*   [Documentation](https://akshare.akfamily.xyz/)
*   [Tutorial](https://akshare.akfamily.xyz/tutorial.html)
*   [Data Dictionary](https://akshare.akfamily.xyz/data/index.html)

## Contributing

AKShare welcomes contributions from the community. If you are interested in contributing, please review the [Contribution Guidelines](https://akshare.akfamily.xyz/contributing.html).

## Show Your Support

Use the AKShare badge in your project to show your support:

```markdown
[![Data: akshare](https://img.shields.io/badge/Data%20Science-AKShare-green)](https://github.com/akfamily/akshare)
```

## Citation

Cite AKShare in your publications using the following BibTeX entry:

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

(The acknowledgements section remains the same)

## Backer and Sponsor

<a href="https://www.jetbrains.com/?from=albertandking/akshare" target="_blank">
<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jetbrains.png" alt="JetBrains logo.">
</a>