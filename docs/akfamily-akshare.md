# AKShare: Your Go-To Python Library for Financial Data 📊

**AKShare simplifies financial data retrieval, providing easy access to a wealth of information for analysis and research. ([Original Repo](https://github.com/akfamily/akshare))**

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

*   **Extensive Data Coverage:** Access a wide range of financial data from various sources.
*   **Ease of Use:** Simple, intuitive API for data retrieval with minimal code.
*   **Highly Customizable:** Easily integrate AKShare into your existing projects.
*   **Open Source & Free:** MIT License, allowing for commercial and personal use.
*   **Well-Documented:** Comprehensive documentation and tutorials to guide you.
*   **Active Community:** Benefit from an active community and ongoing development.

## Installation

### General

```shell
pip install akshare --upgrade
```

### China (For faster installation)

```shell
pip install akshare -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com  --upgrade
```

### Docker

#### Pull images

```shell
docker pull registry.cn-shanghai.aliyuncs.com/akfamily/aktools:jupyter
```

#### Run Container

```shell
docker run -it registry.cn-shanghai.aliyuncs.com/akfamily/aktools:jupyter python
```

#### Test

```python
import akshare as ak

print(ak.__version__)
```

## Usage

### Data Retrieval Example

Fetch historical stock data for a specific ticker:

```python
import akshare as ak

stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20170301", end_date='20231022', adjust="")
print(stock_zh_a_hist_df)
```

### Plotting Example

Visualize US stock data using `mplfinance`:

```python
import akshare as ak
import mplfinance as mpf  # Please install mplfinance as follows: pip install mplfinance

stock_us_daily_df = ak.stock_us_daily(symbol="AAPL", adjust="qfq")
stock_us_daily_df = stock_us_daily_df.set_index(["date"])
stock_us_daily_df = stock_us_daily_df["2020-04-01": "2020-04-29"]
mpf.plot(stock_us_daily_df, type="candle", mav=(3, 6, 9), volume=True, show_nontrading=False)
```
*   See a sample output image in the original readme.

## Tutorials & Documentation

Explore the comprehensive documentation and resources to get started:

1.  [Overview](https://akshare.akfamily.xyz/introduction.html)
2.  [Installation](https://akshare.akfamily.xyz/installation.html)
3.  [Tutorial](https://akshare.akfamily.xyz/tutorial.html)
4.  [Data Dict](https://akshare.akfamily.xyz/data/index.html)
5.  [Subjects](https://akshare.akfamily.xyz/topic/index.html)

## Contributing

Contribute to the project and help improve AKShare.  Check out the [Documentation](https://akshare.akfamily.xyz/contributing.html) for details.

*   Report or fix bugs
*   Require or publish interface
*   Write or fix documentation
*   Add test cases

>   *Code style is enforced using [Ruff](https://github.com/astral-sh/ruff).*

## Important Considerations

*   **Data Usage:** Data provided by AKShare is for academic research and reference only.
*   **Investment Advice:**  This library does not constitute investment advice.
*   **Data Accuracy:**  Be aware that data interfaces may be subject to change or removal due to external factors.

## Show Your Style

Add the AKShare badge to your project:

```markdown
[![Data: akshare](https://img.shields.io/badge/Data%20Science-AKShare-green)](https://github.com/akfamily/akshare)
```

## Citation

Cite AKShare in your research:

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

*   Special thanks to the contributors of FuShare, TuShare, and the numerous data sources used by AKShare (listed in original README).

## Backer and Sponsor

<a href="https://www.jetbrains.com/?from=albertandking/akshare" target="_blank">
<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jetbrains.png" alt="JetBrains logo.">
</a>