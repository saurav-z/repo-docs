# AKShare: Your Go-To Python Library for Financial Data Analysis

[AKShare](https://github.com/akfamily/akshare) provides a comprehensive set of financial data APIs, empowering you to easily access and analyze a wide range of market information.

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

*   **Extensive Data Coverage:** Access a broad range of financial data, including stock prices, futures data, economic indicators, and more.
*   **Ease of Use:** Simple and intuitive API allows you to fetch data with minimal code.
*   **Customization and Extensibility:** Easily integrate AKShare with your existing projects and customize it to meet your specific needs.
*   **Rich Python Ecosystem:** Leverages the power of the Python ecosystem for data analysis, visualization, and machine learning.

## Installation

### General

```shell
pip install akshare --upgrade
```

### China Mirror (for faster downloads in China)

```shell
pip install akshare -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com  --upgrade
```

## Usage Examples

### Fetch Historical Stock Data

```python
import akshare as ak

stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20170301", end_date='20231022', adjust="")
print(stock_zh_a_hist_df)
```

### Visualize Data with Candlestick Charts

```python
import akshare as ak
import mplfinance as mpf

stock_us_daily_df = ak.stock_us_daily(symbol="AAPL", adjust="qfq")
stock_us_daily_df = stock_us_daily_df.set_index(["date"])
stock_us_daily_df = stock_us_daily_df["2020-04-01": "2020-04-29"]
mpf.plot(stock_us_daily_df, type="candle", mav=(3, 6, 9), volume=True, show_nontrading=False)
```

## Tutorials and Documentation

*   [Overview](https://akshare.akfamily.xyz/introduction.html)
*   [Installation](https://akshare.akfamily.xyz/installation.html)
*   [Tutorial](https://akshare.akfamily.xyz/tutorial.html)
*   [Data Dict](https://akshare.akfamily.xyz/data/index.html)
*   [Subjects](https://akshare.akfamily.xyz/topic/index.html)

## Contribution

AKShare is an open-source project, and we welcome contributions from the community!  Please see the [Documentation](https://akshare.akfamily.xyz/contributing.html) for details on how to contribute.

*   Report or fix bugs
*   Require or publish interface
*   Write or fix documentation
*   Add test cases

> Code style is enforced using [Ruff](https://github.com/astral-sh/ruff).

## Important Statements

*   All data provided by AKShare is for academic research purposes only.
*   The data is for reference only and does not constitute any investment advice.
*   Users should be aware of data risks when making investment decisions.
*   AKShare is committed to providing open-source financial data.
*   Some data interfaces may be removed due to uncontrollable factors.
*   Please adhere to the relevant open-source protocols.
*   For users of other programming languages, an HTTP API is available: [AKTools](https://aktools.readthedocs.io/).

## Show Your Style

Add the AKShare badge to your project's README:

```markdown
[![Data: akshare](https://img.shields.io/badge/Data%20Science-AKShare-green)](https://github.com/akfamily/akshare)
```

## Citation

If you use AKShare in your publications, please cite it using the following bibtex:

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

Special thanks to the many projects and data providers who have contributed to AKShare, including (but not limited to):

*   FuShare ([https://github.com/LowinLi/fushare](https://github.com/LowinLi/fushare))
*   TuShare ([https://github.com/waditu/tushare](https://github.com/waditu/tushare))
*   and many more. (See the original README for a full list)

## Backer and Sponsor

<a href="https://www.jetbrains.com/?from=albertandking/akshare" target="_blank">
<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jetbrains.png" alt="JetBrains logo.">
</a>
```

Key improvements and summary of changes:

*   **SEO Optimization:** Added relevant keywords like "financial data," "Python library," "stock prices," "economic indicators," and "data analysis" throughout the README.
*   **Clear Structure with Headings:** Organized the content with clear and concise headings (e.g., Key Features, Installation, Usage Examples, Contribution, etc.) to improve readability and scannability.
*   **Concise Summary/Hook:** Added a strong introductory sentence to immediately convey the purpose of the library.
*   **Bulleted Key Features:**  Presented key features in a bulleted list for easy understanding.
*   **Concise Usage Examples:** Kept the usage examples short and to the point to showcase the simplicity of the library.
*   **Simplified Installation Instructions:**  Improved the installation steps.
*   **Contribution Section:** Clearly outlined contribution guidelines and linked to the contributing documentation.
*   **Emphasis on Open Source:** Highlighted the open-source nature of the project.
*   **Clearer Statements and Acknowledgements:** Enhanced the "Important Statements" and "Acknowledgements" sections for better understanding.
*   **Show Your Style Section:** added "show your style" section, for users to show support.
*   **Badge Integration:** Integrated all the badges for easy reference.
*   **Removed redundant information:** Streamlined the content to focus on the most important aspects of the library.
*   **Formatted for Markdown:** Ensured proper Markdown formatting for a clean and readable display on GitHub.