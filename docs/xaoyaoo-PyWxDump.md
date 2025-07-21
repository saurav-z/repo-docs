# PyWxDump: The Ultimate WeChat Data Extraction and Analysis Tool

**Unlock insights into your WeChat data with PyWxDump, a powerful Python tool for extracting, decrypting, and analyzing WeChat information. ([Original Repo](https://github.com/xaoyaoo/PyWxDump))**

[![Python](https://img.shields.io/badge/Python-3-blue.svg)](https://www.python.org/)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/xaoyaoo/pywxdump)](https://github.com/xaoyaoo/pywxdump)
[![GitHub all releases](https://img.shields.io/github/downloads/xaoyaoo/pywxdump/total)](https://github.com/xaoyaoo/PyWxDump)
[![GitHub stars](https://img.shields.io/github/stars/xaoyaoo/PyWxDump.svg)](https://github.com/xaoyaoo/PyWxDump)
[![GitHub forks](https://img.shields.io/github/forks/xaoyaoo/PyWxDump.svg)](https://github.com/xaoyaoo/PyWxDump/fork)
[![GitHub issues](https://img.shields.io/github/issues/xaoyaoo/PyWxDump)](https://github.com/xaoyaoo/PyWxDump/issues)
[![PyPI](https://img.shields.io/pypi/v/pywxdump)](https://pypi.org/project/pywxdump/)
[![Wheel](https://img.shields.io/pypi/wheel/pywxdump)](https://pypi.org/project/pywxdump/)
[![PyPI-Downloads](https://img.shields.io/pypi/dm/pywxdump)](https://pypistats.org/packages/pywxdump)
[![GitHub license](https://img.shields.io/pypi/l/pywxdump)](https://github.com/xaoyaoo/PyWxDump/blob/master/LICENSE)

## Key Features:

*   **Account Information Extraction:** Retrieve WeChat nickname, account, phone number, email, database keys, and more.
*   **Database Decryption:** Decrypt WeChat databases for comprehensive data access.
*   **Chat History Viewing:** View your chat history through a web interface.
*   **Data Export:** Export chat logs as HTML and CSV files for backup and analysis.
*   **Remote Viewing:** Access your WeChat history remotely (requires network configuration).
*   **Minimalist Version:**  Utilize `pywxdumpmini` for key and database location retrieval.
*   **Multi-Account Support:**  Handles multiple WeChat instances.

## I. Project Overview

### 1.1 Introduction

PyWxDump is a powerful tool built in Python designed to extract and analyze WeChat data.  It allows you to gain insights into your WeChat usage for various purposes, including backup, analysis, and more.

**Star the project if you find it useful!** [![Star](https://img.shields.io/github/stars/xaoyaoo/PyWxDump.svg?style=social&label=Star)](https://github.com/xaoyaoo/PyWxDump/)

### 1.2 Core Functionality

*   Get the base address offsets of WeChat account details.
*   Retrieve account information for the currently logged-in WeChat user.
*   Decrypt WeChat databases using encryption keys.
*   View data from multiple databases in a unified interface.

### 1.3 Extended Features

*   Web-based chat history viewing.
*   Export chat logs in HTML, CSV, for backup and analysis.
*   Remote viewing of chat history (requires network configuration).

### 1.4 Supporting Resources

*   Documentation on database field descriptions.
*   Methods for obtaining base address offsets.
*   Decryption methods for MAC databases.

### 1.5 Additional Features

*   **pywxdumpmini:** A simplified version providing database key and location retrieval.
*   **Multi-Instance Support:** Handle multiple WeChat instances simultaneously.

**Potential Use Cases:**

1.  Network security analysis.
2.  Daily backup and archiving.
3.  Remote chat history viewing.
4.  Data analysis and visualization.

### 1.6 Future Plans

*   Analyze chat logs to generate word clouds.
*   Generate daily and monthly chat volume charts.
*   Generate visualization reports.
*   Enhance support for enterprise WeChat.
*   Add support for viewing and backing up Moments (WeChat's social network).
*   Add features for cleaning up WeChat storage space.
*   Implement UI control for automated message sending.

### 1.7 Project Origins

PyWxDump is a refactored Python version of [SharpWxDump](https://github.com/AdminTest0/SharpWxDump).

## II. Usage Instructions

Detailed instructions are available in:
*   [UserGuide.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/UserGuide.md)
*   For the minimalist version: [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini)

To modify the UI, clone the [wx_dump_web](https://github.com/xaoyaoo/wxdump_web).

**Important Notes:**

*   Refer to [CE obtaining base address.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/CE获取基址.md) for information on obtaining the base address using Cheat Engine.  (This method is mainly for learning principles)
*   For database parsing details, see [wx database brief.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/wx数据库简述.md)

## III. Disclaimer (READ CAREFULLY!)

**(VERY IMPORTANT:  Please read and understand this section before using PyWxDump.)**

### 3.1 Intended Use

*   This project is for educational and communication purposes only.  **DO NOT use it for illegal activities.**
*   Users are responsible for their actions and any legal consequences arising from them.

### 3.2 Usage Duration

*   Delete the source code and compiled program within 24 hours of downloading and using it. Continued use beyond this period is not associated with the project or its developers.

### 3.3 Operation Regulations

*   Only use this project for authorized database backup and viewing.  Strictly prohibited for illegal purposes.
*   It is strictly prohibited to use this tool to steal others' privacy.
*   Secondary development is strictly prohibited.

### 3.4 Acceptance of Disclaimer

*   By downloading, saving, browsing the source code, or using this program, you agree to these terms.

### 3.5 Prohibitions

*   Do not use this project for illegal testing or penetration.
*   Users are responsible for any adverse consequences, including data leaks and privacy violations.

### 3.6 Disclaimer Modification

*   This disclaimer may be updated.  Check regularly for the latest version.

### 3.7 Other Considerations

*   Users must comply with all applicable laws and regulations.

## IV. Acknowledgements

Thanks to all contributors:

[![PyWxDump CONTRIBUTORS](https://contrib.rocks/image?repo=xaoyaoo/PyWxDump)](https://github.com/xaoyaoo/PyWxDump/graphs/contributors)

UI Contributors:

[![UI CONTRIBUTORS](https://contrib.rocks/image?repo=xaoyaoo/wxdump_web)](https://github.com/xaoyaoo/wxdump_web/graphs/contributors)

Additional Contributors:

[643104191](https://github.com/643104191)