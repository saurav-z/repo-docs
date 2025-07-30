# PyWxDump: Your Toolkit for WeChat Data Exploration

> PyWxDump empowers you to access, decrypt, and analyze your WeChat data, providing valuable insights into your conversations and contacts. **[Explore PyWxDump on GitHub](https://github.com/xaoyaoo/PyWxDump)**

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

## Key Features

*   **Account Information Extraction:** Retrieve WeChat account information (nickname, account, phone, email, and database keys).
*   **Database Decryption:** Decrypt WeChat databases for accessible data.
*   **Chat History Viewing:** Access and review your WeChat chat history.
*   **Data Export:** Export chat logs as HTML, CSV for backups and analysis.
*   **Web-Based Viewing:** View chat history through a web interface.
*   **Multi-User Support:** Obtain information from multiple WeChat accounts.

## Project Introduction

### 1. Core Functionality

*   Get the base address offset of WeChat information.
*   Retrieve key WeChat account details (nickname, account, phone number, email, key, original ID, and folder path).
*   Decrypt WeChat databases.
*   Merge multiple database types for unified viewing.

### 2. Extended Capabilities

*   Web-based chat history viewer.
*   Export chat logs in HTML and CSV formats.
*   Supports remote viewing of WeChat chat history.

### 3. Supporting Documentation

*   Database field descriptions.
*   Methods for obtaining base address offsets using Cheat Engine.
*   MAC database decryption methods.

### 4. Other Features

*   Minimalist version: [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini), for obtaining database keys and locations.
*   Supports multiple WeChat instances.

**Potential Use Cases:**

*   Network security research.
*   Personal data archiving.
*   Remote chat history access.

### 5. Future Development

*   Analyze chat logs for word clouds.
*   Analyze the number of chats per person per day and generate a line chart.
*   Analyze the monthly and annual chat volume of different people and generate a line chart.
*   Generate annual visualization reports.
*   Increase support for enterprise WeChat.
*   Viewing and backing up of the circle of friends.
*   Clean up WeChat storage space and reduce the space occupied by WeChat
*   Automatically send messages to specified people through UI control

## Instructions for Use

Detailed instructions are available in the [UserGuide.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/UserGuide.md) file.
A minimalist version is also available in the [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini) repository.

If you wish to customize the UI, you can clone the [wxdump_web](https://github.com/xaoyaoo/wxdump_web) repository.

**Important Notes:**

*   For Cheat Engine usage, see [CE obtaining base address.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/CE获取基址.md).
*   For database parsing, refer to [wx database brief.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/wx数据库简述.md).

## Disclaimer (READ CAREFULLY)

*   **Purpose:** This project is solely for educational and communication purposes.  **Do not use it for illegal activities.**
*   **Data Handling:**  Only back up and view databases under your own authorization.
*   **Responsibility:**  You are responsible for your use of this software and must adhere to all applicable laws and regulations.  The developers are not liable for misuse.
*   **Deletion:**  Delete the source code and compiled program within 24 hours of downloading/using.
*   **Prohibitions:**  Unauthorized use, secondary development, or using this project for unauthorized testing or penetration is strictly prohibited.

## Acknowledgements

Project Contributors:

[![PyWxDump CONTRIBUTORS](https://contrib.rocks/image?repo=xaoyaoo/PyWxDump)](https://github.com/xaoyaoo/PyWxDump/graphs/contributors)

UI Contributors:

[![UI CONTRIBUTORS](https://contrib.rocks/image?repo=xaoyaoo/wxdump_web)](https://github.com/xaoyaoo/wxdump_web/graphs/contributors)

Other Contributors:

[643104191](https://github.com/643104191)

**Star History:**

<details>
<summary>click to expand</summary>

[![Star History Chart](https://api.star-history.com/svg?repos=xaoyaoo/pywxdump&type=Date)](https://star-history.com/#xaoyaoo/pywxdump&Date)

</details>

---