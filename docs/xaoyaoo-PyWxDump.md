# PyWxDump: Your Comprehensive Tool for WeChat Data Extraction and Analysis

**Effortlessly extract, decrypt, and analyze your WeChat data with PyWxDump, a powerful Python-based tool.**  [View the original repository on GitHub](https://github.com/xaoyaoo/PyWxDump).

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

*   **Data Extraction:** Retrieve WeChat account information, including nicknames, accounts, phone numbers, emails, and database keys.
*   **Database Decryption:** Decrypt WeChat databases for data access.
*   **Chat History Viewing:** View your WeChat chat history through a web interface.
*   **Data Export:** Export chat logs as HTML and CSV for backups and analysis.
*   **Remote Access:**  Access your WeChat chat history remotely.
*   **Minimalist Version:** Provides a simplified version (pywxdumpmini) for key and database location retrieval.
*   **Multi-Account Support:**  Handles multiple WeChat instances.

## Project Introduction

PyWxDump is a powerful tool for extracting and analyzing data from WeChat.

### Core Capabilities:

*   Retrieve key WeChat account details.
*   Obtain the base address offset of essential data.
*   Decrypt WeChat databases for detailed access.
*   Consolidate data from multiple database types.

### Extended Functionality:

*   Web-based chat history viewer.
*   Export chat logs in HTML and CSV formats.
*   Remote viewing of chat history (network access required).

### Additional Resources:

*   Detailed database field descriptions are available.
*   Method for obtaining the base address offset using CE (Cheat Engine) is provided.
*   Decryption method for MAC databases.

### Other Features:

*   Minimalist version `pywxdumpmini`.
*   Supports multiple WeChat instances.

## Update Plans

*   Analyze chat logs and generate word clouds.
*   Generate line charts for daily/monthly/annual chat volume analysis.
*   Create annual visualization reports.
*   Increase support for enterprise WeChat.
*   Support viewing and backing up of Moments.
*   Clean up WeChat storage space by clearing media files selectively.
*   Automatically send messages to specified people.

## Important Notes

*   **Disclaimer:** This project is for learning and communication purposes only. Please review the full disclaimer below before use.
*   **Use Cases:** Backup and archiving, remote chat history viewing, and more.

## Instructions

*   Detailed instructions are available in [UserGuide.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/UserGuide.md).
*   Minimalist version: [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini)
*   For UI modifications, clone [wx_dump_web](https://github.com/xaoyaoo/wxdump_web).
*   Refer to [CE obtaining base address.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/CE获取基址.md) and [wx database brief.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/wx数据库简述.md)

## Disclaimer (READ CAREFULLY!)

This project is for educational and informational purposes only. Please adhere to all legal and ethical guidelines when using this software. Misuse of this tool can lead to severe consequences.

1.  **Purpose:** Only for learning and communication. Do not use for illegal activities.
2.  **Usage Time:** Delete the code and compiled program within 24 hours of use.
3.  **Operation:** Backup and view authorized databases only.  Unauthorized activities are strictly prohibited.
4.  **Privacy:** Do not use this tool to steal others' privacy.
5.  **No Secondary Development:** No secondary development is permitted.
6.  **Acceptance:**  By downloading/using this tool, you agree to this disclaimer.
7.  **Prohibited Activities:**  Do not use this tool for illegal testing or penetration.
8.  **Consequences:**  The developers are not responsible for the consequences of misuse.
9.  **Disclaimer Modifications:** This disclaimer can change.  Review regularly.
10. **Compliance:** Users must adhere to laws and ethical norms when using this project.

## Acknowledgments

Thank you to the contributors:

[![PyWxDump CONTRIBUTORS](https://contrib.rocks/image?repo=xaoyaoo/PyWxDump)](https://github.com/xaoyaoo/PyWxDump/graphs/contributors)

UI CONTRIBUTORS:

[![UI CONTRIBUTORS](https://contrib.rocks/image?repo=xaoyaoo/wxdump_web)](https://github.com/xaoyaoo/wxdump_web/graphs/contributors)

otherContributors:

[643104191](https://github.com/643104191) (add [ctypes_utils](https://github.com/xaoyaoo/PyWxDump/blob/9e3e4cb5aec2b9b445c8283d61c58863f4129c6e/pywxdump/wx_info/ctypes_utils.py), Accelerated the acquisition of wxinfo; [9e3e4cb](https://github.com/xaoyaoo/PyWxDump/commit/9e3e4cb5aec2b9b445c8283d61c58863f4129c6e))