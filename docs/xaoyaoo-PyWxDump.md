# PyWxDump: Your Tool for WeChat Data Exploration and Backup

**PyWxDump empowers you to extract and analyze your WeChat data, offering insights and the ability to create backups.** ([Original Repo](https://github.com/xaoyaoo/PyWxDump))

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

*   **Account Information Extraction:** Retrieve WeChat nicknames, accounts, phone numbers, emails, database keys, and original IDs (wxid).
*   **Database Decryption:** Decrypt your WeChat databases using extracted keys.
*   **Chat History Viewing & Export:** View chat history via a web interface and export chat logs as HTML or CSV.
*   **Backup Capabilities:** Create backups of your WeChat chat logs.
*   **Multi-Account Support:** Handle information from multiple WeChat accounts.
*   **Minimalist Version:** Utilize the streamlined [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini) for core functionality.
*   **Extensive Documentation:** Offers descriptions of database fields and methods for base address offset and MAC database decryption.

## I. Project Introduction

### 1. Core Functionality

*   Obtain key information such as base address offsets for essential WeChat data.
*   Retrieve detailed user information for currently logged-in WeChat accounts.
*   Decrypt WeChat databases, enabling data access.
*   Integrate various database types for a unified viewing experience.

### 2. Extended Features

*   **Web-Based Chat Viewing:** Access your chat history through a user-friendly web interface.
*   **Export and Backup Options:** Export chat logs as HTML or CSV files for backup and analysis.
*   **Remote Chat History Access:** View chat history remotely (requires network access).

### 3. Additional Features

*   **Minimalist Version:** [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini) provides core functionalities.
*   **Multiple WeChat Instances:**  Supports handling information from multiple active WeChat instances.

### 4. Update Plan

*   Generate word clouds from chat logs.
*   Analyze per-person chat volume over time and visualize with line charts.
*   Develop annual visualization reports.
*   Improve Enterprise WeChat support.
*   Enable backup and viewing of Moments.
*   Optimize storage space.
*   Implement UI-controlled message sending.

## Ⅱ. Usage Instructions

For detailed instructions on using PyWxDump, please refer to:

*   **User Guide:** [UserGuide.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/UserGuide.md)
*   **Minimalist Version:** [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini)
*   **UI Customization:** Clone and modify the [wx_dump_web](https://github.com/xaoyaoo/wxdump_web) repository.

**Important Notes:**

*   Obtain the base address using Cheat Engine following instructions in [CE obtaining base address.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/CE获取基址.md). (This method is for learning purposes and can be replaced by the `wxdump bias` command)
*   Understand database parsing with [wx database brief.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/wx数据库简述.md)

## Ⅲ. Disclaimer (READ CAREFULLY!)

**This project is for educational and personal use only. Misuse is strictly prohibited.**

*   **Legal Use Only:** Use this project for legal, authorized purposes only.  The developers are not responsible for any illegal activities.
*   **Deletion Requirement:** Delete the source code and any compiled programs within 24 hours of use.
*   **Compliance:** Comply with all laws and regulations.
*   **Privacy:** This project should not be used to access, share, or distribute information illegally or without consent.
*   **No Secondary Development:** Secondary development is prohibited.
*   **No Illegal Testing or Penetration:** Avoid illegal testing or penetration with this project.
*   **Disclaimer Updates:** Regularly check this page for updates to the disclaimer.

## Ⅳ. Acknowledgements

A big thank you to all contributors!

[![PyWxDump CONTRIBUTORS](https://contrib.rocks/image?repo=xaoyaoo/PyWxDump)](https://github.com/xaoyaoo/PyWxDump/graphs/contributors)

UI Contributors:
[![UI CONTRIBUTORS](https://contrib.rocks/image?repo=xaoyaoo/wxdump_web)](https://github.com/xaoyaoo/wxdump_web/graphs/contributors)

Other Contributors:
[643104191](https://github.com/643104191) (add [ctypes_utils](https://github.com/xaoyaoo/PyWxDump/blob/9e3e4cb5aec2b9b445c8283d61c58863f4129c6e/pywxdump/wx_info/ctypes_utils.py), Accelerated the acquisition of wxinfo; [9e3e4cb](https://github.com/xaoyaoo/PyWxDump/commit/9e3e4cb5aec2b9b445c8283d61c58863f4129c6e))