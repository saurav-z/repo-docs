# PyWxDump: Unlock Your WeChat Data with Python

**PyWxDump is a powerful Python tool that allows you to extract and analyze your WeChat data, providing insights into your chats and database.**  [Explore the original repository](https://github.com/xaoyaoo/PyWxDump)

[![Python](https://img.shields.io/badge/Python-3-blue.svg)](https://www.python.org/)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/xaoyaoo/pywxdump)](https://github.com/xaoyaoo/PyWxDump)
[![GitHub all releases](https://img.shields.io/github/downloads/xaoyaoo/pywxdump/total)](https://github.com/xaoyaoo/PyWxDump)
[![GitHub stars](https://img.shields.io/github/stars/xaoyaoo/PyWxDump.svg)](https://github.com/xaoyaoo/PyWxDump)
[![GitHub forks](https://img.shields.io/github/forks/xaoyaoo/PyWxDump.svg)](https://github.com/xaoyaoo/PyWxDump/fork)
[![GitHub issues](https://img.shields.io/github/issues/xaoyaoo/PyWxDump)](https://github.com/xaoyaoo/PyWxDump/issues)

[![PyPI](https://img.shields.io/pypi/v/pywxdump)](https://pypi.org/project/pywxdump/)
[![Wheel](https://img.shields.io/pypi/wheel/pywxdump)](https://pypi.org/project/pywxdump/)
[![PyPI-Downloads](https://img.shields.io/pypi/dm/pywxdump)](https://pypistats.org/packages/pywxdump)
[![GitHub license](https://img.shields.io/pypi/l/pywxdump)](https://github.com/xaoyaoo/PyWxDump/blob/master/LICENSE)

## Key Features:

*   **Account Information Extraction:** Retrieve your WeChat nickname, account, phone number, email, and database keys.
*   **Database Decryption:** Decrypt your WeChat database for in-depth analysis.
*   **Chat History Viewing:** View your chat history through a web interface or export it as HTML or CSV backups.
*   **Data Export:** Export chat logs as HTML or CSV files for archiving and analysis.
*   **Remote Viewing:** Access your WeChat chat history remotely (requires network configuration).
*   **Minimalist Version:** Utilize `pywxdumpmini` for basic key and database location retrieval.
*   **Multiple WeChat Support:** Supports retrieving information for multiple WeChat accounts.

## I. Project Overview

### 1.  Brief Introduction

PyWxDump is a Python tool designed to help you extract information from your WeChat account, decrypt your database, and analyze your chat history.
* <strong><big>Don't forget to star the repo if you find it useful: [![Star](https://img.shields.io/github/stars/xaoyaoo/PyWxDump.svg?style=social&label=Star)](https://github.com/xaoyaoo/PyWxDump/)!</big></strong>

### 2. Core Features

*   **Get Important Offsets**: Retrieve offsets for key WeChat data like nicknames, accounts, and database keys.
*   **Get User Information**: Extract user information, including nickname, account, phone number, email, KEY, original ID, and folder path.
*   **Decrypt Databases**: Decrypt WeChat databases using extracted keys.
*   **Unified Viewing**: Combine multiple database types for easy viewing and analysis.

### 3. Extended Functionality

*   **Web-Based Chat Viewer**: View chat history through an intuitive web interface.
*   **Export Chat Logs**: Export chat logs as HTML, CSV, for easy backup.
*   **Remote Access**: Access chat history remotely (requires network setup).

### 4. Additional Features

*   **Documentation:** Provides descriptions of database fields and methods for obtaining the base address offset, and MAC database decryption.
*   **Minimalist Version:** `pywxdumpmini` offers simplified database key and location retrieval.
*   **Multi-Account Support:** Works with multiple WeChat accounts.

### 5. Potential Use Cases

*   Network security research
*   Personal data backup and archiving
*   Remote chat history access (with proper configuration)

### 6. Planned Updates

The project is under active development with planned features like:

*   Chat log analysis and word cloud generation.
*   Visualization of chat activity over time (daily, monthly, and annual).
*   Support for Enterprise WeChat.
*   Friends circle viewing and backup functionality.
*   Message sending automation.

## II. Instructions for Use

*   Detailed instructions are available in [UserGuide.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/UserGuide.md).
*   For the minimalist version, see [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini).
*   To modify the UI, clone [wx_dump_web](https://github.com/xaoyaoo/wxdump_web).
*   For information on obtaining the base address using Cheat Engine, see [CE obtaining base address.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/CE%E8%8E%B7%E5%8F%96%E5%9F%BA%E5%9D%80.md)
*   For database parsing, refer to [wx database brief.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/wx%E6%95%B0%E6%8D%AE%E5%BA%93%E7%AE%80%E8%BF%B0.md)

## III. Disclaimer (IMPORTANT!)

**Please read and understand the following disclaimer before using PyWxDump:**

*   **Purpose of Use:** This project is intended for learning and communication purposes only.  **Do not use it for illegal activities.**  Users are solely responsible for any misuse.
*   **Usage Period:**  You must delete the source code and compiled program within 24 hours of downloading and using it.
*   **Operation Specifications:** Only authorized database backups and viewing are permitted.  It is strictly forbidden to use this tool for illegal purposes, privacy violations, or secondary development.
*   **Acceptance of Disclaimer:**  By downloading, using, or browsing the source code, you agree to abide by this disclaimer.
*   **Forbidden Activities:**  Do not use this project for illegal testing, penetration, or any activities that violate the law.
*   **Disclaimer Modification:** The disclaimer may be updated, so review it regularly.
*   **Compliance with Laws:** Users must comply with all applicable laws and regulations. The developers are not responsible for any violations.

## IV. Acknowledgments

*   Thanks to all the [contributors](https://github.com/xaoyaoo/PyWxDump/graphs/contributors)!

*   UI Contributors:  [wxdump_web Contributors](https://github.com/xaoyaoo/wxdump_web/graphs/contributors)

*   Other Contributors:
    *   [643104191](https://github.com/643104191)