# PyWxDump: Your Tool for WeChat Data Analysis and Backup

**PyWxDump is a powerful Python-based tool to extract, decrypt, and analyze WeChat data for personal use and archiving.** ([Original Repository](https://github.com/xaoyaoo/PyWxDump))

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

*   **Account Information Extraction:** Retrieve WeChat account information (nickname, account, phone number, email, and database key).
*   **Database Decryption:** Decrypt WeChat databases for data access.
*   **Chat History Viewing:** View chat history via web interface.
*   **Data Export:** Export chat logs as HTML and CSV for backup and archiving.
*   **Remote Viewing:** Access your WeChat chat history remotely.
*   **Minimalist Version:** Includes [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini), for retrieving keys and database locations.

## I. Project Introduction

### 1. Core Functionality:

*   Retrieve base address offsets for essential WeChat data.
*   Extract user-specific information from the currently logged-in WeChat account.
*   Decrypt WeChat databases using the obtained keys.
*   Consolidated viewing of data from various database types.

### 2. Extended Features:

*   Web-based chat history viewing.
*   Export chat logs in HTML and CSV formats.
*   Remote access to chat history (requires network access).
*   Descriptions of database fields and CE method to obtain the base address offset
*   MAC database decryption.

### 3. Use Cases:

*   Personal data backup and archiving.
*   Network security research.
*   Remote access to chat history.

### 4. Update Plan:

(This section contains updates for future releases)

## II. Instructions for Use

*   For detailed usage instructions, see [UserGuide.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/UserGuide.md).
*   For the minimalist version, see [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini)
*   If you want to modify the UI, clone the [wx_dump_web](https://github.com/xaoyaoo/wxdump_web) and modify it as needed (the UI is developed using VUE+ElementUI)

## III. Disclaimer (IMPORTANT!)

(This section contains the disclaimer and usage restrictions)

## IV. Acknowledgments

*   Special thanks to the contributors to the PyWxDump project. ([Contributors](https://github.com/xaoyaoo/PyWxDump/graphs/contributors))
*   UI Contributors: ([UI Contributors](https://github.com/xaoyaoo/wxdump_web/graphs/contributors))