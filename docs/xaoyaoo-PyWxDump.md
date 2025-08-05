# PyWxDump: Powerful WeChat Data Extraction and Analysis Tool

**Unlock the secrets of your WeChat data with PyWxDump, a versatile Python tool for extracting, decrypting, and analyzing your WeChat information.**  [Visit the original repository](https://github.com/xaoyaoo/PyWxDump)

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

## Key Features

*   **WeChat Account Information Retrieval:** Get essential data like nicknames, accounts, phone numbers, emails, and database keys.
*   **Database Decryption:** Decrypt your WeChat database to access your chat history.
*   **Chat History Viewing:**  View chat history through the web interface.
*   **Export Capabilities:**  Export chat logs as HTML and CSV for backup and analysis.
*   **Remote Access:**  View your WeChat chat history remotely (requires network access).
*   **Minimalist Version Available:**  Use the [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini) for key and database location retrieval.

## Project Introduction

### Core Functionality

*   Get the base address offset of WeChat nickname, WeChat account, WeChat phone number, WeChat email, and WeChat KEY
*   Get WeChat account information: nickname, account, phone number, email, KEY, original ID, and folder path.
*   Decrypt WeChat databases using the obtained key.
*   Combine and view data from different database types.

### Extended Functionality

*   Web-based chat history viewing
*   Export chat logs as HTML or CSV for backup.
*   Support for viewing chat logs remotely.

### Resources Provided

*   Database field descriptions.
*   CE (Cheat Engine) method for obtaining base address offset.
*   Decryption method for MAC database.

### Other features
* Added a minimalist version of [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini), which provides only the ability to obtain database keys and database locations
* Support multiple WeChat opening scenarios, obtain multiple user information, etc.

**Star this project if you find it helpful! [![Star](https://img.shields.io/github/stars/xaoyaoo/PyWxDump.svg?style=social&label=Star)](https://github.com/xaoyaoo/PyWxDump/)**

##  Use Cases

1.  Network security analysis.
2.  Daily data backup.
3.  Remote chat history viewing.

## Update Plan

The project is under active development, with plans to add the following features:

*   Analyze chat logs and generate word clouds.
*   Analyze chat volume per person and generate line charts.
*   Generate annual visualization reports.
*   Increase support for enterprise WeChat.
*   View and backup Moments/Friends circle.
*   Clean up WeChat storage space.
*   Automated message sending.

## Disclaimer (Important!)

**This tool is for learning, communication, and authorized backup purposes only.** Improper use can violate user privacy, so please ensure that you read, understand, and abide by all disclaimers found in the [UserGuide.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/UserGuide.md).  Any misuse or illegal activity is solely the user's responsibility.

## Acknowledgments

*   [CONTRIBUTORS](https://github.com/xaoyaoo/PyWxDump/graphs/contributors)
*   [UI CONTRIBUTORS](https://github.com/xaoyaoo/wxdump_web/graphs/contributors)
*   [Other Contributors](https://github.com/xaoyaoo/PyWxDump/graphs/contributors)