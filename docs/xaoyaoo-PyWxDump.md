# PyWxDump: Your Toolkit for WeChat Data Extraction and Analysis

**Unlock the secrets of your WeChat data with PyWxDump, a powerful Python tool for exploring your chat history and more.** ([View on GitHub](https://github.com/xaoyaoo/PyWxDump))

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

*   **Data Extraction:** Obtain crucial WeChat account information, including nicknames, accounts, phone numbers, emails, and database keys.
*   **Database Decryption:** Decrypt your WeChat databases to access your chat history.
*   **Chat Viewing:** View your WeChat chat logs in a user-friendly web interface.
*   **Data Export:** Export chat logs as HTML and CSV for backups and analysis.
*   **Remote Access:** Access your chat history remotely (requires network configuration).
*   **Minimalist Version:** Utilize [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini) for key and database location retrieval.

## Project Introduction

PyWxDump is a powerful tool designed to extract and analyze data from WeChat. It provides the following capabilities:

### Core Functionality

*   Retrieves base address offsets for essential WeChat information.
*   Fetches detailed WeChat user data, including nickname, account, phone number, email, key, original ID, and folder path.
*   Decrypts WeChat databases using provided keys.
*   Merges multiple database types for unified viewing.

### Extended Features

*   Web-based chat history viewer.
*   Export chat logs to HTML and CSV formats.
*   Provides remote access to chat history.

### Supporting Documentation

*   Database field descriptions.
*   Methods for obtaining base address offsets using Cheat Engine.
*   MAC database decryption methods.

## Update Plan

*   Analyze chat logs of each person and generate word clouds.
*   ~~Analyze the number of chats per person per day and generate a line chart (day-number of chats)~~
*   ~~Analyze the monthly and annual chat volume of different people and generate a line chart~~
*   ~~Generate annual visualization reports~~
*   Increase support for enterprise WeChat
*   Viewing and backing up of the circle of friends
*   ~~Clean up WeChat storage space and reduce the space occupied by WeChat (hopefully by selecting a person or group and finding out the media files involved in the chat logs of this group, such as pictures, videos, files, voice recordings, etc., and selectively (such as time periods) or batch-wise clearing them from the computer's cache by group conversation.)~~
*   Automatically send messages to specified people through UI control

## Getting Started

*   Detailed instructions: [UserGuide.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/UserGuide.md)
*   For obtaining the base address using cheat engine, refer to [CE obtaining base address.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/CE获取基址.md)
  (This method can be replaced by the `wxdump bias` command, and is only used for learning principles.)
*   For database parsing, refer to [wx database brief.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/wx数据库简述.md)

## Disclaimer (VERY IMPORTANT)

*   **Use responsibly:** This project is for educational and personal use only. **Do not use it for illegal activities.**
*   **No liability:** The developers are not responsible for any misuse or illegal activities conducted using this tool. Users are solely responsible for their actions.
*   **Data handling:** Delete the source code and compiled program within 24 hours of use.
*   **Strict prohibitions:** Do not use this tool for privacy violations or secondary development.

## Acknowledgements

*   [Contributors](https://github.com/xaoyaoo/PyWxDump/graphs/contributors)
*   [UI Contributors](https://github.com/xaoyaoo/wxdump_web/graphs/contributors)
*   Other Contributors: [643104191](https://github.com/643104191), for ctypes_utils contribution

---