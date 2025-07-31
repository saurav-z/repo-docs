# PyWxDump: Your Tool for WeChat Data Extraction and Analysis

**PyWxDump** is a powerful Python-based tool designed for extracting and analyzing data from WeChat, allowing you to access your chat history and more. [Check it out on GitHub!](https://github.com/xaoyaoo/PyWxDump)

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

*   Welcome to provide more ideas or code to improve this project together.

### If you are a novice, please pay attention to the Official Accounts: `逍遥之芯` (the QR code is below), and reply: `PyWxDump` to get a picture text tutorial.

### If you have any questions, please check first: [FAQ](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/FAQ.md) Whether there is an answer, or follow the Official Accounts to reply: `FAQ`.

QQ GROUP：[276392799](https://s.xaoyo.top/gOLUDl) or [276392799](https://s.xaoyo.top/bgNcRa)（PASSWORD,please read:[UserGuide.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/UserGuide.md)）.

<div>
  <img align="" width="200"  src="https://github.com/xaoyaoo/PyWxDump/blob/master/doc/img/qrcode_gh.jpg" alt="the Official Accounts" title="the Official Accounts" height="200"/>
</div>

## Key Features

*   **WeChat Account Information Extraction:** Get essential account details like nicknames, accounts, phone numbers, and email addresses.
*   **Database Decryption:** Decrypt WeChat databases to access your chat history.
*   **Chat History Viewing:** View your chat history through a web interface.
*   **Data Export:** Export chat logs in HTML and CSV formats for backup and analysis.
*   **Remote Viewing:** Share your chat history remotely (requires network access).
*   **Minimalist Version:** Utilize the simplified [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini) for key and location retrieval.

### Core Functionality

*   Get the **base address offset** of WeChat nickname, WeChat account, WeChat phone number, WeChat email, and WeChat KEY
*   Get the WeChat nickname, WeChat account, WeChat phone number, WeChat email, WeChat KEY, WeChat original ID (wxid_******), and WeChat folder path of the currently logged-in WeChat
*   Decrypt WeChat database based on key
*   Combine multiple types of databases for unified viewing

### Extended Functionality

*   View chat history through the web
*   Support exporting chat logs as html, csv, and backing up WeChat chat logs
*   Remote viewing of WeChat chat history (must be network accessible, such as a local area network)

### Document Class

*   Provide descriptions of some fields in the database
*   Provide CE to obtain the base address offset method
*   Provide a decryption method for MAC database

### Other functions

*   Added a minimalist version of [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini), which provides only the ability to obtain database keys and database locations
*   Support multiple WeChat opening scenarios, obtain multiple user information, etc.

**Potential Uses:**

1.  Network security
2.  Daily backup archiving
3.  View chat history remotely
4.  ...and more!

## Update Plans

* 1.Analyze chat logs of each person and generate word clouds.
* ~~2.Analyze the number of chats per person per day and generate a line chart (day-number of chats)~~
* ~~3.Analyze the monthly and annual chat volume of different people and generate a line chart~~
* ~~4.Generate annual visualization reports~~
* 8.Increase support for enterprise WeChat
* 12.Viewing and backing up of the circle of friends
* ~~13.Clean up WeChat storage space and reduce the space occupied by WeChat (hopefully by selecting a person or group and finding out the media files involved in the chat logs of this group, such as pictures, videos, files, voice recordings, etc., and selectively (such as time periods) or batch-wise clearing them from the computer's cache by group conversation.)~~
* 14.Automatically send messages to specified people through UI control

##  Getting Started

*   **Detailed Instructions:** [UserGuide.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/UserGuide.md)
*   **Minimalist Version:** [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini)
*   **Web UI Customization:** Clone and modify [wx_dump_web](https://github.com/xaoyaoo/wxdump_web) to tailor the user interface.

【note】:

* For obtaining the base address using cheat engine, refer to [CE obtaining base address.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/CE获取基址.md)
  (This method can be replaced by the `wxdump bias` command, and is only used for learning principles.)
* For database parsing, refer to [wx database brief.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/wx数据库简述.md)

##  Disclaimer (READ CAREFULLY!)

**Important Notice: Please review the complete disclaimer in the original repository's README.**

This project is intended for educational purposes only. Improper use could violate privacy or legal regulations.

## Acknowledgements

*   Thank you to all [contributors](https://github.com/xaoyaoo/PyWxDump/graphs/contributors) for their valuable contributions.
*   Special thanks to the UI contributors for the [wxdump_web](https://github.com/xaoyaoo/wxdump_web/graphs/contributors).
*   Special thanks to [643104191](https://github.com/643104191) for their contributions.