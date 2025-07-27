# PyWxDump: Your Ultimate Tool for WeChat Data Extraction and Analysis

**Unlock the secrets of your WeChat data with PyWxDump, a powerful Python tool for extracting, decrypting, and analyzing your WeChat chat history.**  [Explore the original repository on GitHub](https://github.com/xaoyaoo/PyWxDump).

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

*   Welcome to contribute ideas or code to improve this project!

### Get a Quick Start Guide
If you're new, check out the Official Account `逍遥之芯` (scan the QR code below) and reply `PyWxDump` for a picture text tutorial.

### Find Answers to Your Questions
First check the [FAQ](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/FAQ.md) for answers. You can also get help by following the Official Account `逍遥之芯` and replying `FAQ`.

### Join the Community
QQ Group: [276392799](https://s.xaoyo.top/gOLUDl) or [276392799](https://s.xaoyo.top/bgNcRa) (Password: see [UserGuide.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/UserGuide.md)).

<div>
  <img align="" width="200"  src="https://github.com/xaoyaoo/PyWxDump/blob/master/doc/img/qrcode_gh.jpg" alt="the Official Accounts" title="the Official Accounts" height="200"/>
</div>

## Key Features

*   **Account Information Retrieval:** Get WeChat nickname, account, phone number, email, and database keys.
*   **Database Decryption:** Decrypt WeChat databases for data access.
*   **Chat History Viewing:** View and export chat history as HTML, CSV.
*   **Backup and Export:**  Create HTML backups of your chat logs.
*   **Web UI:** Access chat history through a web interface.
*   **Multi-WeChat Support:** Handles multiple WeChat instances.

## Project Introduction

### 1. Overview
PyWxDump is a powerful tool for obtaining WeChat account information (nicknames/accounts/phones/emails/database keys), decrypting databases, viewing wx chat, and exporting chat as html backups.

*   **<big>Support the project with a [![Star](https://img.shields.io/github/stars/xaoyaoo/PyWxDump.svg?style=social&label=Star)](https://github.com/xaoyaoo/PyWxDump/)!  Thank you! </big>**

### 2. Core Capabilities

*   **Get WeChat Info:** Retrieve base address offsets and extract user information (nickname, account, phone, email, original ID, folder path, and encryption key).
*   **Decrypt Databases:** Decrypt your WeChat database with the extracted key.
*   **Unified Viewing:** View chat history from various database types in a combined view.

### 3. Additional Features

*   **Web-Based Chat History:** View your chat history through a web interface.
*   **Export and Backup:** Export chat logs as HTML and CSV, and create backups.
*   **Remote Access:** View chat history remotely (requires network access).

### 4. Useful Resources

*   **Database Field Descriptions:** Documentation for some database fields.
*   **CE Tutorial:** Instructions on using Cheat Engine (CE) to obtain base address offsets.
*   **MAC Database Decryption:** Instructions on decrypting MAC databases.

### 5. Alternative
*   A minimalist version, [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini), is also available, providing only the ability to obtain database keys and database locations.

**Potential Use Cases:**

1.  Network security.
2.  Daily backup and archiving.
3.  Remote chat history viewing.
4.  And much more...

### 6. Future Development
*   Analyze chat logs of each person and generate word clouds.
*   Generate annual visualization reports
*   Increase support for enterprise WeChat
*   Viewing and backing up of the circle of friends
*   Clean up WeChat storage space
*   Automatically send messages to specified people through UI control

## Ⅲ. Disclaimer (VERY VERY VERY IMPORTANT ! ! ! ! ! !)

**Please carefully review and understand the disclaimer before using PyWxDump.**  This is for learning and communication purposes only.

### 1. Purpose of Use
*   This project is only for learning and communication purposes. **Please do not use it for illegal purposes.** Any illegal use is the sole responsibility of the user.

### 2. Usage Period
*   You must delete the source code and (compiled) program within 24 hours of use.

### 3. Operation Specifications
*   Backup and viewing of databases only under authorization.  Strictly prohibited for illegal use.
*   Strictly prohibited from stealing others' privacy.
*   Secondary development is strictly prohibited.

### 4. Acceptance of Disclaimer
*   Downloading, saving, browsing, or using this program constitutes agreement with and adherence to this disclaimer.

### 5. Forbidden Activities
*   It is prohibited to engage in illegal testing or penetration using the project's technologies. Any adverse consequences are the user's responsibility.

### 6. Disclaimer Modifications
*   The disclaimer may be modified.  Users should regularly review this page for updates.

### 7. Other
*   Users must comply with all relevant laws and ethical norms.

## Ⅳ. Acknowledgements

Thank you to the [contributors](https://github.com/xaoyaoo/PyWxDump/graphs/contributors) and the UI [contributors](https://github.com/xaoyaoo/wxdump_web/graphs/contributors)

otherContributors:

[643104191](https://github.com/643104191) (add [ctypes_utils](https://github.com/xaoyaoo/PyWxDump/blob/9e3e4cb5aec2b9b445c8283d61c58863f4129c6e/pywxdump/wx_info/ctypes_utils.py), Accelerated the acquisition of wxinfo; [9e3e4cb](https://github.com/xaoyaoo/PyWxDump/commit/9e3e4cb5aec2b9b445c8283d61c58863f4129c6e))