# PyWxDump: Your Comprehensive WeChat Data Extraction & Analysis Tool

**PyWxDump empowers you to extract, decrypt, and analyze your WeChat data, offering a powerful suite of features for data exploration and backup.**  Find the original project here: [https://github.com/xaoyaoo/PyWxDump](https://github.com/xaoyaoo/PyWxDump)

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

*   **Account Information Extraction:** Retrieves WeChat account details like nicknames, accounts, phone numbers, emails, and database keys.
*   **Database Decryption:** Decrypts WeChat databases for data access.
*   **Chat History Viewing:** Access and explore your chat history through a web interface.
*   **Export Options:** Export chat logs as HTML or CSV for backup and archiving.
*   **Remote Chat Viewing:** (Requires network access).
*   **Minimalist Version:** Utilize the standalone [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini) for essential key and location retrieval.
*   **Supports Multiple WeChat Instances:** Handle multiple logged-in WeChat accounts.

## Project Introduction

PyWxDump, a powerful tool for managing your WeChat data, offers a refactored Python version of SharpWxDump, enhanced with a range of new features.

### Core Functionality:

*   Obtain base address offsets for key WeChat information.
*   Retrieve essential user data from the currently logged-in WeChat instance.
*   Decrypt WeChat databases using the acquired keys.
*   Consolidated view of multiple database types for simplified analysis.

### Extended Functionality:

*   View and analyze chat history via a web interface.
*   Export chat logs in HTML and CSV formats, facilitating backup and archiving.
*   Enable remote viewing of WeChat chat history.

### Useful Documentation:

*   Database field descriptions.
*   CE method to obtain base address offset.
*   MAC database decryption.

### Additional Features:

*   A minimalist version, [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini), focusing on database key and location retrieval.
*   Support for accessing information from multiple WeChat instances.

### Potential Use Cases:

1.  Network Security Analysis
2.  Personal Data Backup
3.  Remote Chat History Review
4.  [More Use Cases]

## Update Plans:

*   Analyze chat logs for individual word clouds.
*   Create daily chat volume charts per person.
*   Generate charts for chat activity per month/year.
*   Generate yearly data visualization reports.
*   Support for enterprise WeChat.
*   [More Features]

## Important Information:

*   The project is primarily tested on Windows.
*   Please report any issues or suggestions via GitHub issues.
*   Refer to the FAQ ([FAQ](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/FAQ.md)) for common questions and the CHANGELOG ([CHANGELOG](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/CHANGELOG.md)) for update details.
*   Web UI Repository: [wxdump_web](https://github.com/xaoyaoo/wxdump_web )

## Usage Instructions:

Detailed instructions are available in [UserGuide.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/UserGuide.md).

To modify the UI, clone and customize the [wx_dump_web](https://github.com/xaoyaoo/wxdump_web) repository (developed using VUE+ElementUI).

## Disclaimer (READ CAREFULLY!)

**Important legal and ethical considerations. This project is strictly for authorized use.**

1.  **Intended Use:** This project is for learning and communication only; **do not use it for illegal activities.** The user is solely responsible for their actions.
2.  **Obligation:** Delete the project's source code and compiled program within 24 hours of use.
3.  **Operation Guidelines:** Authorized use of the database only. Prohibited for unauthorized purposes, privacy breaches, or reverse engineering.
4.  **Acceptance:** By downloading, saving, or using this program, you agree to and are bound by this disclaimer.
5.  **Prohibitions:** Unauthorized testing, penetration attempts, or use for illegal work are strictly forbidden.
6.  **Responsibility:** All risks are the user's responsibility.
7.  **Modification:** The disclaimer is subject to modification; users should regularly check for updates.
8.  **Compliance:** Users must comply with all applicable laws and ethics.

## Acknowledgments

**Huge thanks to all contributors!**

*   Project Contributors:  [![PyWxDump CONTRIBUTORS](https://contrib.rocks/image?repo=xaoyaoo/PyWxDump)](https://github.com/xaoyaoo/PyWxDump/graphs/contributors)
*   UI Contributors: [![UI CONTRIBUTORS](https://contrib.rocks/image?repo=xaoyaoo/wxdump_web)](https://github.com/xaoyaoo/wxdump_web/graphs/contributors)
*   Other Contributors: [643104191](https://github.com/643104191),