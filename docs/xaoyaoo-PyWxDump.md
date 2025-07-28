# PyWxDump: Your Tool for WeChat Data Analysis and Backup

[PyWxDump](https://github.com/xaoyaoo/PyWxDump) is a powerful Python-based tool designed for extracting and analyzing data from WeChat, offering decryption, chat export, and more.

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

*   **Extract WeChat Account Information:** Obtain nicknames, accounts, phone numbers, emails, and database keys.
*   **Database Decryption:** Decrypt your WeChat databases for data access.
*   **Chat History Export:** Export your chat history as HTML, CSV, or for web viewing.
*   **Web-Based Chat Viewing:**  View your WeChat chat history remotely via web interface (requires network access).
*   **Data Analysis**: Analyze chat logs and visualize them (word cloud generation, charts, etc.).

## I. Project Introduction

### 1. Core Functionality

*   Get base address offsets for key WeChat data (nickname, account, phone, email, KEY).
*   Retrieve logged-in WeChat information: nickname, account, phone, email, KEY, original ID (wxid), and folder path.
*   Decrypt WeChat databases using encryption keys.
*   Combine multiple database types for unified viewing.

### 2. Extended Features

*   Web-based chat history viewing for easy access.
*   Export chat logs in HTML, CSV formats for backups and archiving.
*   Remote access to chat history.

### 3. Useful Documents

*   Database field descriptions.
*   Cheat Engine method to obtain base address offsets.
*   Decryption method for MAC databases.

### 4.  Other Functions
*   Minimalist version [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini), providing database key and location retrieval.
*   Support for multiple WeChat instances, retrieving user information from different accounts.

### 5.  Potential Applications

1.  Network security analysis
2.  Daily backup and archiving.
3.  Remote chat history viewing (via web).

### 6. Future Development

*   Analyze chat logs for individual users and generate word clouds.
*   Generate interactive charts and visual reports.
*   Support for Enterprise WeChat.
*   Backup of moments/WeChat friend's timeline.
*   Clean up WeChat storage space.
*   Automatically send messages via UI control.

## II. Instructions

Detailed usage instructions can be found in [UserGuide.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/UserGuide.md).  For the minimalist version, see [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini).

## III. Disclaimer (READ CAREFULLY!)

**Important Note:** This project is designed for learning and communication. **Do not use it for illegal activities.  You are solely responsible for your usage.**

*   **Purpose of Use:**  For educational purposes only.  Any misuse is strictly prohibited.
*   **Time Limit:** Delete source code and (compiled) program within 24 hours of use.  Continuing to use it after this period is not permitted.
*   **Operation Guidelines:**  Authorized database backup/viewing only.  Prohibited use cases include: privacy invasion, stealing data, reverse engineering, or malicious uses.
*   **Disclaimer Acceptance:** Downloading or using this project implies agreement with the disclaimer.
*   **Forbidden Activities:**  No illegal testing, penetration attempts, or malicious use.
*   **Liability:** You are responsible for any negative consequences resulting from the use of this project, including legal liabilities.
*   **Disclaimer Updates:**  Check this page for updates to the disclaimer.
*   **Compliance:**  Adhere to all applicable laws, regulations, and ethical standards.

## IV. Acknowledgements

Thanks to the contributors:
[![PyWxDump CONTRIBUTORS](https://contrib.rocks/image?repo=xaoyaoo/PyWxDump)](https://github.com/xaoyaoo/PyWxDump/graphs/contributors)  

UI CONTRIBUTORS:    

[![UI CONTRIBUTORS](https://contrib.rocks/image?repo=xaoyaoo/wxdump_web)](https://github.com/xaoyaoo/wxdump_web/graphs/contributors)

otherContributors:

[643104191](https://github.com/643104191) (add [ctypes_utils](https://github.com/xaoyaoo/PyWxDump/blob/9e3e4cb5aec2b9b445c8283d61c58863f4129c6e/pywxdump/wx_info/ctypes_utils.py), Accelerated the acquisition of wxinfo; [9e3e4cb](https://github.com/xaoyaoo/PyWxDump/commit/9e3e4cb5aec2b9b445c8283d61c58863f4129c6e))

**Star the Project!** 
<a href="https://github.com/xaoyaoo/PyWxDump"><img src="https://img.shields.io/github/stars/xaoyaoo/PyWxDump?style=social&label=Star" alt="Star PyWxDump on GitHub"></a>

**Need Help or Have Questions?**

*   Check the [FAQ](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/FAQ.md).
*   For implementation details, see Official Accounts: `逍遥之芯` (reply: `原理`).
*   [Donate to Support the Project]( https://github.com/xaoyaoo/xaoyaoo/blob/main/donate.md)
```

Key improvements:

*   **SEO-optimized title and introduction:**  Used keywords like "WeChat," "data analysis," and "backup" in the title and introduction.
*   **Clear headings:**  Organized the README with clear headings and subheadings for readability and SEO.
*   **Bulleted key features:**  Highlight key features using bullet points for easy scanning.
*   **Concise descriptions:**  Improved the descriptions of features and functionality.
*   **Call to action (Star the project!):** Added a clear call to action for starring the repository.
*   **Improved Disclaimer:** The disclaimer section is more detailed and explicit to protect the developer from liability.
*   **Added a one-sentence hook:**  Used "PyWxDump: Your Tool for WeChat Data Analysis and Backup"
*   **Removed unnecessary badges**: The badges weren't that important, and removing some makes the page more readable.