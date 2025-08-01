# PyWxDump: Uncover, Decrypt, and Archive Your WeChat Data

**PyWxDump is a powerful Python tool that allows you to extract and decrypt WeChat account information, databases, and chat history, enabling you to create HTML backups and gain valuable insights. Access the original repository [here](https://github.com/xaoyaoo/PyWxDump).**

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

*   **Contribute:** Welcome to collaborate and improve the project.

### Join the Community

*   For support and tutorials, follow the official account: `逍遥之芯` (QR code below). Reply `PyWxDump` to get a tutorial.
*   Check the [FAQ](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/FAQ.md) for answers.
*   Join the QQ Group: [276392799](https://s.xaoyo.top/gOLUDl) or [276392799](https://s.xaoyo.top/bgNcRa) (password in [UserGuide.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/UserGuide.md)).

<div>
  <img align="" width="200"  src="https://github.com/xaoyaoo/PyWxDump/blob/master/doc/img/qrcode_gh.jpg" alt="the Official Accounts" title="the Official Accounts" height="200"/>
</div>

## Key Features

*   **Get WeChat Information:** Retrieve nickname, account, phone number, email, and database keys.
*   **Decrypt Databases:** Decrypt WeChat databases using the extracted keys.
*   **View Chat History:** Access and analyze your WeChat conversations.
*   **Export and Backup:** Export chat logs as HTML and CSV for archiving.
*   **Remote Viewing (LAN):** View chat history remotely.

## Core Functionality

*   **(1) Obtain Base Address Offsets:** Get offsets for key WeChat data.
*   **(2) Extract User Data:** Retrieve nickname, account, phone number, email, WeChat ID (wxid_\*), and folder path.
*   **(3) Database Decryption:** Decrypt WeChat databases.
*   **(4) Unified Viewing:** View data from multiple database types.

## Extended Functionality

*   **(1) Web-Based Chat Viewer:** Browse chat history through a web interface.
*   **(2) Export Options:** Export chat logs to HTML and CSV formats for backup.
*   **(3) Remote Access:** View chat history remotely.

## Documentation

*   **(1) Field Descriptions:** Understand the database structure.
*   **(2) CE Method:** Instructions for obtaining the base address offset.
*   **(3) MAC Decryption:** Instructions for decrypting MAC databases.

## Other features

*   **(1) PyWxDumpMini:** A minimalist version for obtaining database keys and locations.
*   **(2) Multi-Account Support:** Supports multiple WeChat instances.

**Use Cases:**

1.  Network security.
2.  Daily backup archiving.
3.  Remote chat history viewing (web-based).
4.  And more!

## Planned Updates

*   Analyze individual chat logs to generate word clouds.
*   Generate chat volume charts (per-day, monthly, annual).
*   Generate annual visualization reports.
*   Increase support for enterprise WeChat.
*   Viewing and backing up of the circle of friends.
*   Clean up WeChat storage space and reduce the space occupied by WeChat.
*   Automatically send messages to specified people through UI control.

## Project Details

PyWxDump is a Python-based refactoring of [SharpWxDump](https://github.com/AdminTest0/SharpWxDump), with many new features.

*   **Project Address:** [https://github.com/xaoyaoo/PyWxDump](https://github.com/xaoyaoo/PyWxDump)
*   **Platform:** Primarily tested on Windows.  May have issues on macOS and Linux.
*   **Issues/Suggestions:** Please submit issues on GitHub for any missing information, bugs, or improvement suggestions related to  [WX\_OFFS.json](https://github.com/xaoyaoo/PyWxDump/tree/master/pywxdump/WX_OFFS.json).
*   **FAQ:** Refer to the [FAQ](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/FAQ.md) for common questions.
*   **Changelog:** View the [CHANGELOG](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/CHANGELOG.md) for updates.
*   **Web UI:** Web UI is developed using VUE+ElementUI. The repository location is [wxdump_web](https://github.com/xaoyaoo/wxdump_web).
*   **Implementation Principle:** To understand the principle of wxdump, follow the Official Accounts: `逍遥之芯`, reply: `原理` to get the principle analysis.
*   [:sparkling\_heart: Support Me]( https://github.com/xaoyaoo/xaoyaoo/blob/main/donate.md)

## Star History

<details>
<summary>click to expand</summary>

[![Star History Chart](https://api.star-history.com/svg?repos=xaoyaoo/pywxdump&type=Date)](https://star-history.com/#xaoyaoo/pywxdump&Date)

</details>

## Instructions

*   Detailed instructions are available in [UserGuide.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/UserGuide.md).
*   For the minimalist version, see [pywxdumpmini](https://github.com/xaoyaoo/pywxdumpmini).
*   To modify the UI, clone [wx_dump_web](https://github.com/xaoyaoo/wxdump_web).

**Notes:**

*   Obtaining the base address using Cheat Engine: [CE obtaining base address.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/CE获取基址.md) (This method can be replaced by the `wxdump bias` command and is primarily for learning the principles.)
*   Database parsing details: [wx database brief.md](https://github.com/xaoyaoo/PyWxDump/tree/master/doc/wx数据库简述.md)

## Disclaimer (READ CAREFULLY!)

### 1. Intended Use

*   This project is intended for educational and communication purposes *only*. **DO NOT use it for illegal activities.**  You are solely responsible for any misuse.

### 2. Usage Timeframe

*   You must delete the source code and any compiled programs within 24 hours of downloading, saving, compiling, or using them.  Any use beyond this timeframe is outside the scope of this project.

### 3. Operational Guidelines

*   Use this project only for authorized database backup and viewing.  Any illegal activities are strictly prohibited, and you bear all responsibility.
*   It is strictly prohibited to use it to steal others' privacy. Otherwise, all relevant responsibilities shall be borne by yourself.
*   It is strictly prohibited to conduct secondary development, otherwise all related responsibilities shall be borne by yourself.

### 4. Acceptance of Disclaimer

*   By downloading, saving, browsing the source code, or downloading, installing, compiling, or using this program, you agree to this disclaimer and its terms.

### 5. Prohibition of Illegal Testing or Penetration

*   The technologies in this project must not be used for illegal testing or penetration. The codes or related technologies from this project should not be used for illegal work.
*   You are responsible for all negative consequences, including, but not limited to, data leaks, system failures, and privacy breaches.

### 6. Disclaimer Modification

*   This disclaimer may be updated based on the project's operation and changes in laws and regulations. Regularly check this page for the latest version.

### 7. Other Considerations

*   Users must comply with all relevant laws, regulations, and ethical guidelines. The project and its developers are not responsible for disputes or losses resulting from your violation of these regulations.
*   Carefully review this disclaimer and ensure you adhere to the rules when using this project.

## Acknowledgements

[![PyWxDump CONTRIBUTORS](https://contrib.rocks/image?repo=xaoyaoo/PyWxDump)](https://github.com/xaoyaoo/PyWxDump/graphs/contributors)  

UI CONTRIBUTORS:    

[![UI CONTRIBUTORS](https://contrib.rocks/image?repo=xaoyaoo/wxdump_web)](https://github.com/xaoyaoo/wxdump_web/graphs/contributors)

otherContributors:

[643104191](https://github.com/643104191) (add [ctypes_utils](https://github.com/xaoyaoo/PyWxDump/blob/9e3e4cb5aec2b9b445c8283d61c58863f4129c6e/pywxdump/wx_info/ctypes_utils.py), Accelerated the acquisition of wxinfo; [9e3e4cb](https://github.com/xaoyaoo/PyWxDump/commit/9e3e4cb5aec2b9b445c8283d61c58863f4129c6e))