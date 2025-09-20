# TagStudio: Organize and Rediscover Your Files with Powerful Tagging

**Tired of endless file searches? TagStudio offers a user-focused document management system to help you regain control of your digital life.**  [View on GitHub](https://github.com/TagStudioDev/TagStudio)

[![Translation](https://hosted.weblate.org/widget/tagstudio/strings/svg-badge.svg)](https://hosted.weblate.org/projects/tagstudio/strings/)
[![PyTest](https://github.com/TagStudioDev/TagStudio/actions/workflows/pytest.yaml/badge.svg)](https://github.com/TagStudioDev/TagStudio/actions/workflows/pytest.yaml)
[![MyPy](https://github.com/TagStudioDev/TagStudio/actions/workflows/mypy.yaml/badge.svg)](https://github.com/TagStudioDev/TagStudio/actions/workflows/mypy.yaml)
[![Ruff](https://github.com/TagStudioDev/TagStudio/actions/workflows/ruff.yaml/badge.svg)](https://github.com/TagStudioDev/TagStudio/actions/workflows/ruff.yaml)
[![Downloads](https://img.shields.io/github/downloads/TagStudioDev/TagStudio/total.svg?maxAge=2592001)](https://github.com/TagStudioDev/TagStudio/releases)

<p align="center">
  <img width="60%" src="docs/assets/github_header.png">
</p>

TagStudio empowers you to organize photos and files using a flexible, tag-based system without the limitations of proprietary formats or file structure overhauls.  Explore a new way to manage your digital assets. **Read the documentation and more at [docs.tagstud.io](https://docs.tagstud.io)!**

> [!NOTE]
> Thank you for being patient as we've migrated our database backend from JSON to SQL! The previous warnings about the main branch being experimental and unsupported have now been removed, and any pre-existing library save files created with official TagStudio releases are able to be opened and migrated with the new v9.5+ releases!

> [!IMPORTANT]
> This project is still in an early state. There are many missing optimizations and QoL features, as well as the presence of general quirks and occasional jankiness. Making frequent backups of your library save data is **always** important, regardless of what state the program is in.
>
> With this in mind, TagStudio will _NOT:_
>
> -   Touch, move, or mess with your files in any way _(unless explicitly using the "Delete File(s)" feature, which is locked behind a confirmation dialog)_.
> -   Ask you to recreate your tags or libraries after new releases. It's our highest priority to ensure that your data safely and smoothly transfers over to newer versions.
> -   Cause you to suddenly be able to recall your 10 trillion downloaded images that you probably haven't even seen firsthand before. You're in control here, and even tools out there that use machine learning still needed to be verified by human eyes before being deemed accurate.

<p align="center">
  <img width="80%" src="docs/assets/screenshot.png" alt="TagStudio Screenshot">
</p>
<p align="center">
  <i>TagStudio Alpha v9.5.0 running on macOS Sequoia.</i>
</p>

## Key Features

*   **Flexible Tagging System:** Organize files with custom tags, tag inheritance, and tag categories.
*   **Custom Metadata:** Add comprehensive information like names, descriptions, and more.
*   **Powerful Search:** Find files quickly using tags, file paths, file types, and boolean operators.
*   **File Entry Management:**  Preview and open a wide range of file types.
*   **File Integrity:** Designed to work with existing file structures and workflows, without mandatory sidecar files or file modification.

##  Getting Started

### Installation

Download executable builds of TagStudio from the [Releases](https://github.com/TagStudioDev/TagStudio/releases) page.

*   **Supported Platforms:** Windows, macOS (Apple Silicon & Intel), and Linux.
*   **Portable Releases:** Available for Windows and Linux for easier portability.
*   **FFmpeg Dependency:** Requires FFmpeg for video thumbnails and playback. Please see the [FFmpeg Help](/docs/help/ffmpeg.md) guide.

> [!CAUTION]
> **We do not currently publish TagStudio to any package managers. Any TagStudio distributions outside of the GitHub [Releases](https://github.com/TagStudioDev/TagStudio/releases) page are _unofficial_ and not maintained by us.**

### Usage

1.  **Create or Open a Library:** Use `File -> Open/Create Library` to start.
2.  **Automatic Scanning:**  TagStudio scans for files upon library creation or refresh.
3.  **Tagging:**  Add tags via the "Add Tag" button, menu option, or keyboard shortcut (<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>T</kbd>).
4.  **Adding Metadata:** Use the "Add Field" button to assign metadata to a file entry.
5.  **Library Saving:** Libraries are automatically saved.  Backups can be created via `File -> Save Library Backup`.

## Contribute

Contribute to TagStudio by reviewing the [contribution guidelines](/CONTRIBUTING.md).

Translation hosting generously provided by [Weblate](https://weblate.org/en/). Check out our [project page](https://hosted.weblate.org/projects/tagstudio/) to help translate TagStudio!

## FAQ

*   **Project State:** Currently in Alpha (v9.5.0), with ongoing feature development and bug fixes.
*   **Future Features:** See the [Feature Roadmap](https://docs.tagstud.io/updates/roadmap/) for planned features.
*   **Features Not Planned:**
    *   Native Cloud Integration
    *   Native ChatGPT/Non-Local LLM Integration

For further details, consult the [documentation](https://docs.tagstud.io/) and [FAQ](#faq).