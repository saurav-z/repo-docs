# TagStudio: Organize Your Files Your Way

Tired of messy files and endless searching? **TagStudio is a user-focused document management system that puts you in control of your digital life.** ([Original Repo](https://github.com/TagStudioDev/TagStudio))

[![Translation](https://hosted.weblate.org/widget/tagstudio/strings/svg-badge.svg)](https://hosted.weblate.org/projects/tagstudio/strings/)
[![PyTest](https://github.com/TagStudioDev/TagStudio/actions/workflows/pytest.yaml/badge.svg)](https://github.com/TagStudioDev/TagStudio/actions/workflows/pytest.yaml)
[![MyPy](https://github.com/TagStudioDev/TagStudio/actions/workflows/mypy.yaml/badge.svg)](https://github.com/TagStudioDev/TagStudio/actions/workflows/mypy.yaml)
[![Ruff](https://github.com/TagStudioDev/TagStudio/actions/workflows/ruff.yaml/badge.svg)](https://github.com/TagStudioDev/TagStudio/actions/workflows/ruff.yaml)
[![Downloads](https://img.shields.io/github/downloads/TagStudioDev/TagStudio/total.svg?maxAge=2592001)](https://github.com/TagStudioDev/TagStudio/releases)

<p align="center">
  <img width="60%" src="docs/assets/github_header.png">
</p>

TagStudio offers a flexible, tag-based system for organizing your photos and files, avoiding proprietary formats and intrusive filesystem changes.  **Learn more and explore the documentation at [docs.tagstud.io](https://docs.tagstud.io)!**

> [!NOTE]
> We've migrated to an SQL database backend. Pre-existing libraries created with official TagStudio releases can be opened and migrated with the new v9.5+ releases!

> [!IMPORTANT]
> This project is in early development. Backups are crucial. TagStudio will not:
>
> -   Modify your files unless you explicitly use the "Delete File(s)" feature.
> -   Require you to recreate tags or libraries with new releases.
> -   Magically recall all your unseen images. You control the organization.

<p align="center">
  <img width="80%" src="docs/assets/screenshot.png" alt="TagStudio Screenshot">
</p>
<p align="center">
  <i>TagStudio Alpha v9.5.0 running on macOS Sequoia.</i>
</p>

## Key Features

*   **Tagging and Metadata:**
    *   Create custom, powerful tags with names, aliases, colors and parent/category relationships for organized searching
    *   Add custom metadata fields (name, author, description, notes) to your files.
    *   Copy and paste tags and fields across file entries
    *   Automatically organize tags into groups based on parent tags marked as "categories"
    *   Generate tags from your existing folder structure
*   **Powerful Search:**
    *   Search by tags, file paths, file types, and media types
    *   Use boolean operators (AND, OR, NOT) and parentheses for complex queries
    *   Use special search conditions (`special:untagged` and `special:empty`) to find file entries without tags or fields, respectively
*   **Flexible Library Management:**
    *   Create libraries/vaults centered around a system directory. Libraries contain a series of entries: the representations of your files combined with metadata fields. Each entry represents a file in your library’s directory, and is linked to its location.
    *   Automatically fix unlinked files
    *   Preview most image file types, animated GIFs, videos, plain text documents, audio files, Blender projects, and more!
    *   Open files or file locations by right-clicking
    *   Delete files from your library and drive via context menu
*   **Open Format & Future-Proof:** No proprietary formats or sidecar files, ensuring long-term access to your data.
*   **Cross-Platform:** Compatible with Windows, macOS (Apple Silicon & Intel), and Linux.

## Installation

Download the latest release from the [Releases](https://github.com/TagStudioDev/TagStudio/releases) page.  Install builds for Windows, macOS, and Linux, and also portable releases for Windows and Linux.

For detailed instructions and development instructions, see the [Installation](https://docs.tagstud.io/install/) page.

> [!CAUTION]
>  **Unofficial distributions of TagStudio are unsupported.**

### Dependencies

*   [FFmpeg](https://ffmpeg.org/download.html) is required for video thumbnails and playback. See the [FFmpeg Help](/docs/help/ffmpeg.md) guide for assistance.

## Contributing

Contribute to TagStudio by following the [contribution guidelines](/CONTRIBUTING.md).

Translation hosting provided by [Weblate](https://weblate.org/en/). Visit the [project page](https://hosted.weblate.org/projects/tagstudio/) to translate!

## FAQ

*   **Project Status:** TagStudio (Alpha v9.5.0) is functional, with planned features, bug fixes, and improvements detailed in the [Feature Roadmap](https://docs.tagstud.io/updates/roadmap/) and [milestones](https://github.com/TagStudioDev/TagStudio/milestones).
*   **Future Features:**  The [Feature Roadmap](https://docs.tagstud.io/updates/roadmap/) details planned core features.
*   **Features NOT Planned:**
    *   Native Cloud Integration (use existing services)
    *   Native ChatGPT/Non-Local LLM Integration (focus on local model integration)