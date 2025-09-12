# TagStudio: Organize Your Files Your Way

**Tired of generic file management?**  TagStudio is a user-focused document management system that puts you in control with flexible tagging and a non-intrusive approach to file organization.  [Check out the original repository](https://github.com/TagStudioDev/TagStudio) for the latest updates.

[![Translation](https://hosted.weblate.org/widget/tagstudio/strings/svg-badge.svg)](https://hosted.weblate.org/projects/tagstudio/strings/)
[![PyTest](https://github.com/TagStudioDev/TagStudio/actions/workflows/pytest.yaml/badge.svg)](https://github.com/TagStudioDev/TagStudio/actions/workflows/pytest.yaml)
[![MyPy](https://github.com/TagStudioDev/TagStudio/actions/workflows/mypy.yaml/badge.svg)](https://github.com/TagStudioDev/TagStudio/actions/workflows/mypy.yaml)
[![Ruff](https://github.com/TagStudioDev/TagStudio/actions/workflows/ruff.yaml/badge.svg)](https://github.com/TagStudioDev/TagStudio/actions/workflows/ruff.yaml)
[![Downloads](https://img.shields.io/github/downloads/TagStudioDev/TagStudio/total.svg?maxAge=2592001)](https://github.com/TagStudioDev/TagStudio/releases)

<p align="center">
  <img width="60%" src="docs/assets/github_header.png">
</p>

TagStudio is designed for photo & file organization with a tag-based system that prioritizes user freedom and flexibility.  It avoids proprietary formats and sidecar files, letting you organize without disrupting your existing file structure.  **Learn more and explore the documentation at [docs.tagstud.io](https://docs.tagstud.io)!**

> [!NOTE]
> The database backend has been migrated from JSON to SQL, allowing for migration with pre-existing library save files created with official TagStudio releases!

> [!IMPORTANT]
> TagStudio is in early development. Backups are essential. TagStudio will NOT: touch, move, or change your files (unless explicitly deleting them); require you to recreate tags after updates; or suddenly recall your terabytes of images.

<p align="center">
  <img width="80%" src="docs/assets/screenshot.png" alt="TagStudio Screenshot">
</p>
<p align="center">
  <i>TagStudio Alpha v9.5.0 running on macOS Sequoia.</i>
</p>

## Key Features

*   **Tagging System:** Create custom tags with inheritance, aliases, and colors. Organize tags into groups.
*   **Custom Metadata:** Add file-specific metadata, including names, descriptions, and more.
*   **Flexible Search:** Find files using tags, file paths, file types, and media types with boolean operators and more.
*   **File Previews:** View previews for most major file types, including images, videos, and documents.
*   **File Management:** Delete files from your library and drive.
*   **No Filesystem Interference:** Your files are untouched unless you specifically use the 'Delete' function

## Installation

Download the latest releases for Windows, macOS (Apple Silicon & Intel), and Linux from the [Releases](https://github.com/TagStudioDev/TagStudio/releases) page.  Portable versions are also available. For detailed instructions, please see the "[Installation](https://docs.tagstud.io/install/)" page on our documentation website.

> [!CAUTION]
> Only use distributions from the GitHub Releases page.

## Contributing

Contribute to TagStudio!  See the [contribution guidelines](/CONTRIBUTING.md) for details.  Translation is hosted by [Weblate](https://weblate.org/en/) - [help translate TagStudio!](https://hosted.weblate.org/projects/tagstudio/)

## FAQ

### What is the current state of the project?

As of writing (Alpha v9.5.0) the project is very usable, however there's some plenty of quirks and missing QoL features. Several additional features and changes are still planned (see: [Feature Roadmap](https://docs.tagstud.io/updates/roadmap/)) that add even more power and flexibility to the tagging and field systems while making it easier to tag in bulk and perform automated operations. Bugfixes and polish are constantly trickling in along with the larger feature releases.

### What features are you planning on adding?

See the [Feature Roadmap](https://docs.tagstud.io/updates/roadmap/) page for the core features being planned and implemented for TagStudio. For a more up to date look on what's currently being added for upcoming releases, see our GitHub [milestones](https://github.com/TagStudioDev/TagStudio/milestones) for versioned releases.

### Features that will NOT be added

-   Native Cloud Integration
    -   There are plenty of services already (native or third-party) that allow you to mount your cloud drives as virtual drives on your system. Hosting a TagStudio library on one of these mounts should function similarly to what native integration would look like.
    -   Supporting native cloud integrations such as these would be an unnecessary "reinventing the wheel" burden for us that is outside the scope of this project.
-   Native ChatGPT/Non-Local LLM Integration
    -   This could mean different things depending on your intentions. Whether it's trying to use an LLM to replace the native search, or to trying to use a model for image recognition, I'm not interested in hooking people's TagStudio libraries into non-local LLMs such as ChatGPT and/or turn the program into a "chatbot" interface (see: [Goals/Privacy](#goals)). I wouldn't, however, mind using **locally** hosted models to provide the _optional_ ability for additional searching and tagging methods (especially when it comes to facial recognition) - but this would likely take the form of plugins external to the core program anyway.

### Why Is this Already Version 9?

Over the first few years of private development the project went through several major iterations and rewrites. These major version bumps came quickly, and by the time TagStudio was opened-sourced the version number had already reached v9.0. Instead of resetting to "v0.0" or "v1.0" for this public release I decided to keep my v9.x numbering scheme and reserve v10.0 for when all the core features on the [Feature Roadmap](https://docs.tagstud.io/updates/roadmap/) are implemented. I’ve also labeled this version as an "Alpha" and will drop this once either all of the core features are implemented or the project feels stable and feature-rich enough to be considered "Beta" and beyond.
```

Key improvements:

*   **SEO Optimization:**  Added a strong one-sentence hook, used relevant keywords ("file organization," "tagging system," etc.), and included headings for easy navigation and search engine indexing.
*   **Summarization:** Condensed the information while retaining key details. Removed redundant phrases.
*   **Formatting:**  Used bullet points for key features, improved heading structure, and maintained important notes/warnings.
*   **Clarity:**  Reworded sentences for better readability.
*   **Call to Action:**  Reinforced the link to the documentation.
*   **Focus:**  Removed some detailed information that wasn't critical to the initial overview, such as usage instructions for the other half-implemented features, and moved the less important details to the FAQ.