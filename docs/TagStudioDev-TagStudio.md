# TagStudio: Organize Your Files with Powerful Tagging and Metadata

**Tired of disorganized files? TagStudio is a user-focused document management system that puts you in control.** [Explore TagStudio on GitHub](https://github.com/TagStudioDev/TagStudio).

[![Translation](https://hosted.weblate.org/widget/tagstudio/strings/svg-badge.svg)](https://hosted.weblate.org/projects/tagstudio/strings/)
[![PyTest](https://github.com/TagStudioDev/TagStudio/actions/workflows/pytest.yaml/badge.svg)](https://github.com/TagStudioDev/TagStudio/actions/workflows/pytest.yaml)
[![MyPy](https://github.com/TagStudioDev/TagStudio/actions/workflows/mypy.yaml/badge.svg)](https://github.com/TagStudioDev/TagStudio/actions/workflows/mypy.yaml)
[![Ruff](https://github.com/TagStudioDev/TagStudio/actions/workflows/ruff.yaml/badge.svg)](https://github.com/TagStudioDev/TagStudio/actions/workflows/ruff.yaml)
[![Downloads](https://img.shields.io/github/downloads/TagStudioDev/TagStudio/total.svg?maxAge=2592001)](https://github.com/TagStudioDev/TagStudio/releases)

<p align="center">
  <img width="60%" src="docs/assets/github_header.png" alt="TagStudio Header">
</p>

TagStudio offers a flexible, tag-based system for organizing photos and files, avoiding proprietary formats and intrusive file structure changes.  **Learn more about TagStudio and its features at [docs.tagstud.io](https://docs.tagstud.io)!**

> [!NOTE]
> Thank you for your patience as we've migrated our database backend from JSON to SQL!

> [!IMPORTANT]
> This project is still in an early state. Making frequent backups of your library save data is **always** important.

<p align="center">
  <img width="80%" src="docs/assets/screenshot.png" alt="TagStudio Screenshot">
</p>
<p align="center">
  <i>TagStudio Alpha v9.5.0 running on macOS Sequoia.</i>
</p>

## Key Features

*   **Flexible Tagging:** Create custom tags with names, aliases, colors, and parent tags for powerful organization.
*   **Metadata Management:** Add and edit metadata like titles, descriptions, and notes to your files.
*   **Advanced Search:** Find files quickly using tag, path, file type, and media type searches with boolean operators and advanced syntax.
*   **File Preview:** Preview image files, videos, documents, and more directly within TagStudio.
*   **Non-Destructive:**  TagStudio *never* modifies your original files unless you explicitly choose to delete them.

## Table of Contents

*   [Goals](#goals)
*   [Priorities](#priorities)
*   [Current Features](#current-features)
*   [Contributing](#contributing)
*   [Installation](#installation)
*   [Usage](#usage)
*   [FAQ](#faq)

## Goals

*   To achieve a portable, private, extensible, open-format, and feature-rich system of organizing and rediscovering files.
*   To provide powerful methods for organization, notably the concept of tag inheritance, or "taggable tags" _(and in the near future, the combination of composition-based tags)._
*   To create an implementation of such a system that is resilient against a user’s actions outside the program (modifying, moving, or renaming files) while also not burdening the user with mandatory sidecar files or requiring them to change their existing file structures and workflows.
*   To support a wide range of users spanning across different platforms, multi-user setups, and those with large (several terabyte) libraries.
*   To make the dang thing look nice, too. It’s 2025, not 1995.

## Priorities

1.  **The Concept:** The core idea of TagStudio is paramount.
2.  **The System:**  The underlying metadata management system should be interoperable.
3.  **The Application:** TagStudio itself serves as a primary implementation of this system.
4.  (The name.) I think it’s fine for an app or client, but it doesn’t really make sense for a system or standard. I suppose this will evolve with time...

## Contributing

Contribute to TagStudio! See the [contribution guidelines](/CONTRIBUTING.md) for more information.

Help translate TagStudio on [Weblate](https://hosted.weblate.org/projects/tagstudio/).

## Current Features

### Libraries

*   Create and manage libraries centered around a system directory.
*   Manage unlinked files using the "Fix Unlinked Entries" option.

### Tagging + Custom Metadata

*   Add custom tags to your library entries
*   Add metadata to your library entries, including:
    *   Name, Author, Artist (Single-Line Text Fields)
    *   Description, Notes (Multiline Text Fields)
*   Create rich tags composed of a name, color, a list of aliases, and a list of "parent tags" - these being tags in which these tags inherit values from.
*   Copy and paste tags and fields across file entries
*   Automatically organize tags into groups based on parent tags marked as "categories"
*   Generate tags from your existing folder structure with the "Folders to Tags" macro (NOTE: these tags do NOT sync with folders after they are created)

### Search

*   Search based on tags, file path, file types, and media types.
*   Use boolean operators and parenthesis to create detailed search queries.
*   Use special search conditions (`special:untagged` and `special:empty`) to find file entries without tags or fields, respectively

### File Entries

*   Preview most file types.
*   Open files or file locations with context menu options.
*   Move files to trash/recycle bin.

> [!NOTE]
> For more information, consult the [FAQ](#faq) and the [documentation](https://docs.tagstud.io/)!

## Installation

Download the latest release from the [Releases](https://github.com/TagStudioDev/TagStudio/releases) page.

Available for **Windows**, **macOS** _(Apple Silicon & Intel)_, and **Linux**.  Also offers portable releases.

See the "[Installation](https://docs.tagstud.io/install/)" page on our documentation website for detailed instructions.

<!-- prettier-ignore -->
> [!CAUTION]
> **Only download from the GitHub [Releases](https://github.com/TagStudioDev/TagStudio/releases) page. Unofficial distributions are unsupported.**

### Third-Party Dependencies

Install [FFmpeg](https://ffmpeg.org/download.html) for video thumbnails and playback.  See our [FFmpeg Help](/docs/help/ffmpeg.md) guide.

## Usage

### Creating/Opening a Library

Use `File -> Open/Create Library`. TagStudio automatically scans the chosen directory.

### Refreshing the Library

Libraries under 10,000 files automatically scan for new or modified files when opened. In order to refresh the library manually, select "Refresh Directories" under the File menu.

### Adding Tags to File Entries

Access the "Add Tag" search box by either clicking on the "Add Tag" button at the bottom of the right sidebar, accessing the "Add Tags to Selected" option from the File menu, or by pressing <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>T</kbd>.

From here you can search for existing tags or create a new one if the one you're looking for doesn't exist. Click the "+" button next to any tags you want to to the currently selected file entries. To quickly add the top result, press the <kbd>Enter</kbd>/<kbd>Return</kbd> key to add the the topmost tag and reset the tag search. Press <kbd>Enter</kbd>/<kbd>Return</kbd> once more to close the dialog box. By using this method, you can quickly add various tags in quick succession just by using the keyboard!

To remove a tag from a file entry, hover over the tag in the preview panel and click on the "-" icon that appears.

### Adding Metadata to File Entries

To add a metadata field to a file entry, start by clicking the "Add Field" button at the bottom of the preview panel. From the dropdown menu, select the type of metadata field you’d like to add to the entry

### Editing Metadata Fields

#### Text Line / Text Box

Hover over the field and click the pencil icon. From there, add or edit text in the dialog box popup.

> [!WARNING]
> Keyboard control and navigation is currently _very_ buggy, but will be improved in future versions.

### Creating Tags

Create a new tag by accessing the "New Tag" option from the Edit menu or by pressing <kbd>Ctrl</kbd>+<kbd>T</kbd>. In the tag creation panel, enter a tag name, optional shorthand name, optional tag aliases, optional parent tags, and an optional color.

-   The tag **name** is the base name of the tag. **_This does NOT have to be unique!_**
-   The tag **shorthand** is a special type of alias that displays in situations where screen space is more valuable, notably with name disambiguation.
-   **Aliases** are alternate names for a tag. These let you search for terms other than the exact tag name in order to find the tag again.
-   **Parent Tags** are tags in which this this tag can substitute for in searches. In other words, tags under this section are parents of this tag.
    -   Parent tags with the disambiguation check next to them will be used to help disambiguate tag names that may not be unique.
    -   For example: If you had a tag for "Freddy Fazbear", you might add "Five Nights at Freddy's" as one of the parent tags. If the disambiguation box is checked next to "Five Nights at Freddy's" parent tag, then the tag "Freddy Fazbear" will display as "Freddy Fazbear (Five Nights at Freddy's)". Furthermore, if the "Five Nights at Freddy's" tag has a shorthand like "FNAF", then the "Freddy Fazbear" tag will display as "Freddy Fazbear (FNAF)".
-   The **color** option lets you select an optional color palette to use for your tag.
-   The **"Is Category"** property lets you treat this tag as a category under which itself and any child tags inheriting from it will be sorted by inside the preview panel.

#### Tag Manager

You can manage your library of tags from opening the "Tag Manager" panel from Edit -> "Tag Manager". From here you can create, search for, edit, and permanently delete any tags you've created in your library.

### Editing Tags

To edit a tag, click on it inside the preview panel or right-click the tag and select "Edit Tag" from the context menu.

### Relinking Moved Files

Use the "Manage Unlinked Entries" option under the Tools menu to relink or delete moved files.

> [!WARNING]
> Relinking entries to renamed files is not yet supported.

> [!WARNING]
> If multiple matches for a moved file are found (matches are currently defined as files with a matching filename as the original), TagStudio will currently ignore the match groups. Adding a GUI for manual selection, as well as smarter automated relinking, are high priorities for future versions.

### Saving the Library

Libraries are saved automatically. Backups can be created using `File -> Save Library Backup`.

### Half-Implemented Features

These features were present in pre-public versions of TagStudio (9.0 and below) and have yet to be fully implemented.

#### Fix Duplicate Files

Load in a .dupeguru file generated by [dupeGuru](https://github.com/arsenetar/dupeguru/) and mirror metadata across entries marked as duplicates. After mirroring, return to dupeGuru to manage deletion of the duplicate files. After deletion, use the "Fix Unlinked Entries" feature in TagStudio to delete the duplicate set of entries for the now-deleted files

> [!CAUTION]
> While this feature is functional, it’s a pretty roundabout process and can be streamlined in the future.

#### Macros

Apply tags and other metadata automatically depending on certain criteria. Set specific macros to run when the files are added to the library. Part of this includes applying tags automatically based on parent folders.

> [!CAUTION]
> Macro options are hardcoded, and there’s currently no way for the user to interface with this (still incomplete) system at all.

#### Gallery-dl Sidecar Importing

Import JSON sidecar data generated by [gallery-dl](https://github.com/mikf/gallery-dl).

> [!CAUTION]
> This feature is not supported or documented in any official capacity whatsoever. It will likely be rolled-in to a larger and more generalized sidecar importing feature in the future.

## Launching/Building From Source

See instructions in the "[Creating Development Environment](/CONTRIBUTING.md/#creating-a-development-environment)" section from the [contribution documentation](/CONTRIBUTING.md).

## FAQ

### What State Is the Project Currently In?

As of writing (Alpha v9.5.0) the project is very usable, however there's some plenty of quirks and missing QoL features. Several additional features and changes are still planned (see: [Feature Roadmap](https://docs.tagstud.io/updates/roadmap/)) that add even more power and flexibility to the tagging and field systems while making it easier to tag in bulk and perform automated operations. Bugfixes and polish are constantly trickling in along with the larger feature releases.

### What Features Are You Planning on Adding?

See the [Feature Roadmap](https://docs.tagstud.io/updates/roadmap/) page for the core features being planned and implemented for TagStudio. For a more up to date look on what's currently being added for upcoming releases, see our GitHub [milestones](https://github.com/TagStudioDev/TagStudio/milestones) for versioned releases.

### Features That Will NOT Be Added

-   Native Cloud Integration
    -   There are plenty of services already (native or third-party) that allow you to mount your cloud drives as virtual drives on your system. Hosting a TagStudio library on one of these mounts should function similarly to what native integration would look like.
    -   Supporting native cloud integrations such as these would be an unnecessary "reinventing the wheel" burden for us that is outside the scope of this project.
-   Native ChatGPT/Non-Local LLM Integration
    -   This could mean different things depending on your intentions. Whether it's trying to use an LLM to replace the native search, or to trying to use a model for image recognition, I'm not interested in hooking people's TagStudio libraries into non-local LLMs such as ChatGPT and/or turn the program into a "chatbot" interface (see: [Goals/Privacy](#goals)). I wouldn't, however, mind using **locally** hosted models to provide the _optional_ ability for additional searching and tagging methods (especially when it comes to facial recognition) - but this would likely take the form of plugins external to the core program anyway.

### Why Is this Already Version 9?

Over the first few years of private development the project went through several major iterations and rewrites. These major version bumps came quickly, and by the time TagStudio was opened-sourced the version number had already reached v9.0. Instead of resetting to "v0.0" or "v1.0" for this public release I decided to keep my v9.x numbering scheme and reserve v10.0 for when all the core features on the [Feature Roadmap](https://docs.tagstud.io/updates/roadmap/) are implemented. I’ve also labeled this version as an "Alpha" and will drop this once either all of the core features are implemented or the project feels stable and feature-rich enough to be considered "Beta" and beyond.