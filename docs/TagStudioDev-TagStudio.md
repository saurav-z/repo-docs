# TagStudio: Organize Your Files with Powerful Tagging (and without the headaches!)

Tired of endless folders and unorganized files? [TagStudio](https://github.com/TagStudioDev/TagStudio) offers a flexible, tag-based document management system that puts you in control.

## Key Features:

*   **Tag-Based Organization:** Categorize files using a flexible tagging system with tag inheritance, aliases, and categories.
*   **No File Changes:** TagStudio *never* touches, moves, or renames your original files.
*   **Customizable Metadata:** Add custom metadata like descriptions, authors, and notes to your entries.
*   **Powerful Search:** Easily find files using tags, paths, file types, and Boolean operators.
*   **Cross-Platform Support:** Available for Windows, macOS (including Apple Silicon), and Linux.

## Core Tenets

1.  **The Idea:** Even if TagStudio as an application fails, the idea of a portable, private, extensible, open-format, and feature-rich system of organizing and rediscovering files lives on.
2.  **The System:** A core underlying metadata management system should be interoperable between different frontends, programs, and operating systems.
3.  **The Application:** TagStudio the application serves as the first (and so far only) implementation for this system of metadata management.

## Current Features

### Libraries

*   Create libraries/vaults centered around a system directory. Libraries contain a series of entries: the representations of your files combined with metadata fields. Each entry represents a file in your library’s directory, and is linked to its location.
*   Address moved, deleted, or otherwise "unlinked" files by using the "Fix Unlinked Entries" option in the Tools menu.

### Tagging + Custom Metadata

*   Add custom powerful tags to your library entries
*   Add metadata to your library entries, including:
    *   Name, Author, Artist (Single-Line Text Fields)
    *   Description, Notes (Multiline Text Fields)
*   Create rich tags composed of a name, color, a list of aliases, and a list of "parent tags" - these being tags in which these tags inherit values from.
*   Copy and paste tags and fields across file entries
*   Automatically organize tags into groups based on parent tags marked as "categories"
*   Generate tags from your existing folder structure with the "Folders to Tags" macro (NOTE: these tags do NOT sync with folders after they are created)

### Search

*   Search for file entries based on tags, file path (`path:`), file types (`filetype:`), and even media types! (`mediatype:`). Path searches currently use [glob](<https://en.wikipedia.org/wiki/Glob_(programming)>) syntax, so you may need to wrap your filename or filepath in asterisks while searching. This will not be strictly necessary in future versions of the program.
*   Use and combine boolean operators (`AND`, `OR`, `NOT`) along with parentheses groups, quotation escaping, and underscore substitution to create detailed search queries
*   Use special search conditions (`special:untagged` and `special:empty`) to find file entries without tags or fields, respectively

### File Entries

*   Nearly all file types are supported in TagStudio libraries - just not all have dedicated thumbnail support.
*   Preview most image file types, animated GIFs, videos, plain text documents, audio files, Blender projects, and more!
*   Open files or file locations by right-clicking on thumbnails and previews and selecting the respective context menu options. You can also click on the preview panel image to open the file, and click the file path label to open its location.
*   Delete files from both your library and drive by right-clicking the thumbnail(s) and selecting the "Move to Trash"/"Move to Recycle Bin" option.

## Installation

Download the latest builds from the [Releases](https://github.com/TagStudioDev/TagStudio/releases) page. Detailed installation instructions can be found in the [documentation](https://docs.tagstud.io/install/).

*   **Windows**: Download from Releases, or portable releases
*   **macOS**: Download from Releases
*   **Linux**: Download from Releases, or portable releases

> **Note:** TagStudio is not yet available on package managers. Use only official releases for the most stable experience.

## Contributing

Help translate TagStudio via [Weblate](https://hosted.weblate.org/projects/tagstudio/) and contribute to the project by following the [contribution guidelines](/CONTRIBUTING.md).

## Additional Resources

*   [Documentation](https://docs.tagstud.io/)
*   [FAQ](#faq)
*   [Feature Roadmap](https://docs.tagstud.io/updates/roadmap/)