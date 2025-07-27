# git-filter-repo: The Ultimate Git History Rewriting Tool

**git-filter-repo is the recommended tool for rewriting Git history, providing unparalleled performance and capabilities compared to alternatives.** ([Original Repo](https://github.com/newren/git-filter-repo))

## Key Features

*   **Superior Performance:** Significantly faster than `git filter-branch` for complex rewriting tasks.
*   **Extensive Functionality:** Offers advanced features for path filtering, renaming, and more.
*   **Safer Rewrites:** Designed to avoid common pitfalls and data corruption issues found in `git filter-branch`.
*   **User-Friendly:** Simple command-line interface for common use cases, with comprehensive documentation and examples.
*   **Extensible:**  Provides a library for creating custom history rewriting tools.
*   **Automatic Cleanup:** Automatically removes old objects and repacks the repository after filtering.
*   **Commit Message Rewriting:** Automatically updates commit messages to reflect new commit IDs.
*   **Comprehensive Support:** Includes cheat sheets for converting commands from `git filter-branch` and BFG Repo Cleaner.

## Installation

To install `git-filter-repo`, simply place the `git-filter-repo` Python script into your `$PATH`.  Refer to [INSTALL.md](INSTALL.md) for detailed installation instructions, especially for advanced usage.

## How to Use

For detailed usage instructions:

*   Consult the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html).
*   Explore the [cheat sheets](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) for example command conversions.
*   Review the extensive [examples](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES) in the user manual.
*   Check the [Frequently Answered Questions](Documentation/FAQ.md) for common problems.

## Why Choose `git-filter-repo`?

`git-filter-repo` excels as the preferred tool over `git filter-branch` and BFG Repo Cleaner. Here's why:

### `git filter-branch`

*   **Slow and Inefficient:** Significantly slower, especially for large or complex repositories.
*   **Safety Concerns:** Prone to silent data corruption and difficult to recover from.
*   **Difficult Usage:**  Command syntax is complex and error-prone for anything beyond simple tasks.

### BFG Repo Cleaner

*   **Limited Functionality:** Restricts to a limited number of rewriting scenarios.
*   **Architectural Limitations:**  Not designed to handle a wide array of rewrite operations.

## Simple Example: Extracting a Directory

Let's say you want to extract a directory named `src/` and rename it to `my-module/` while also renaming tags:

### Using `git-filter-repo`

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

## Contribution

See the [contributing guidelines](Documentation/Contributing.md).

## Code of Conduct

The project follows the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

Work on `git-filter-repo` has driven numerous improvements to fast-export and fast-import in core Git, improving performance and expanding the capabilities of core Git commands.  (See original README for a detailed list of commits.)