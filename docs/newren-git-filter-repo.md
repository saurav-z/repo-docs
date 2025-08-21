# git-filter-repo: The Powerful Git History Rewriting Tool

**Tired of slow, buggy, and limited git history rewriting?**  `git filter-repo` is the recommended alternative to `git filter-branch`, providing superior performance, more features, and a safer experience for complex Git history modifications.  [Explore the git-filter-repo repository](https://github.com/newren/git-filter-repo).

## Key Features

*   **Blazing Fast Performance:** Significantly faster than `git filter-branch` for complex rewrites.
*   **Comprehensive Functionality:** Offers capabilities not found in other tools, enabling advanced history manipulation.
*   **Safe and Reliable:** Designed to avoid the pitfalls and data corruption risks associated with `git filter-branch`.
*   **User-Friendly:** Provides a clear and intuitive command-line interface for common tasks.
*   **Extensible:** Built as a library for creating custom history rewriting tools.
*   **Path and Directory Renaming:** Easily rename or move files and directories within your repository.
*   **Automatic Cleanup:** Automatically removes unnecessary objects and repacks your repository for optimal performance.
*   **Commit Message Rewriting:**  Handles rewriting commit messages to refer to new commit IDs.
*   **Become-Empty and Become-Degenerate Pruning:** Automatically removes commits that become empty or degenerate due to filtering.

## Table of Contents

*   [Prerequisites](#prerequisites)
*   [How to Install](#how-to-install)
*   [How to Use](#how-to-use)
*   [Why Use `git filter-repo`?](#why-use-git-filter-repo)
*   [Simple Example: Extracting a Directory](#simple-example-extracting-a-directory)
*   [Design Rationale](#design-rationale)
*   [How to Contribute](#how-to-contribute)
*   [Code of Conduct](#is-there-a-code-of-conduct)
*   [Upstream Improvements](#upstream-improvements)

## Prerequisites

*   git >= 2.36.0
*   python3 >= 3.6

## How to Install

Installation is straightforward: simply place the single-file Python script `git-filter-repo` into your system's `$PATH`.  See [INSTALL.md](INSTALL.md) for advanced scenarios.

## How to Use

For detailed documentation, consult the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html).  You can also use the examples on the following pages for guidance:

*   [Cheat Sheet for converting filter-branch commands](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage)
*   [Cheat Sheet for converting BFG Repo Cleaner commands](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg)
*   [User Manual Examples Section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES)
*   [Examples from user-filed issues](Documentation/examples-from-user-filed-issues.md)

## Why Use `git filter-repo`?

`git filter-repo` surpasses other tools in the following ways:

*   **Superior Performance:** Significantly faster and more efficient than `git filter-branch`, especially for large repositories.
*   **Increased Safety:** Avoids data corruption and silent failures common with `git filter-branch`.
*   **Enhanced Capabilities:** Offers a wider range of features and flexibility compared to BFG Repo Cleaner.
*   **Git Project Recommendation:** `git-filter-repo` is now the recommended tool by the Git project for history rewriting, instead of `git filter-branch`.

## Simple Example: Extracting a Directory

To extract the history of the `src/` directory, rename files, and prefix tag names:

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

## Design Rationale

`git filter-repo` was built to address the limitations of existing tools and provide a more robust and versatile history rewriting solution. Key design principles include:

*   **Performance:** Fast operation for all common tasks.
*   **Safety:** Preventing data loss and ensuring consistent results.
*   **Completeness:** Capabilities not found in other tools
*   **User-Friendliness:** Easy to use and understand.
*   **Extensibility:** Openness to add new features.

## How to Contribute

Contribute to `git filter-repo` by following the [contributing guidelines](Documentation/Contributing.md).

## Code of Conduct

Adhere to the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md) when participating in the `git filter-repo` community.

## Upstream Improvements

`git filter-repo` has driven numerous improvements to core Git commands, including `fast-export` and `fast-import`.