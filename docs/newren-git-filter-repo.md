# git-filter-repo: Rewrite Git History with Ease and Power

**Tired of slow and error-prone Git history rewriting?** `git-filter-repo` is a powerful and versatile tool, recommended by the Git project itself, for rewriting your Git history with unparalleled speed, safety, and flexibility. Learn more at the original repository: [https://github.com/newren/git-filter-repo](https://github.com/newren/git-filter-repo).

## Key Features

*   **Fast and Efficient:** Dramatically faster than `git filter-branch` for complex rewrites.
*   **Safe and Reliable:** Designed to avoid common pitfalls and data corruption issues.
*   **Highly Customizable:** Offers a wide array of options for complex history manipulations.
*   **Recommended by Git:** The Git project now recommends `git-filter-repo` over `git filter-branch`.
*   **Extensible Library:** Provides a Python library for creating custom history rewriting tools.

## Table of Contents

*   [Prerequisites](#prerequisites)
*   [How to Install](#how-do-i-install-it)
*   [How to Use](#how-do-i-use-it)
*   [Why Choose git-filter-repo?](#why-filter-repo-instead-of-other-alternatives)
*   [Simple Example: Path Extraction and Renaming](#simple-example-with-comparisons)
*   [Design Rationale](#design-rationale-behind-filter-repo)
*   [How to Contribute](#how-do-i-contribute)
*   [Code of Conduct](#is-there-a-code-of-conduct)
*   [Upstream Improvements](#upstream-improvements)

## Prerequisites

*   git >= 2.36.0
*   python3 >= 3.6

## How to Install

Installation is easy: simply place the single-file Python script `git-filter-repo` into your system's `$PATH`. Refer to [INSTALL.md](INSTALL.md) for more advanced installation needs, such as installing documentation or using with a Python executable other than "python3".

## How to Use

*   **Comprehensive Documentation:** Explore the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html) for detailed information.
*   **Alternative Formatting:** Consider alternative documentation sources like [mankier.com](https://www.mankier.com/1/git-filter-repo).
*   **Learn by Example:**
    *   Convert `filter-branch` commands with the [cheat sheet](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage).
    *   Convert `BFG Repo Cleaner` commands with the [cheat sheet](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg).
    *   See the [simple example](#simple-example-with-comparisons) below.
    *   Explore the [examples section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES) of the user manual.
    *   Review [example filterings based on user-filed issues](Documentation/examples-from-user-filed-issues.md).
*   **FAQ:** Find answers to common questions in the [FAQ](Documentation/FAQ.md).

## Why Choose git-filter-repo?

`git-filter-repo` offers significant advantages over existing tools like `git filter-branch` and `BFG Repo Cleaner`:

*   **`git filter-branch`:**  Significantly faster, more reliable, and easier to use, especially for complex rewrites, with the git project recommending against the use of `filter-branch` due to its inherent limitations.
*   **BFG Repo Cleaner:**  Offers more flexibility and capabilities than BFG, while offering reimplementations like `bfg-ish` built on top of filter-repo.

## Simple Example: Path Extraction and Renaming

Let's extract a directory and rename it:

*   Extract the history of `src/`.
*   Rename files to `my-module/`.
*   Rename tags to prefix with `my-module-`.

**Solution with `git-filter-repo`:**

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

This contrasts with the complexities and limitations of using `filter-branch` and  `fast-export/fast-import` or the inability of `BFG Repo Cleaner` to perform these operations.

## Design Rationale Behind `git-filter-repo`

`git-filter-repo` was created to address shortcomings in existing tools, focusing on:

*   **User Guidance:** Provides analysis to help understand and begin rewriting.
*   **Flexibility in Filtering:** Easily keep or remove specific paths.
*   **Powerful Renaming Capabilities:** Rename files and directories with checks for name conflicts.
*   **Enhanced Safety:** Enforces fresh clone workflows for safer operations.
*   **Automated Cleanup:** Automatically removes cruft and repacks the repository.
*   **Clean Separation:** Avoids mixing old and rewritten history.
*   **Extensibility:** Supports custom tools and shell-independent commands.
*   **Old Commit Reference Handling:** Map old commit IDs to new ones.
*   **Commit Message Rewriting:** Update commit messages to refer to the new IDs.
*   **Empty Commit Pruning:** Properly handles empty commits created by the filter.
*   **Degenerate Merge Pruning:** Improves the reliability of merge commit handling.
*   **Performance:** Designed for speed.

## How to Contribute

Contribute to `git-filter-repo` by following the [contributing guidelines](Documentation/Contributing.md).

## Code of Conduct

The `git-filter-repo` community adheres to the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

`git-filter-repo` development has driven numerous improvements to the core Git tools, including:

*   [List of commits](https://github.com/newren/git-filter-repo#upstream-improvements) impacting  `fast-export` and `fast-import`.