# git-filter-repo: The Ultimate Git History Rewriting Tool

**Tired of slow and error-prone git history rewriting?**  [git-filter-repo](https://github.com/newren/git-filter-repo) is a powerful and efficient tool, recommended by the Git project, designed to rewrite your Git repository's history safely and effectively.

**Key Features:**

*   **Performance:** Significantly faster than `git filter-branch`, especially for large repositories.
*   **Safety:**  Designed to avoid the pitfalls and potential data corruption of `git filter-branch`.
*   **Versatility:** Handles a wide range of history rewriting tasks, including path filtering, renaming, and more.
*   **Extensibility:**  Provides a library for creating custom history rewriting tools.
*   **Comprehensive:** Offers a wide array of features and options not found in other tools.
*   **Commit Message Rewriting:** Automatically updates commit messages to reflect changes in commit IDs.
*   **Become-Empty and Become-Degenerate Pruning:** Efficiently handles empty or degenerate commits.
*   **Automated Repository Shrinking:** Streamlines the process by automatically removing old cruft and repacking the repository.
*   **Fresh Clone Enforcement:**  Encourages a safe workflow by operating on a fresh clone unless overridden.
*   **Upstream Improvements:** Directly contributes to core Git improvements, making the entire ecosystem better.

## Table of Contents

*   [Prerequisites](#prerequisites)
*   [How to Install](#how-to-install)
*   [How to Use](#how-to-use)
*   [Why Choose git-filter-repo?](#why-choose-git-filter-repo)
*   [Simple Example with Comparisons](#simple-example-with-comparisons)
*   [Design Rationale](#design-rationale-behind-filter-repo)
*   [How to Contribute](#how-to-contribute)
*   [Code of Conduct](#code-of-conduct)
*   [Upstream Improvements](#upstream-improvements)

## Prerequisites

*   git >= 2.36.0
*   python3 >= 3.6

## How to Install

Installation is straightforward: simply place the single-file `git-filter-repo` script into your system's `$PATH`.  See [INSTALL.md](INSTALL.md) for more detailed instructions for advanced use cases, such as documentation installation or running contributed examples.

## How to Use

*   **User Manual:** Consult the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html) for comprehensive documentation.
*   **Cheat Sheets:**  Quickly convert commands from `filter-branch` and BFG Repo Cleaner using the conversion [cheat sheets](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) and [cheat sheets](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg).
*   **Examples:** Explore example use cases in the [examples section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES) of the user manual.

## Why Choose git-filter-repo?

`git-filter-repo` surpasses alternatives like `git filter-branch` and BFG Repo Cleaner by offering superior performance, enhanced safety, and a broader range of capabilities. It's the recommended solution for modern Git history rewriting.  Specifically, it avoids the performance and safety issues of `filter-branch` and provides an architecture that is more flexible than BFG Repo Cleaner.

## Simple Example with Comparisons

This example demonstrates extracting a directory and renaming files, showcasing the simplicity of `git-filter-repo` compared to other tools.

**Scenario:** Extract the `src/` directory, rename it to `my-module/src/`, and prefix tags with `my-module-`.

**Solving this with git-filter-repo:**
```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

**Comparison to `filter-branch`, BFG Repo Cleaner, and `fast-export/fast-import`:**  See the original README for detailed comparisons.  `filter-repo` accomplishes this with a single command, whereas the other options are either incapable (BFG), or require multiple complex commands and are much more prone to errors (filter-branch & fast-export/fast-import).

## Design Rationale behind filter-repo

`git-filter-repo` was created to address the limitations of existing tools, aiming for:

1.  **Repository Analysis:**  Provides repo analysis tools to help get started.
2.  **Path Keeping:** Offer flags to *keep* specific paths, and not have to list everything.
3.  **Path Renaming:** Facilitate easy file and directory renaming with sanity checks.
4.  **Safety First:** Encourage the use of fresh clones for safe rewriting.
5.  **Automated Cleanup:** Automatically shrink and repack the repo.
6.  **Avoid Mixing:** Prevent confusion caused by mixing old and rewritten history.
7.  **Extensibility:**  Allow users to extend the tool without performance issues.
8.  **Commit References:** Allow the repository to use the old commit ID's with the new repo.
9.  **Commit Message Rewriting:** Rewrite messages to reference new commit IDs.
10. **Become-Empty Pruning:** Efficiently prune commits that become empty after filtering.
11. **Become-Degenerate Pruning:** Prune merge commits that become degenerate.
12. **Speed:**  Ensure reasonably fast filtering operations.

## How to Contribute

Contribute to the project by following the [contributing guidelines](Documentation/Contributing.md).

## Code of Conduct

Adhere to the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

`git-filter-repo` has driven numerous improvements to core Git commands like `fast-export` and `fast-import`. See the original README for a list of the specific Git commits.