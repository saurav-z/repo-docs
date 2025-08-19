# git-filter-repo: The Powerful Git History Rewriting Tool

**Tired of slow and error-prone history rewriting?**  `git-filter-repo` is the modern, efficient solution for rewriting Git history, recommended by the Git project itself. ([Original Repository](https://github.com/newren/git-filter-repo))

## Key Features

*   **High Performance:** Significantly faster than `git filter-branch`.
*   **Enhanced Capabilities:** Offers a wider range of history rewriting options beyond traditional tools.
*   **Safety First:** Designed to avoid common pitfalls and data corruption issues.
*   **User-Friendly:** Simple command-line interface for common tasks, with comprehensive documentation.
*   **Extensible:** Built as a library, enabling the creation of custom history rewriting tools.
*   **Advanced Features:**
    *   Keeps and removes specific paths in your repository.
    *   Easily renames files and directories.
    *   Rewrites commit messages to refer to new commit IDs.
    *   Handles pruning commits that become empty during the rewrite.
    *   Includes features for merging in extracted portions.
    *   Offers starting reports to help with analysis and optimization.

## Installation

`git-filter-repo` is a single-file Python script, making installation easy: simply place the `git-filter-repo` file in your system's `$PATH`.  See [INSTALL.md](INSTALL.md) for more complex scenarios.

**Prerequisites:**

*   Git >= 2.36.0
*   Python3 >= 3.6

## How to Use

*   For detailed instructions, consult the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html).
*   Learn by example with cheat sheets converting commands from `filter-branch` and `BFG Repo Cleaner`.
    *   [Cheat Sheet for Converting filter-branch Commands](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage)
    *   [Cheat Sheet for Converting BFG Repo Cleaner Commands](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg)
*   Explore the [examples section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES) in the user manual.
*   Refer to the [Frequently Answered Questions](Documentation/FAQ.md) for solutions to common issues.

## Why Choose git-filter-repo?

`git-filter-repo` excels where other tools fall short:

*   **Superior Speed:** Significantly faster than `git filter-branch`.
*   **Enhanced Safety:** Reduces the risk of data corruption.
*   **Greater Flexibility:** Supports a wider variety of history rewriting tasks.
*   **Recommended by Git:**  The Git project itself recommends using `git-filter-repo`.

## Simple Example

Extracting a directory, renaming files, and renaming tags with `git-filter-repo`:

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

## Contribute

Contribute to the project by following the [contributing guidelines](Documentation/Contributing.md).

## Code of Conduct

The project adheres to the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

`git-filter-repo` has spurred significant improvements to Git's core, including enhancements to `fast-export` and `fast-import`.

```