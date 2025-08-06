# git-filter-repo: The Modern and Powerful Git History Rewriting Tool

**Tired of slow and error-prone Git history modifications?**  `git-filter-repo` is the recommended alternative to `git filter-branch`, offering superior performance, advanced features, and a safer approach to rewriting your Git repository history. [See the original repo](https://github.com/newren/git-filter-repo).

## Key Features:

*   **High Performance:** Significantly faster than `git filter-branch` for complex rewriting tasks.
*   **Comprehensive Functionality:** Rewrites history with advanced options not found elsewhere.
*   **Safe and Reliable:** Designed to avoid common pitfalls and data corruption issues.
*   **Intuitive Interface:** Easier to use for both simple and complex rewriting scenarios.
*   **Flexible and Extensible:** Core library allows you to create your own custom history rewriting tools.
*   **Automated Cleanup:** Automatically shrinks and repacks your repository for optimal performance.
*   **Commit Message Rewriting:** Updates commit messages to reflect changes in the rewritten history.
*   **Path and Content Filtering:** Easily remove, rename, or modify files and directories.

## Prerequisites

*   git >= 2.36.0
*   python3 >= 3.6

## Installation

`git-filter-repo` is a single-file Python script, making installation straightforward: place the `git-filter-repo` file into your `$PATH`. See [INSTALL.md](INSTALL.md) for advanced setup and customization.

## How to Use

*   **User Manual:** Explore the comprehensive [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html) for detailed documentation and examples.
*   **Cheat Sheets:** Find cheat sheets for converting commands from `git filter-branch` and BFG Repo Cleaner:
    *   [filter-branch](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage)
    *   [BFG Repo Cleaner](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg)
*   **Examples:** Learn from example use cases in the user manual's [EXAMPLES section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES) and the [examples from user-filed issues](Documentation/examples-from-user-filed-issues.md).

## Why Choose `git-filter-repo`?

`git-filter-repo` addresses the performance, safety, and usability limitations of older tools like `git filter-branch` and BFG Repo Cleaner.

## Example: Extracting and Renaming a Directory

Let's say you want to extract the history of the `src/` directory, rename files to `my-module/src/`, and prefix tags with `my-module-`. `git-filter-repo` makes this simple:

```bash
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

Compare this to the complex and error-prone commands needed with `git filter-branch`.

## Contributing

Contribute to the project!  See the [contributing guidelines](Documentation/Contributing.md).

## Code of Conduct

The project adheres to the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

`git-filter-repo` has driven numerous improvements to Git's fast-export and fast-import commands, benefiting the entire Git ecosystem.