# Git Filter-Repo: The Modern Way to Rewrite Git History

Tired of slow and unreliable history rewriting tools? **Git filter-repo** offers a fast, feature-rich, and safe alternative to `git filter-branch`. [Learn more about git filter-repo here](https://github.com/newren/git-filter-repo).

## Key Features

*   **High Performance:** Significantly faster than `git filter-branch`.
*   **Comprehensive Functionality:** More capabilities than alternatives like `git filter-branch` and BFG Repo Cleaner.
*   **User-Friendly:** Designed for usability, even for complex rewriting scenarios.
*   **Safe by Design:** Minimizes the risk of data corruption compared to older tools.
*   **Extensible:** Provides a library for creating custom history rewriting tools.

## Installation

To quickly get started, simply place the single-file python script `git-filter-repo` into your `$PATH`. For more advanced installations or specific requirements, consult the [INSTALL.md](INSTALL.md) file.

## How to Use

*   **User Manual:** For detailed documentation and examples, explore the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html).
*   **Conversion Cheat Sheets:** Simplify the transition from other tools with cheat sheets for converting [filter-branch commands](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) and [BFG Repo Cleaner commands](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg).
*   **Examples:** Explore the [examples section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES) in the user manual or consult the [examples based on user-filed issues](Documentation/examples-from-user-filed-issues.md).

## Why Choose Git Filter-Repo?

Git filter-repo addresses the shortcomings of older tools:

*   **Performance:** Filter-branch is notoriously slow, while filter-repo is optimized for speed.
*   **Safety:** Filter-branch's potential for data corruption is a serious concern, whereas filter-repo is designed to be safer.
*   **Usability:** Filter-repo simplifies complex rewriting tasks, which are often difficult or impossible with other tools.

### Comparison

Filter-repo excels where `git filter-branch` and BFG Repo Cleaner fall short, providing a more intuitive and reliable solution.

## Simple Example: Extracting and Renaming a Directory

Suppose you want to extract a `src/` directory from your repository, rename it to `my-module/`, and prefix tags with `my-module-`.

**With git filter-repo:**

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

**Contrast this with:**

*   The complex and error-prone commands needed with `git filter-branch`.
*   The limitations of BFG Repo Cleaner, which cannot handle this type of rewrite.

## Contribute

Contribute to the project by following the [contributing guidelines](Documentation/Contributing.md).

## Code of Conduct

The project follows the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

Git filter-repo has driven significant improvements in core Git functionalities like fast-export and fast-import.  See the original README for a comprehensive list.