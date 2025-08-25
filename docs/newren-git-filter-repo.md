# git-filter-repo: The Powerful Git History Rewriting Tool

**Tired of slow, error-prone git history rewrites?** `git-filter-repo` is a fast, flexible, and safe tool for rewriting Git history, now recommended by the Git project itself.  [Learn more about it on GitHub](https://github.com/newren/git-filter-repo).

## Key Features

*   **Speed:** Significantly faster than `git filter-branch`.
*   **Safety:** Designed to avoid common pitfalls and data corruption.
*   **Flexibility:** Offers a wide range of rewriting options, including:
    *   Path-based filtering (keeping or removing files and directories).
    *   Subdirectory extraction and renaming.
    *   Tag renaming and prefixing.
    *   Commit message rewriting.
*   **Extensibility:**  Can be used as a library to build custom history rewriting tools.
*   **Comprehensive Documentation:** User manual, cheat sheets, and examples to get you started quickly.
*   **Upstream Improvements:**  Contributed significantly to improvements in core Git commands like `fast-export` and `fast-import`.

## Table of Contents

  * [Prerequisites](#prerequisites)
  * [How do I install it?](#how-do-i-install-it)
  * [How do I use it?](#how-do-i-use-it)
  * [Why filter-repo instead of other alternatives?](#why-filter-repo-instead-of-other-alternatives)
    * [filter-branch](#filter-branch)
    * [BFG Repo Cleaner](#bfg-repo-cleaner)
  * [Simple example, with comparisons](#simple-example-with-comparisons)
    * [Solving this with filter-repo](#solving-this-with-filter-repo)
    * [Solving this with BFG Repo Cleaner](#solving-this-with-bfg-repo-cleaner)
    * [Solving this with filter-branch](#solving-this-with-filter-branch)
    * [Solving this with fast-export/fast-import](#solving-this-with-fast-exportfast-import)
  * [Design rationale behind filter-repo](#design-rationale-behind-filter-repo)
  * [How do I contribute?](#how-do-i-contribute)
  * [Is there a Code of Conduct?](#is-there-a-code-of-conduct)
  * [Upstream Improvements](#upstream-improvements)

## Prerequisites

*   git >= 2.36.0
*   python3 >= 3.6

## How do I install it?

Installation is easy! Simply place the `git-filter-repo` Python script into your `$PATH`. More detailed instructions are available in [INSTALL.md](INSTALL.md) for advanced use cases.

## How do I use it?

Explore the comprehensive documentation:

*   [User Manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html)
*   Alternative Formatting: Available on external sites like [mankier.com](https://www.mankier.com/1/git-filter-repo).

Get started with examples:

*   [Cheat sheet for converting filter-branch commands](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage)
*   [Cheat sheet for converting BFG Repo Cleaner commands](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg)
*   [Simple Example](#simple-example-with-comparisons)
*   [Extensive examples section in the User Manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES)
*   [Examples based on user-filed issues](Documentation/examples-from-user-filed-issues.md)

Also, check the [Frequently Answered Questions](Documentation/FAQ.md).

## Why filter-repo instead of other alternatives?

`git-filter-repo` addresses the limitations of older tools:

### filter-branch

*   Slow and prone to errors.
*   Difficult to use for complex rewrites.
*   The Git project recommends against using it.
*   [filter-lamely](contrib/filter-repo-demos/filter-lamely)  offers a reimplementation based on filter-repo.
*   [Cheat Sheet](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) for command conversion.

### BFG Repo Cleaner

*   Limited to specific rewrite types.
*   Lacks flexibility for advanced use cases.
*   [bfg-ish](contrib/filter-repo-demos/bfg-ish)  is a filter-repo-based reimplementation with added features and bug fixes.
*   [Cheat Sheet](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg) for command conversion.

## Simple example, with comparisons

**Scenario:** Extract a directory ("src/") to a new repository with a prefix ("my-module/"), renaming tags.

### Solving this with filter-repo

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

### Solving this with BFG Repo Cleaner

BFG Repo Cleaner cannot perform this kind of rewrite.

### Solving this with filter-branch

filter-branch solution is significantly more complex and error-prone.  See original README for full example, including its caveats.

### Solving this with fast-export/fast-import

fast-export/fast-import is possible, but complex, with numerous caveats and limitations including risk of data corruption and lack of clean up features. See original README for full example, including its caveats.

## Design rationale behind filter-repo

`git-filter-repo` was created to address shortcomings in existing tools, including:

1.  Analysis reports
2.  Keeping vs. removing
3.  Renaming
4.  Improved Safety
5.  Automatic repository shrinking
6.  Clean separation of old and new repositories
7.  Extensibility
8.  Old commit ID references
9.  Commit message consistency
10. Become-empty pruning
11. Become-degenerate pruning
12. Speed

## How do I contribute?

See the [contributing guidelines](Documentation/Contributing.md).

## Is there a Code of Conduct?

Follow the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

`git-filter-repo` has driven significant improvements in core Git, particularly in `fast-export` and `fast-import`.  Refer to the original README for a list of Git commit improvements.