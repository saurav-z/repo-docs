# git-filter-repo: The Powerful Git History Rewriting Tool

Tired of slow and error-prone history rewriting? **git-filter-repo** is a modern, high-performance alternative to `git filter-branch`, recommended by the Git project itself, offering a vast array of capabilities for rewriting Git history with ease and accuracy. [Learn more at the official repository](https://github.com/newren/git-filter-repo).

**Key Features:**

*   **Fast and Efficient:** Significantly faster than `git filter-branch` for complex rewriting tasks.
*   **Comprehensive:** Offers a wide range of options for path filtering, renaming, tag manipulation, and more.
*   **Safe and Reliable:** Designed to avoid the common pitfalls and data corruption issues found in `git filter-branch`.
*   **Extensible:**  Provides a library for creating custom history rewriting tools.
*   **User-Friendly:** Simplifies complex operations with intuitive command-line arguments and clear documentation.
*   **Rewrites Commit Messages:** Automatically updates commit messages to reflect changes.
*   **Handles Empty Commits:** Correctly prunes commits that become empty due to filtering without losing intended versions.
*   **Upstream Improvements:** Continuously drives improvements in core Git through the development process.

## Table of Contents

*   [Prerequisites](#prerequisites)
*   [How do I install it?](#how-do-i-install-it)
*   [How do I use it?](#how-do-i-use-it)
*   [Why filter-repo instead of other alternatives?](#why-filter-repo-instead-of-other-alternatives)
    *   [filter-branch](#filter-branch)
    *   [BFG Repo Cleaner](#bfg-repo-cleaner)
*   [Simple Example, with comparisons](#simple-example-with-comparisons)
*   [Design rationale behind filter-repo](#design-rationale-behind-filter-repo)
*   [How do I contribute?](#how-do-i-contribute)
*   [Is there a Code of Conduct?](#is-there-a-code-of-conduct)
*   [Upstream Improvements](#upstream-improvements)

## Prerequisites

*   git >= 2.36.0
*   python3 >= 3.6

## How do I install it?

Installation is simple:
1.  Place the `git-filter-repo` Python script into your `$PATH`.

See [INSTALL.md](INSTALL.md) for more advanced instructions.

## How do I use it?

*   **User Manual:** Refer to the detailed [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html).
*   **Cheat Sheets:**  Utilize [cheat sheets](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) to convert `filter-branch` commands. Also see [cheat sheets](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg)
*   **Examples:** Explore various examples in the [user manual examples section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES) and [examples based on user-filed issues](Documentation/examples-from-user-filed-issues.md).
*   **FAQ:** Consult the [Frequently Answered Questions](Documentation/FAQ.md).

## Why filter-repo instead of other alternatives?

See details in the [Git Rev News article](https://git.github.io/rev_news/2019/08/21/edition-54/#an-introduction-to-git-filter-repo--written-by-elijah-newren)

### filter-branch

*   **Slow Performance:** Significantly slower than `git filter-repo`, especially for non-trivial repositories.
*   **Safety Issues:** Prone to silently corrupting your rewrite efforts.
*   **Difficult to Use:** Complicated to use for anything beyond simple rewrites.

### BFG Repo Cleaner

*   **Limited Functionality:** Only handles a limited set of rewrite types.
*   **Architecture Limitations:** Not designed for more complex rewriting scenarios.

## Simple example, with comparisons

*   **Goal:** Extract a specific directory (`src/`) into a new module with a new prefix (`my-module-`)
*   **git-filter-repo:**
    ```shell
    git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
    ```

## Design rationale behind filter-repo

*   **Comprehensive Features:** Offers advanced features to address the deficiencies of existing tools.
*   **Safer Workflow:** Encourages using a fresh clone to prevent any issues.
*   **Extensibility:** Designed to allow users to extend its capabilities.
*   **Commit Message Consistency:** Correctly updates commit messages after a rewrite.
*   **Optimize Commit handling:** Correctly handle empty commits and merge commits.
*   **Optimize for speed** High performance

## How do I contribute?

See the [contributing guidelines](Documentation/Contributing.md).

## Is there a Code of Conduct?

The project adheres to the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

filter-repo has driven numerous improvements to core Git.