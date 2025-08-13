# git-filter-repo: The Modern Tool for Rewriting Git History

git-filter-repo is a powerful and efficient tool for rewriting and cleaning up Git repository history, offering capabilities not found in other tools while being recommended by the Git project itself. [Explore the original repository](https://github.com/newren/git-filter-repo).

## Key Features

*   **High Performance:** Significantly faster than `git filter-branch` for complex rewrites.
*   **Enhanced Functionality:** Offers a wide range of features for history manipulation, including path filtering, renaming, and more.
*   **User-Friendly:** Simplifies history rewriting tasks with intuitive commands and a comprehensive user manual.
*   **Safety First:** Designed to prevent data corruption and provide a safe environment for rewriting operations, with automatic checks and warnings.
*   **Extensible:** Provides a core library for building custom history rewriting tools.
*   **Commit Message Rewriting:** Rewrites commit messages to reflect changes in commit IDs.
*   **Become-Empty and Become-Degenerate Pruning:** Automatically removes empty and degenerate commits after filtering.
*   **Upstream Improvements:** Contributed to many improvements in core Git commands like `fast-export` and `fast-import`.

## Installation

To install `git-filter-repo`, simply place the single-file Python script (`git-filter-repo`) into your system's `$PATH`. For advanced installations, refer to the [INSTALL.md](INSTALL.md) file.

## How to Use

*   **User Manual:** Consult the comprehensive [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html) for detailed documentation.
*   **Examples:** Find practical examples in the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES), [cheat sheets](Documentation/), and [examples from user-filed issues](Documentation/examples-from-user-filed-issues.md).

## Why Choose git-filter-repo?

`git-filter-repo` offers significant advantages over alternatives like `git filter-branch` and `BFG Repo Cleaner`:

*   **Superior Performance:** Outperforms `git filter-branch` by orders of magnitude for non-trivial rewrites.
*   **Robustness:** Addresses safety issues and gotchas present in `git filter-branch`.
*   **Versatility:** Provides a broader range of features and extensibility compared to `BFG Repo Cleaner`.

## Simple Example: Extracting a Directory with Renaming

Let's say you want to extract a directory named `src/`, rename its contents, and prefix tags.

**Using `git-filter-repo`:**

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

This single command extracts the `src/` directory, renames all files to have a `my-module/` prefix, and renames tags to `my-module-*`.

(See the original README for comparisons to other tools.)

## Contributing

We welcome contributions! Please review the [contributing guidelines](Documentation/Contributing.md).

## Code of Conduct

The project adheres to the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).