# git-filter-repo: The Modern Git History Rewriting Tool

Tired of slow and error-prone Git history rewriting? **git-filter-repo** offers a powerful and efficient solution, now recommended by the Git project, to transform and clean up your repository's history.

[View the original repo here](https://github.com/newren/git-filter-repo)

## Key Features

*   **Speed:** Significantly faster than `git filter-branch`, especially for complex rewrites.
*   **Safety:** Designed to avoid common pitfalls and data corruption issues present in older tools.
*   **Versatility:** Provides a wide range of capabilities for rewriting history, including path filtering, renaming, and more.
*   **User-Friendly:** Offers a simple command-line interface with extensive documentation and examples.
*   **Extensible:** Built as a library, allowing you to create custom history rewriting tools.
*   **Comprehensive:** Addresses limitations found in other tools like `BFG Repo Cleaner` and `git filter-branch`.
*   **Upstream Improvements:** Contributed to numerous core Git improvements.

## Installation

`git-filter-repo` is typically installed by placing the single Python script `git-filter-repo` into your `$PATH`.

For advanced usage, see the [INSTALL.md](INSTALL.md) file.

## How to Use

*   **Comprehensive Documentation:** Access the user manual for detailed information.
    *   [User Manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html)
*   **Cheat Sheets:**
    *   Convert `filter-branch` commands: [Cheat Sheet](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage)
    *   Convert `BFG Repo Cleaner` commands: [Cheat Sheet](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg)
*   **Examples:** Explore examples in the manual and the `contrib/filter-repo-demos` directory.

## Why Choose git-filter-repo?

`git-filter-repo` offers several advantages over older alternatives like `git filter-branch` and `BFG Repo Cleaner`:

*   **Performance:** Significantly faster than `git filter-branch`, especially for large repositories.
*   **Functionality:** Provides more features and flexibility compared to `BFG Repo Cleaner`.
*   **Safety:** Designed to avoid data corruption issues that can occur with `git filter-branch`.
*   **Maintainability:** Actively maintained and improved by the Git community.

## Example: Extracting and Renaming a Directory

Let's say you want to extract the history of the `src/` directory, rename it to `my-module/src/`, and prefix any tags with `my-module-`. With `git-filter-repo`, this is a simple command:

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

## Contributing

Contribute to `git-filter-repo` by following the [contributing guidelines](Documentation/Contributing.md).

## Code of Conduct

The project follows the Git project's [Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

The development of `git-filter-repo` has directly contributed to improvements in core Git commands like `fast-export` and `fast-import`, ensuring compatibility and performance. See the original README for specific improvements.