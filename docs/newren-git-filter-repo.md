# Git Filter-Repo: Powerful and Efficient Git History Rewriting

Tired of slow and unreliable git history rewriting? **Git filter-repo** provides a robust and efficient solution for complex repository transformations, offering capabilities not found in other tools. [Explore the original repository](https://github.com/newren/git-filter-repo) for more details.

## Key Features:

*   **Performance:** Significantly faster than `git filter-branch`, enabling efficient rewriting even on large repositories.
*   **Versatility:** Offers a wide range of rewriting options, including path filtering, renaming, and tag manipulation.
*   **Safety:** Designed to prevent common pitfalls of other history rewriting tools, reducing the risk of data corruption.
*   **Extensibility:** Provides a library for creating custom history rewriting tools tailored to specific needs.
*   **User-Friendly:** Offers a simple command-line interface with comprehensive documentation and examples.
*   **Commit Message Rewriting:** Rewrites commit messages to reflect changes in commit IDs.
*   **Automatic Cleanup:** Automatically removes old objects and repacks the repository after filtering.
*   **Fresh Clone Enforcement:** Encourages working from a fresh clone to ensure a clean and safe rewriting process.
*   **Pruning and Degeneracy handling:** Automatic pruning of commits that become empty or degenerate due to filtering.
*   **Upstream Improvements:** Numerous improvements to core Git commands (e.g., `fast-export`, `fast-import`) have been driven by the needs of `git filter-repo`.

## Prerequisites:

*   git >= 2.36.0
*   python3 >= 3.6

## Installation:

Simply place the `git-filter-repo` Python script into your system's `$PATH`. For advanced usage, consult the [INSTALL.md](INSTALL.md) file.

## How to Use:

*   Comprehensive documentation is available in the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html).
*   Convert commands from `filter-branch` using the [cheat sheet](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage).
*   Convert commands from `BFG Repo Cleaner` using the [cheat sheet](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg).
*   Explore the [examples section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES) in the user manual for practical applications.
*   Refer to the [Frequently Answered Questions](Documentation/FAQ.md) for common issues and solutions.

## Why Choose Git Filter-Repo?

Compared to alternatives like `git filter-branch` and `BFG Repo Cleaner`, git filter-repo offers:

*   **Superior Speed and Reliability:** Addresses performance and safety limitations of `git filter-branch`.
*   **Greater Flexibility:** Handles a broader range of rewriting scenarios than `BFG Repo Cleaner`.
*   **Simplified Workflow:** Automates key processes like repository cleanup, making it easier to use.
*   **Community Support:** Actively maintained and recommended by the Git project.

## Example: Extracting and Renaming a Directory

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

This command extracts the `src/` directory, moves its contents to `my-module/src/`, and prefixes tags with `my-module-`.

## Design Rationale

Git filter-repo was created to address the shortcomings of existing tools, including issues with speed, safety, and flexibility.

## Contributing:

See the [contributing guidelines](Documentation/Contributing.md) for information on how to contribute to the project.

## Code of Conduct:

The project adheres to the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).