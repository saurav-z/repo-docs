# git-filter-repo: The Modern Tool for Rewriting Git History

**Tired of slow and error-prone Git history rewriting?**  [git-filter-repo](https://github.com/newren/git-filter-repo) offers a powerful, efficient, and safer alternative to `git filter-branch`.  Recommended by the Git project, it's your go-to solution for complex history modifications.

## Key Features

*   **Speed:** Significantly faster than `git filter-branch`, especially for large repositories.
*   **Safety:** Designed to avoid common pitfalls and data corruption issues found in other tools.
*   **Versatility:**  Handles a wide range of rewriting tasks, from simple path filtering to complex refactoring.
*   **User-Friendly:**  Easy to use for common tasks, and provides extensive documentation and examples.
*   **Extensible:**  Provides a core library that can be used to create custom history rewriting tools.
*   **Modern:** Incorporates numerous improvements to core Git commands (fast-import/fast-export).

## Installation

`git filter-repo` is designed for easy installation:  simply place the single-file python script into your $PATH.  See [INSTALL.md](INSTALL.md) for advanced options.

## How to Use

*   **User Manual:**  [View the user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html)
*   **Cheat Sheets:**
    *   [Converting from `filter-branch`](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage)
    *   [Converting from BFG Repo Cleaner](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg)
*   **Examples:** Explore the [example section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES) in the user manual and [examples from user-filed issues](Documentation/examples-from-user-filed-issues.md).
*   **FAQ:** Consult the [Frequently Answered Questions](Documentation/FAQ.md) for additional help.

## Why Choose git-filter-repo?

`git-filter-repo` provides significant advantages over its primary competitors:

*   **Compared to `git filter-branch`:** It's orders of magnitude faster, more reliable, and easier to use. `git filter-branch` is known for performance issues, data corruption risks, and complex usage.
*   **Compared to BFG Repo Cleaner:** While BFG is good for a limited set of tasks, `git-filter-repo` offers much greater flexibility and scalability for a wider range of history rewriting scenarios.

## Simple Example: Extracting a Directory

Let's say you want to extract the `src/` directory from your repository, rename it to `my-module/src/`, and prefix all tags with `my-module-`. Here's how to do it with `git-filter-repo`:

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

This single command replaces the multi-step, complex, and error-prone solutions required by `git filter-branch` and BFG.

## Contributing

Contribute to the project by following the [contributing guidelines](Documentation/Contributing.md).

## Code of Conduct

The project adheres to the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).