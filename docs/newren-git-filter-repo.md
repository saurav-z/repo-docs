# git-filter-repo: The Modern Git History Rewriting Tool

**Tired of slow and unreliable git history rewriting?**  `git-filter-repo` offers a fast, safe, and versatile solution for transforming your Git repository's history. [Learn more on GitHub](https://github.com/newren/git-filter-repo).

## Key Features

*   **Superior Performance:** Significantly faster than `git filter-branch`.
*   **Enhanced Safety:** Designed to avoid common pitfalls and data corruption.
*   **Versatile Functionality:** Includes features not found in other tools.
*   **Extensible:** Provides a library for building custom history rewriting tools.
*   **Recommended by Git Project:**  The Git project itself recommends `git-filter-repo` over `git filter-branch`.

## Installation

`git-filter-repo` is easy to install: simply place the single-file Python script named `git-filter-repo` into your system's `$PATH`.  See [INSTALL.md](INSTALL.md) for more complex installations.

### Prerequisites

*   git >= 2.36.0
*   python3 >= 3.6

## Usage

For detailed documentation and examples:

*   **User Manual:** [View the comprehensive user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html).
*   **Cheat Sheets:** Convert existing commands from [git filter-branch](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) and [BFG Repo Cleaner](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg).
*   **Examples:** Explore the [examples section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES) in the user manual.

## Why Choose git-filter-repo?

`git-filter-repo` surpasses alternatives like `git filter-branch` and BFG Repo Cleaner in speed, safety, and flexibility.  It addresses the limitations and potential data corruption issues inherent in `filter-branch` while offering a more powerful and adaptable approach than BFG Repo Cleaner.

## Example: Extracting a Directory

This example demonstrates extracting the history of the `src/` directory into a new repository:

**Goal:**
*   Extract the history of a single directory, src/.
*   Rename all files to have a new leading directory, my-module/
*   Rename any tags to have a 'my-module-' prefix.

**Solution with `git-filter-repo`:**

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

This single command efficiently achieves the desired results.  Compare this to the complexity and limitations of other tools in the original README for a more thorough comparison.

## Contributing

Contribute to the project by following the [contributing guidelines](Documentation/Contributing.md).

## Code of Conduct

The project adheres to the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

`git-filter-repo` has driven numerous improvements to core Git, enhancing fast-export and fast-import functionality. See the original README for a full list.