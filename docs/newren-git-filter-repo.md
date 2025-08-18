# git-filter-repo: The Modern and Powerful Tool for Rewriting Git History

Tired of slow and error-prone history rewriting? **git-filter-repo** is the recommended tool for efficiently and reliably modifying your Git repository's history.  ([Original Repo](https://github.com/newren/git-filter-repo))

## Key Features

*   **High Performance:** Significantly faster than `git filter-branch`, especially for large repositories.
*   **Comprehensive Functionality:**  Offers advanced capabilities beyond basic filtering, including path renaming, content filtering, and more.
*   **Safe and Reliable:** Designed to avoid common pitfalls and data corruption issues associated with older tools.
*   **Flexible and Extensible:**  Includes a library for creating custom history rewriting tools to fit your specific needs.
*   **Recommended by Git Project:**  The preferred alternative to `git filter-branch`.
*   **Easy to Use:**  Simple command-line interface for common tasks.
*   **Detailed Documentation:** Comprehensive user manual, cheat sheets, and examples to guide you.

## Table of Contents

*   [Prerequisites](#prerequisites)
*   [How do I install it?](#how-do-i-install-it)
*   [How do I use it?](#how-do-i-use-it)
*   [Why filter-repo instead of other alternatives?](#why-filter-repo-instead-of-other-alternatives)
    *   [filter-branch](#filter-branch)
    *   [BFG Repo Cleaner](#bfg-repo-cleaner)
*   [Simple example, with comparisons](#simple-example-with-comparisons)
    *   [Solving this with filter-repo](#solving-this-with-filter-repo)
    *   [Solving this with BFG Repo Cleaner](#solving-this-with-bfg-repo-cleaner)
    *   [Solving this with filter-branch](#solving-this-with-filter-branch)
    *   [Solving this with fast-export/fast-import](#solving-this-with-fast-exportfast-import)
*   [Design rationale behind filter-repo](#design-rationale-behind-filter-repo)
*   [How do I contribute?](#how-do-i-contribute)
*   [Is there a Code of Conduct?](#is-there-a-code-of-conduct)
*   [Upstream Improvements](#upstream-improvements)

## Prerequisites

*   git >= 2.36.0
*   python3 >= 3.6

## How do I install it?

Installation is straightforward: simply place the `git-filter-repo` Python script into your system's `$PATH`. See [INSTALL.md](INSTALL.md) for more detailed instructions, such as installing documentation or working with non-standard Python setups.

## How do I use it?

*   **Comprehensive documentation:** See the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html).
*   **Examples and cheat sheets:** Convert `filter-branch` commands ([cheat sheet](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage)) and `BFG Repo Cleaner` commands ([cheat sheet](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg)) for easy transition. There are also many [examples](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES) and a [FAQ](Documentation/FAQ.md).

## Why filter-repo instead of other alternatives?

`git filter-repo` offers significant advantages over older tools such as `filter-branch` and `BFG Repo Cleaner`:

### filter-branch

*   **Slow Performance:** Often extremely slow, especially for large repositories.
*   **Error-Prone:** Prone to data corruption and other issues.
*   **Complex Usage:**  Difficult to use for anything beyond simple tasks.
*   **Deprecated:** The Git project recommends against using it.

### BFG Repo Cleaner

*   **Limited Scope:**  Restricted to a limited set of rewriting operations.
*   **Architectural Limitations:** The design is not easily extensible.

## Simple example, with comparisons

This example demonstrates extracting a directory to a new repository:

*   Extract the history of a single directory, `src/`.
*   Rename all files to have a new leading directory, `my-module/`.
*   Rename any tags in the extracted repository to have a `my-module-` prefix.

### Solving this with filter-repo

```shell
  git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

### Solving this with BFG Repo Cleaner

BFG Repo Cleaner cannot perform this rewrite.

### Solving this with filter-branch

This process is more complex and error-prone, and includes additional steps.

### Solving this with fast-export/fast-import

This method is also complex with several caveats.

## Design rationale behind filter-repo

This section explains the design choices behind filter-repo, covering aspects such as intelligent safety, renaming capabilities, and more. See original document for a detailed list.

## How do I contribute?

See the [contributing guidelines](Documentation/Contributing.md).

## Is there a Code of Conduct?

The project adheres to the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

`git-filter-repo` has driven numerous improvements in core Git, including:

*   [List of Git improvements](See original document for the full list of improvements and commit IDs)