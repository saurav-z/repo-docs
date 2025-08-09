# git-filter-repo: The Modern Git History Rewriting Tool

**Tired of slow and unreliable history rewriting?** `git-filter-repo` is a powerful and efficient tool, recommended by the Git project, for rewriting and manipulating your Git repository's history. Find the original repo [here](https://github.com/newren/git-filter-repo).

## Key Features

*   **Superior Performance:** Significantly faster than `git filter-branch`.
*   **Comprehensive Capabilities:** Offers features not found in other tools.
*   **User-Friendly Design:** Easier to use for complex rewriting tasks.
*   **Extensible:** Provides a library for creating custom history rewriting tools.
*   **Safer Operations:** Designed to avoid common pitfalls and data corruption.
*   **Complete History Management:** Handles pruning, renaming, and rewriting with precision.
*   **Git Project Recommendation:** The recommended alternative to the older `git filter-branch`.

## Table of Contents

*   [Prerequisites](#prerequisites)
*   [How to Install](#how-do-i-install-it)
*   [How to Use](#how-do-i-use-it)
*   [Why `filter-repo`?](#why-filter-repo-instead-of-other-alternatives)
*   [Simple Example](#simple-example-with-comparisons)
*   [Design Rationale](#design-rationale-behind-filter-repo)
*   [How to Contribute](#how-do-i-contribute)
*   [Code of Conduct](#is-there-a-code-of-conduct)
*   [Upstream Improvements](#upstream-improvements)

## Prerequisites

`git-filter-repo` requires:

*   git >= 2.36.0
*   python3 >= 3.6

## How to Install

Installation is simple:

1.  Download the `git-filter-repo` python script.
2.  Place the file in your system's `$PATH`.

For more advanced installation options and special cases, refer to [INSTALL.md](INSTALL.md).

## How to Use

*   For a comprehensive user manual:  [https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html)
*   Find alternative formatting on external sites [example](https://www.mankier.com/1/git-filter-repo)
*   **Cheat Sheets:**
    *   Converting from `filter-branch` commands: [Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage)
    *   Converting from BFG Repo Cleaner commands: [Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg)
*   **Examples:**
    *   [Simple example](#simple-example-with-comparisons) below.
    *   User manual [EXAMPLES section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES)
    *   Examples based on [user-filed issues](Documentation/examples-from-user-filed-issues.md)
*   [Frequently Answered Questions](Documentation/FAQ.md)

## Why `filter-repo` Instead of Other Alternatives?

`git-filter-repo` excels where other tools like `filter-branch` and BFG Repo Cleaner fall short, offering significant advantages in performance, safety, and functionality.

### `filter-branch`

*   Significantly slower, especially for non-trivial repos.
*   Prone to errors and data corruption.
*   Difficult to use for more complex rewrites.

### BFG Repo Cleaner

*   Limited in the types of rewrites it can handle.
*   Lacks the architecture to support new features.

## Simple Example, with Comparisons

Let's say you want to extract a `src/` directory from your repo into a new module and rename the tags:

```
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

### Solving this with `filter-branch`

*   Complex and error-prone commands, like:
    ```shell
    git filter-branch \
        --tree-filter 'mkdir -p my-module && \
                       git ls-files \
                           | grep -v ^src/ \
                           | xargs git rm -f -q && \
                       ls -d * \
                           | grep -v my-module \
                           | xargs -I files mv files my-module/' \
            --tag-name-filter 'echo "my-module-$(cat)"' \
            --prune-empty -- --all
    git clone file://$(pwd) newcopy
    cd newcopy
    git for-each-ref --format="delete %(refname)" refs/tags/ \
        | grep -v refs/tags/my-module- \
        | git update-ref --stdin
    git gc --prune=now
    ```

### Solving this with BFG Repo Cleaner

*   BFG Repo Cleaner cannot handle this specific rewrite scenario.

### Solving this with fast-export/fast-import

*   Complex commands that are more error-prone
    ```shell
    git fast-export --no-data --reencode=yes --mark-tags --fake-missing-tagger \
        --signed-tags=strip --tag-of-filtered-object=rewrite --all \
        | grep -vP '^M [0-9]+ [0-9a-f]+ (?!src/)' \
        | grep -vP '^D (?!src/)' \
        | perl -pe 's%^(M [0-9]+ [0-9a-f]+ )(.*)$%\1my-module/\2%' \
        | perl -pe 's%^(D )(.*)$%\1my-module/\2%' \
        | perl -pe s%refs/tags/%refs/tags/my-module-% \
        | git -c core.ignorecase=false fast-import --date-format=raw-permissive \
              --force --quiet
    git for-each-ref --format="delete %(refname)" refs/tags/ \
        | grep -v refs/tags/my-module- \
        | git update-ref --stdin
    git reset --hard
    git reflog expire --expire=now --all
    git gc --prune=now
    ```

## Design Rationale Behind `filter-repo`

`git-filter-repo` was created to overcome the limitations of existing tools, focusing on these key areas:

*   **Analysis and Reporting:** Provides an initial analysis to help users.
*   **Keep/Remove Path Options:** Offers flexibility in selecting what to keep.
*   **Path Renaming:** Simple and safe path renaming options.
*   **Enhanced Safety:** Encourages working in fresh clones for safer operations.
*   **Automated Shrinking:** Simplifies post-filtering repository maintenance.
*   **Clean Separation:** Avoids mixing old and rewritten history.
*   **Extensibility:** Allows users to extend tool capabilities.
*   **Commit ID Mapping:** Provides a mechanism for old commit IDs.
*   **Consistent Commit Messages:** Rewrites commit messages to reference the new commit IDs.
*   **Empty/Degenerate Commit Pruning:** Handles pruning of commits that become empty.
*   **Speed:** Optimized for reasonably fast filtering.

## How to Contribute

See the [contributing guidelines](Documentation/Contributing.md) to get started.

## Is there a Code of Conduct?

The [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md) applies.

## Upstream Improvements

Work on `git-filter-repo` has contributed to numerous improvements in core Git's `fast-export` and `fast-import` commands. (See the original README for the exhaustive list of upstream improvements.)