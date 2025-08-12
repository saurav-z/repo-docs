# git-filter-repo: The Powerful Git History Rewriter

**Tired of slow, unreliable history rewriting?** **git-filter-repo** provides a robust and efficient solution for rewriting your Git repository history, offering features and performance that surpass traditional tools like `git filter-branch`. Find out more about git-filter-repo [here](https://github.com/newren/git-filter-repo).

## Key Features:

*   **Superior Performance:** Significantly faster than `git filter-branch`, especially for complex rewrites.
*   **Comprehensive Functionality:** Addresses limitations of other tools, providing features like path renaming, keeping specific files, and advanced safety checks.
*   **Simplified Workflow:** Simplifies common tasks like extracting subdirectories, renaming files/directories, and removing sensitive data.
*   **Extensible Design:** Serves as a library for building custom history rewriting tools, enabling advanced users to tailor solutions to their needs.
*   **Safety First:** Encourages the use of fresh clones to prevent data loss and includes numerous safety checks to protect against accidental corruption.
*   **Commit Message Rewriting:** Rewrites commit messages to reflect changes in commit IDs, maintaining the integrity of your history.
*   **Empty Commit Management:** Automatically handles empty commits created by filtering, ensuring correct graph topology.
*   **Upstream Improvements:** Contributes to the improvement of core Git commands like `fast-export` and `fast-import`.

## Table of Contents

*   [Prerequisites](#prerequisites)
*   [How to Install](#how-do-i-install-it)
*   [How to Use](#how-do-i-use-it)
*   [Why git-filter-repo?](#why-filter-repo-instead-of-other-alternatives)
    *   [filter-branch](#filter-branch)
    *   [BFG Repo Cleaner](#bfg-repo-cleaner)
*   [Simple Example](#simple-example-with-comparisons)
    *   [Solving with git-filter-repo](#solving-this-with-filter-repo)
    *   [Solving with BFG Repo Cleaner](#solving-this-with-bfg-repo-cleaner)
    *   [Solving with filter-branch](#solving-this-with-filter-branch)
    *   [Solving with fast-export/fast-import](#solving-this-with-fast-exportfast-import)
*   [Design Rationale](#design-rationale-behind-filter-repo)
*   [How to Contribute](#how-do-i-contribute)
*   [Code of Conduct](#is-there-a-code-of-conduct)
*   [Upstream Improvements](#upstream-improvements)

## Prerequisites

*   git >= 2.36.0
*   python3 >= 3.6

## How to Install

Installation is easy! The main script is a single Python file. Just place the `git-filter-repo` file into your system's `$PATH`.

For more detailed instructions and special cases, see [INSTALL.md](INSTALL.md).

## How to Use

*   **User Manual:** Comprehensive documentation can be found in the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html).
*   **Examples:**
    *   [Cheat sheet for converting filter-branch commands](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage)
    *   [Cheat sheet for converting BFG Repo Cleaner commands](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg)
    *   [Simple example](#simple-example-with-comparisons) below
    *   [Examples section in the user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES)
    *   [Example filterings based on user-filed issues](Documentation/examples-from-user-filed-issues.md)
*   **FAQ:** The [Frequently Answered Questions](Documentation/FAQ.md) may be helpful.

## Why git-filter-repo Instead of Other Alternatives?

### filter-branch

*   **Slow Performance:** Significantly slower than `git-filter-repo` (multiple orders of magnitude slower).
*   **Error-Prone:** Prone to silently corrupting your rewrite or frustrating cleanup efforts.
*   **Complex Syntax:** Difficult to use for even slightly complex rewrites.
*   **Deprecated:** The Git project recommends against using it.
*   **Alternative:** Consider [filter-lamely](contrib/filter-repo-demos/filter-lamely) (a reimplementation of filter-branch based on `git-filter-repo`).
*   [Cheat Sheet:](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage)  for converting filter-branch commands to git-filter-repo commands.

### BFG Repo Cleaner

*   **Limited Scope:**  Limited to a few specific rewrite types.
*   **Architectural Limitations:**  Not easily extended to handle new rewrite scenarios.
*   **Shortcomings:** Contains known bugs and limitations.
*   **Alternative:** Consider [bfg-ish](contrib/filter-repo-demos/bfg-ish) (a reimplementation of BFG based on `git-filter-repo`).
*   [Cheat Sheet:](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg) for converting BFG Repo Cleaner commands to git-filter-repo commands.

## Simple Example, with Comparisons

**Scenario:** Extract a subdirectory (`src/`) of a repository, rename it, and rename tags for merging purposes.

*   Extract `src/` to a new repository.
*   Rename all files under `src/` to `my-module/`.
*   Rename all tags to be prefixed with `my-module-`.

### Solving this with git-filter-repo

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

### Solving this with BFG Repo Cleaner

BFG Repo Cleaner is not capable of this kind of rewrite.

### Solving this with filter-branch

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

**Caveats:** This is a slow and complex method with many potential pitfalls and OS-specific issues.

### Solving this with fast-export/fast-import

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

**Caveats:** This approach is complex and error-prone, with risks of data corruption and limitations in handling filenames.

## Design Rationale behind filter-repo

`git-filter-repo` was developed to address the shortcomings of existing tools. It provides:

1.  **Analysis:** Provides an initial analysis of the repo.
2.  **Keep vs. Remove:** Allows users to only *keep* certain paths.
3.  **Renaming:** Simplifies path renaming and provides sanity checks.
4.  **Safety:** Detects and bails in fresh clones unless --force is used.
5.  **Auto Shrink:** Automatically removes old cruft.
6.  **Clean Separation:** Avoids mixing of old and rewritten repos.
7.  **Versatility:** Offers the ability to extend the tool or write new tools.
8.  **Old Commit References:** Provides mapping from old to new hashes.
9.  **Commit Message Consistency:** Rewrites messages to use the new commit IDs.
10. **Become-Empty Pruning:** Commits which become empty due to filtering should be pruned.
11. **Become-Degenerate Pruning:** Pruning of commits with topological issues.
12. **Speed:** High performance.

## How to Contribute

See the [contributing guidelines](Documentation/Contributing.md).

## Code of Conduct

The [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md) applies to the filter-repo community.

## Upstream Improvements

`git-filter-repo` has driven many improvements in core Git commands like `fast-export` and `fast-import`. A list of those is provided in the original README.