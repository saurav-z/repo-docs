# Git Filter-Repo: The Ultimate Tool for Rewriting Git History

**Git filter-repo** is a powerful and efficient tool recommended by the Git project for rewriting your repository's history, offering capabilities not found in other tools.  [Explore the original repository here](https://github.com/newren/git-filter-repo).

## Key Features:

*   **Superior Performance:** Outperforms `git filter-branch` by orders of magnitude, especially for large repositories.
*   **Comprehensive Capabilities:** Offers advanced features for history rewriting, including path filtering, renaming, and more.
*   **User-Friendly Design:** Designed for usability, making complex rewriting tasks manageable.
*   **Extensible Library:** Provides a core library for creating custom history rewriting tools.
*   **Safety First:** Enforces best practices to prevent common pitfalls and data corruption during rewriting.
*   **Commit Message Rewriting:** Automatically updates commit messages to reflect changes in commit IDs.
*   **Automatic Repository Cleanup:** Streamlines the process by removing old cruft and repacking the repository.
*   **Upstream Improvements:** Contributes to core Git with improvements to related tools like `fast-export` and `fast-import`.

## Table of Contents

  * [Prerequisites](#prerequisites)
  * [How do I install it?](#how-do-i-install-it)
  * [How do I use it?](#how-do-i-use-it)
  * [Why filter-repo instead of other alternatives?](#why-filter-repo-instead-of-other-alternatives)
  * [Simple example, with comparisons](#simple-example-with-comparisons)
  * [Design rationale behind filter-repo](#design-rationale-behind-filter-repo)
  * [How do I contribute?](#how-do-i-contribute)
  * [Is there a Code of Conduct?](#is-there-a-code-of-conduct)
  * [Upstream Improvements](#upstream-improvements)

## Prerequisites

*   git >= 2.36.0
*   python3 >= 3.6

## How do I install it?

Installation is straightforward. The main logic is in a single-file Python script named `git-filter-repo`. Simply place the script in your `$PATH`. See [INSTALL.md](INSTALL.md) for more detailed instructions.

## How do I use it?

*   **User Manual:** Comprehensive documentation is available in the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html).
*   **Examples:**
    *   [Converting from filter-branch commands](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage)
    *   [Converting from BFG Repo Cleaner commands](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg)
    *   [Simple Example](#simple-example-with-comparisons)
    *   [Examples Section in User Manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES)
    *   [Examples from User-Filed Issues](Documentation/examples-from-user-filed-issues.md)
*   **FAQ:** Consult the [Frequently Answered Questions](Documentation/FAQ.md) for common issues.

## Why filter-repo instead of other alternatives?

Git filter-repo provides significant advantages over `git filter-branch` and BFG Repo Cleaner.

### filter-branch

*   **Performance:**  `filter-branch` is slow, potentially unusably so, for non-trivial repositories.
*   **Safety:**  Prone to silent corruption and gotchas that can result in a messy or unusable repository.
*   **Complexity:** Difficult to use for non-trivial rewriting operations.
*   **Deprecation:**  The Git project recommends against using `filter-branch`.

### BFG Repo Cleaner

*   **Limited Functionality:**  Restricted to a specific set of rewrite operations.
*   **Architecture:**  Not designed to handle a wider variety of rewriting needs.
*   **Shortcomings:**  Presents some shortcomings and bugs even for its intended use case.

## Simple example, with comparisons

Let's extract the history of the `src/` directory into a new repository:

*   Extract the history of a single directory, `src/`.
*   Rename all files to have a new leading directory, `my-module/`.
*   Rename any tags with a `my-module-` prefix.

### Solving this with filter-repo

```shell
  git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

### Solving this with BFG Repo Cleaner

BFG Repo Cleaner cannot perform all of the desired operations.

### Solving this with filter-branch

```shell
git filter-branch --tree-filter 'mkdir -p my-module && git ls-files | grep -v ^src/ | xargs git rm -f -q && ls -d * | grep -v my-module | xargs -I files mv files my-module/' --tag-name-filter 'echo "my-module-$(cat)"' --prune-empty -- --all
  git clone file://$(pwd) newcopy
  cd newcopy
  git for-each-ref --format="delete %(refname)" refs/tags/ | grep -v refs/tags/my-module- | git update-ref --stdin
  git gc --prune=now
```

This requires more steps and has many caveats.

### Solving this with fast-export/fast-import

```shell
git fast-export --no-data --reencode=yes --mark-tags --fake-missing-tagger --signed-tags=strip --tag-of-filtered-object=rewrite --all | grep -vP '^M [0-9]+ [0-9a-f]+ (?!src/)' | grep -vP '^D (?!src/)' | perl -pe 's%^(M [0-9]+ [0-9a-f]+ )(.*)$%\1my-module/\2%' | perl -pe 's%^(D )(.*)$%\1my-module/\2%' | perl -pe s%refs/tags/%refs/tags/my-module-% | git -c core.ignorecase=false fast-import --date-format=raw-permissive --force --quiet
  git for-each-ref --format="delete %(refname)" refs/tags/ | grep -v refs/tags/my-module- | git update-ref --stdin
  git reset --hard
  git reflog expire --expire=now --all
  git gc --prune=now
```

This has multiple limitations, including the potential for data corruption.

## Design rationale behind filter-repo

The tool was designed with the following key principles:

  1. Starting report
  2. Keep vs. remove
  3. Renaming
  4. More intelligent safety
  5. Auto shrink
  6. Clean separation
  7. Versatility
  8. Old commit references
  9. Commit message consistency
  10. Become-empty pruning
  11. Become-degenerate pruning
  12. Speed

## How do I contribute?

See the [contributing guidelines](Documentation/Contributing.md).

## Is there a Code of Conduct?

The project follows the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

Git filter-repo has driven numerous improvements to core git, including the following commits in `fast-export` and `fast-import`.