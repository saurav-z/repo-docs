# Git Filter-Repo: The Modern Solution for Rewriting Git History

Tired of slow and error-prone history rewriting? **git filter-repo** provides a superior alternative to `git filter-branch`, offering speed, safety, and a wealth of features for even the most complex Git history modifications.  Access the original repo at [https://github.com/newren/git-filter-repo](https://github.com/newren/git-filter-repo).

**Key Features:**

*   **Performance:** Significantly faster than `git filter-branch`, especially for large repositories.
*   **Safety:** Designed to avoid common pitfalls that can corrupt your repository with `git filter-branch`.
*   **Versatility:** Handles a wide range of history rewriting tasks, including path filtering, renaming, and more.
*   **Easy Installation:**  Simple to install as a single Python script, just place it in your `$PATH`.
*   **Extensibility:**  Provides a library for creating custom history rewriting tools.
*   **Commit Message Rewriting:**  Automatically updates commit messages to reflect new commit IDs after rewriting.
*   **Pruning:** Cleans up empty commits and degenerate merge commits.
*   **Comprehensive Documentation:** Detailed user manual, cheat sheets, and examples to guide you.

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

To install, simply place the `git-filter-repo` Python script in a directory within your `$PATH`. For more advanced installation options, refer to [INSTALL.md](INSTALL.md).

## How do I use it?

Comprehensive documentation is available in the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html). You can also find a [cheat sheet](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) for converting `filter-branch` commands and a wealth of [examples](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES) to get you started.

## Why filter-repo instead of other alternatives?

`git filter-repo` is designed to overcome the limitations of older tools:

### filter-branch

`git filter-branch` is slow, prone to errors, and lacks the flexibility to handle complex rewrites. The Git project itself recommends using `git filter-repo` instead.

### BFG Repo Cleaner

While BFG Repo Cleaner is a good tool, it's limited to a few rewrite types, and its architecture is not easily extensible.

## Simple example, with comparisons

Let's extract a subdirectory from a repository:

*   Extract a single directory
*   Rename the files in the extracted repo
*   Rename the tags

### Solving this with filter-repo

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

## Design rationale behind filter-repo

`filter-repo` was created to address shortcomings of other existing tools. It offers the following features, most other existing tools lack:
1. Starting report
2. Keep vs remove functionality
3. Renaming support
4. More Intelligent Safety
5. Auto Shrink feature
6. Clean Separation
7. Versatility for Customization
8. Old commit references
9. Commit message consistency
10. Become-empty pruning
11. Become-degenerate pruning
12. Speed
## How do I contribute?

See the [contributing guidelines](Documentation/Contributing.md) to learn how to contribute to the project.

## Is there a Code of Conduct?

Yes, the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md) applies.

## Upstream Improvements

Work on `filter-repo` has directly led to numerous improvements in core Git, including enhancements to `fast-export` and `fast-import`.