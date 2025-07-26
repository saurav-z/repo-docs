# Git Filter-Repo: The Modern Solution for Rewriting Git History

Tired of slow, unreliable, and feature-limited history rewriting tools? **Git filter-repo** is a powerful, fast, and versatile tool for rewriting your Git repository's history, designed to overcome the limitations of `git filter-branch`. You can find the source code and more details at the [original repo](https://github.com/newren/git-filter-repo).

**Key Features:**

*   **Speed:** Significantly faster than `git filter-branch`, especially for large repositories.
*   **Safety:** Designed to avoid common pitfalls that can corrupt your repository.
*   **Versatility:** Offers a wide range of rewriting capabilities, far exceeding `git filter-branch` and the BFG Repo Cleaner.
*   **Usability:** Simple command-line interface for common tasks, with extensive documentation and examples.
*   **Extensibility:** Provides a library for creating custom history rewriting tools.
*   **Commit Message Rewriting:** Automatically updates commit messages to reflect changes in commit IDs.
*   **Become-Empty Pruning:** Cleans up the history by removing commits that become empty due to filtering.
*   **Become-Degenerate Pruning:** Removes merges commits that has become degenerate.
*   **Upstream Improvements:** Contributed to numerous improvements in core Git's `fast-export` and `fast-import` commands.

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

`git filter-repo` requires:

*   git >= 2.36.0
*   python3 >= 3.6

## How do I install it?

Installation is often as simple as placing the single-file Python script, `git-filter-repo`, into your system's `$PATH`.

See [INSTALL.md](INSTALL.md) for more detailed instructions if you need to:

*   Work with a Python 3 executable other than "python3".
*   Install documentation.
*   Run the examples in the `contrib` directory.
*   Use `filter-repo` as a Python module.

## How do I use it?

For comprehensive documentation:

*   See the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html)
*   Alternative formatting is available on sites like [mankier.com](https://www.mankier.com/1/git-filter-repo).

If you prefer learning from examples:

*   There is a [cheat sheet for converting filter-branch commands](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage)
*   There is a [cheat sheet for converting BFG Repo Cleaner commands](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg)
*   The [simple example](#simple-example-with-comparisons) below.
*   The user manual has an extensive [examples section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES).
*   I have collected a set of [example filterings based on user-filed issues](Documentation/examples-from-user-filed-issues.md).

You may also find the [Frequently Answered Questions](Documentation/FAQ.md) useful.

## Why filter-repo instead of other alternatives?

See a [Git Rev News article on filter-repo](https://git.github.io/rev_news/2019/08/21/edition-54/#an-introduction-to-git-filter-repo--written-by-elijah-newren) for a more detailed comparison:

### filter-branch

*   Significantly slower (multiple orders of magnitude) than `filter-repo`, especially in complex cases.
*   Prone to data corruption.
*   Onerous and error-prone for non-trivial rewrites.
*   The Git project recommends that you [stop using filter-branch](https://git-scm.com/docs/git-filter-branch#_warning).
*   Alternative:  [filter-lamely](contrib/filter-repo-demos/filter-lamely), a reimplementation of filter-branch based on filter-repo.
*   A [cheat sheet](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) is available for command conversion.

### BFG Repo Cleaner

*   Good tool but limited to specific rewrite types.
*   Its architecture limits its ability to handle more types of rewrites.
*   Includes bugs even for its intended use case.
*   Alternative: [bfg-ish](contrib/filter-repo-demos/bfg-ish), a reimplementation of bfg based on filter-repo.
*   A [cheat sheet](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg) is available for command conversion.

## Simple example, with comparisons

Imagine extracting a directory (`src/`) into a separate repository module:

*   Extract the history of `src/`.
*   Rename files to `my-module/` (e.g., `src/foo.c` becomes `my-module/src/foo.c`).
*   Rename tags to include a `my-module-` prefix.

### Solving this with filter-repo

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

### Solving this with BFG Repo Cleaner

BFG Repo Cleaner cannot perform this type of rewrite.

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
(or an index filter approach, both with the same caveats)

Caveats:

*   Commit messages are not rewritten.
*   The `--prune-empty` flag may miss commits.
*   Commands are OS-specific.
*   Both are multiple orders of magnitude slower than `filter-repo`.
*   Assumes ASCII filenames (special characters may wreak havoc).

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

Caveats:

*   Regex replacements operate on the entire fast-export stream, which might corrupt commit messages or file contents.
*   Assumes all ASCII filenames.
*   Leaves behind useless empty commits.
*   Commit messages are not rewritten.

## Design rationale behind filter-repo

`filter-repo` was created because existing tools lacked essential features, including:

1.  Provide starting reports to aid users.
2.  Allow *keeping* and *removing* paths.
3.  Easy renaming capabilities.
4.  Intelligent safety measures.
5.  Automatic repository shrinking and repacking.
6.  Clean separation of old and rewritten repositories.
7.  Extensibility.
8.  Old commit reference support.
9.  Commit message consistency.
10. Become-empty pruning.
11. Become-degenerate pruning.
12. Speed.

## How do I contribute?

See the [contributing guidelines](Documentation/Contributing.md).

## Is there a Code of Conduct?

Yes, the project adheres to the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

The development of `filter-repo` has driven numerous improvements in core Git, specifically in `fast-export` and `fast-import`. A list of those contributions is provided above.