# git-filter-repo: Powerful and Efficient Git History Rewriting

**Rewrite and refine your Git repository history with git-filter-repo, the recommended alternative to `git filter-branch`.**  Visit the original repo at: [https://github.com/newren/git-filter-repo](https://github.com/newren/git-filter-repo)

## Key Features:

*   **Superior Performance:** Significantly faster than `git filter-branch`, especially for complex rewrites.
*   **Enhanced Capabilities:** Offers features not found in other tools, including advanced path manipulation, renaming, and more.
*   **Safe and Reliable:** Designed to avoid common pitfalls that can corrupt your repository history.
*   **Easy to Use:** Simple command-line interface for common tasks, with extensive documentation and examples.
*   **Extensible:** Built as a library for creating custom history rewriting tools.
*   **Commit Message Rewriting:** Automatically updates commit messages to reflect changes in commit IDs.
*   **Automated Repository Shrinking:** Automatically removes old data and optimizes the repository after filtering.

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

Installation is simple: just place the single-file Python script `git-filter-repo` into your system's `$PATH`.

For detailed instructions, including special cases, see [INSTALL.md](INSTALL.md). This is particularly useful if you:

*   Use a Python 3 executable named something other than "python3".
*   Want to install documentation.
*   Want to run the `contrib` examples.
*   Want to use `filter-repo` as a module/library.

## How do I use it?

*   **Comprehensive documentation:**
    *   [User manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html)
    *   Alternative formatting on external sites (e.g., [mankier.com](https://www.mankier.com/1/git-filter-repo)).
*   **Examples:**
    *   [Cheat sheet for converting filter-branch commands](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage)
    *   [Cheat sheet for converting BFG Repo Cleaner commands](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg)
    *   [Simple example](#simple-example-with-comparisons) below.
    *   [Examples section in the user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES)
    *   [Examples from user-filed issues](Documentation/examples-from-user-filed-issues.md)
*   [Frequently Asked Questions](Documentation/FAQ.md)

## Why filter-repo instead of other alternatives?

For more details, see the [Git Rev News article](https://git.github.io/rev_news/2019/08/21/edition-54/#an-introduction-to-git-filter-repo--written-by-elijah-newren).

### filter-branch

*   [Extremely slow](https://public-inbox.org/git/CABPp-BGOz8nks0+Tdw5GyGqxeYR-3FF6FT5JcgVqZDYVRQ6qog@mail.gmail.com/) for non-trivial repositories.
*   [Riddled with gotchas](https://git-scm.com/docs/git-filter-branch#SAFETY) that can corrupt your rewrite.
*   [Onerous to use](#simple-example-with-comparisons) for complex rewrites.
*   The Git project recommends [stopping use](https://git-scm.com/docs/git-filter-branch#_warning).
*   Consider [filter-lamely](contrib/filter-repo-demos/filter-lamely) for a performant (but not as fast or safe) alternative.
*   Use the [cheat sheet](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) to convert commands.

### BFG Repo Cleaner

*   Limited to a few rewrite types.
*   Architecture is not easily extended.
*   May have some shortcomings even for its intended use cases.
*   Consider [bfg-ish](contrib/filter-repo-demos/bfg-ish) for a reimplementation.
*   Use the [cheat sheet](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg) to convert commands.

## Simple example, with comparisons

Let's extract a directory `src/` and rename it to `my-module/`, with tag prefixing.

### Solving this with filter-repo

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

### Solving this with BFG Repo Cleaner

BFG Repo Cleaner cannot handle this kind of rewrite.

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
(and with many caveats, see original README)

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
(and with even more caveats, see original README)

## Design rationale behind filter-repo

Key design goals include:

1.  [Starting report]: Analysis for easier start.
2.  [Keep vs. remove]: "Keep" paths for ease.
3.  [Renaming]: Easy path renaming and checks.
4.  [More intelligent safety]: Fresh clone by default.
5.  [Auto shrink]: Automatic repacking.
6.  [Clean separation]: Avoid confusion.
7.  [Versatility]: Extensibility without external processes.
8.  [Old commit references]: Handling of old commit IDs.
9.  [Commit message consistency]: Rewrite commit messages.
10. [Become-empty pruning]: Prune empty commits correctly.
11. [Become-degenerate pruning]: Special pruning for merge commits.
12. [Speed]: Fast performance.

## How do I contribute?

See the [contributing guidelines](Documentation/Contributing.md).

## Is there a Code of Conduct?

The project follows the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

Work on `git-filter-repo` has driven improvements in core Git, including `fast-export` and `fast-import`:

*   (List of upstream commits, as in original README)