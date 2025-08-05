# git-filter-repo: The Modern Way to Rewrite Git History

Tired of slow, error-prone history rewriting tools? **git-filter-repo** offers a powerful and efficient solution, now recommended by the Git project.  Get started with [git-filter-repo](https://github.com/newren/git-filter-repo) today!

## Key Features

*   **High Performance:** Significantly faster than `git filter-branch` and other alternatives, especially for complex rewrites.
*   **Versatile:**  Easily handle a wide range of history rewriting tasks, including path filtering, renaming, and tag manipulation.
*   **Safe:**  Designed to prevent common pitfalls and data corruption issues found in older tools.
*   **User-Friendly:** Simple command-line interface with intuitive options, plus extensive documentation and examples.
*   **Extensible:** Built as a library, allowing advanced users to create custom history rewriting tools.
*   **Advanced Features:** Includes features such as commit message rewriting, become-empty pruning, and more intelligent safety checks.

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

Installation is straightforward: simply place the single-file python script `git-filter-repo` into your `$PATH`.  See [INSTALL.md](INSTALL.md) for more detailed instructions.

## How do I use it?

For comprehensive documentation:

*   See the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html)

If you prefer learning from examples:

*   There is a [cheat sheet for converting filter-branch commands](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage), which covers every example from the filter-branch manual
*   There is a [cheat sheet for converting BFG Repo Cleaner commands](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg), which covers every example from the BFG website
*   The [simple example](#simple-example-with-comparisons) below may be of interest.
*   The user manual has an extensive [examples section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES)
*   I have collected a set of [example filterings based on user-filed issues](Documentation/examples-from-user-filed-issues.md)

In either case, you may also find the [Frequently Answered Questions](Documentation/FAQ.md) useful.

## Why filter-repo instead of other alternatives?

Here's a comparison of `git-filter-repo` to the main competitors:

### filter-branch

*   **Performance Issues:** `filter-branch` is often *much* slower than necessary, even unusable, for non-trivial repositories.
*   **Error-Prone:** `filter-branch` has numerous potential pitfalls that can silently corrupt your repository.
*   **Complex and Cumbersome:**  It's difficult to use for even slightly complex rewrites.
*   **Deprecated:** The Git project recommends *against* using `filter-branch`.
*   [filter-lamely](contrib/filter-repo-demos/filter-lamely) and a [cheat sheet](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) can help you convert filter-branch commands to filter-repo.

### BFG Repo Cleaner

*   **Limited Capabilities:** While good for certain tasks, BFG's architecture restricts the types of rewrites it can perform.
*   **Shortcomings:**  It has shortcomings and bugs, even within its intended use cases.
*   [bfg-ish](contrib/filter-repo-demos/bfg-ish) and a [cheat sheet](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg) can help you convert BFG commands to filter-repo.

## Simple example, with comparisons

Let's say you want to extract a directory (`src/`) from a repository into a new module, renaming the files and tags in the process.

### Solving this with filter-repo

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

### Solving this with BFG Repo Cleaner

BFG Repo Cleaner cannot perform this operation.

### Solving this with filter-branch

This is an example of how difficult it is to use filter-branch:

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

Here are some other filter-branch gotchas:

*   Commit messages are not rewritten and so commit IDs are not updated
*   The --prune-empty flag misses commits that should be pruned and prunes commits that started empty
*   The above commands are OS-specific
*   The --index-filter version of the filter-branch command may be two to three times faster than the --tree-filter version, but both filter-branch commands are going to be multiple orders of magnitude slower than filter-repo.
*   Both commands assume all filenames are composed entirely of ascii characters

### Solving this with fast-export/fast-import

One can kind of hack this together with something like:

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

But this comes with some nasty caveats and limitations:

*   The various greps and regex replacements operate on the entire
    fast-export stream and thus might accidentally corrupt unintended
    portions of it, such as commit messages.
*   This command assumes all filenames in the repository are composed
    entirely of ascii characters
*   This command will leave behind huge numbers of useless empty
    commits, and has no realistic way of pruning them.
*   Commit messages which reference other commits by hash will now
    reference old commits that no longer exist.

## Design rationale behind filter-repo

git-filter-repo was created to address the shortcomings of existing tools and provide:

1.  Starting report
2.  Keep vs. remove
3.  Renaming
4.  More intelligent safety
5.  Auto shrink
6.  Clean separation
7.  Versatility
8.  Old commit references
9.  Commit message consistency
10. Become-empty pruning
11. Become-degenerate pruning
12. Speed

## How do I contribute?

See the [contributing guidelines](Documentation/Contributing.md).

## Is there a Code of Conduct?

Yes, the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md) applies.

## Upstream Improvements

filter-repo has driven many improvements to core Git commands, including `fast-export` and `fast-import`. Here's a list of Git commits that have been driven by this project:
(list of commits from original README)