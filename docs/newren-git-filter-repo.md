# git-filter-repo: Rewrite Your Git History with Speed and Precision

Tired of slow and unreliable history rewriting? **git-filter-repo** offers a powerful and efficient solution, recommended by the Git project itself, for complex repository transformations. [Visit the repository](https://github.com/newren/git-filter-repo) for more information.

**Key Features:**

*   **Speed and Efficiency:** Outperforms `git filter-branch` by orders of magnitude.
*   **Comprehensive Rewriting Capabilities:** Handles a wide range of history modifications, including path filtering, renaming, and more.
*   **User-Friendly Design:** Simplifies complex rewriting tasks with intuitive commands and options.
*   **Safety First:** Encourages fresh clone workflows and provides built-in safety checks.
*   **Extensible Architecture:** Offers a library for creating custom history rewriting tools.
*   **Commit Message Rewriting:**  Ensures commit messages are updated to reflect new commit IDs.
*   **Empty Commit Handling:**  Intelligently prunes commits that become empty due to filtering.
*   **Upstream Improvements:** Contributes to core Git's fast-export and fast-import commands for enhanced functionality.

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

To get started, simply place the `git-filter-repo` Python script into your system's `$PATH`.

For more advanced usage and installation options, see [INSTALL.md](INSTALL.md). This is required if:

*   You need more detailed instructions.
*   You are using a python3 executable that isn't named "python3".
*   You want to install documentation.
*   You'd like to run the [contrib](contrib/filter-repo-demos/) examples.
*   You plan to use filter-repo as a module/library.

## How do I use it?

For in-depth documentation:

*   Consult the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html).
*   Alternative formatting is available on external sites ([example](https://www.mankier.com/1/git-filter-repo)).

If you prefer to learn by example:

*   The [cheat sheet](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) is helpful for converting filter-branch commands.
*   See the [cheat sheet](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg) for BFG Repo Cleaner commands.
*   The [simple example](#simple-example-with-comparisons) below.
*   The user manual also includes a comprehensive [examples section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES).
*   [Example filterings based on user-filed issues](Documentation/examples-from-user-filed-issues.md)

You may also find the [Frequently Answered Questions](Documentation/FAQ.md) useful.

## Why filter-repo instead of other alternatives?

The [Git Rev News article on filter-repo](https://git.github.io/rev_news/2019/08/21/edition-54/#an-introduction-to-git-filter-repo--written-by-elijah-newren) provides a more detailed discussion, but here's a quick comparison with its main competitors:

### filter-branch

*   Significantly slower than `filter-repo`, potentially by multiple orders of magnitude.
*   Prone to errors that can corrupt your rewrite.
*   Complex to use for anything beyond basic rewriting tasks.
*   Not backwards-compatible fixable; Git project recommends against its use.
*   [filter-lamely](contrib/filter-repo-demos/filter-lamely) is a reimplementation based on filter-repo and may interest die-hard fans.
*   A [cheat sheet](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) is available.

### BFG Repo Cleaner

*   Limited to a few types of rewrites.
*   Its architecture is limiting for new features.
*   Can present shortcomings and bugs, even for its intended usecase.
*   [bfg-ish](contrib/filter-repo-demos/bfg-ish) is a reimplementation based on filter-repo which may interest fans of BFG.
*   A [cheat sheet](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg) is available.

## Simple example, with comparisons

Let's extract a specific part of your repository. We want to:

*   Extract the history of a single directory, src/.
*   Rename all files to have a new leading directory, my-module/
*   Rename any tags to have a 'my-module-' prefix.

### Solving this with filter-repo

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

### Solving this with BFG Repo Cleaner

BFG Repo Cleaner isn't capable of this type of rewrite.

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

or with `index-filter`:
```shell
  git filter-branch \
      --index-filter 'git ls-files \
                          | grep -v ^src/ \
                          | xargs git rm -q --cached;
                      git ls-files -s \
                          | sed "s%$(printf \\t)%&my-module/%" \
                          | git update-index --index-info;
                      git ls-files \
                          | grep -v ^my-module/ \
                          | xargs git rm -q --cached' \
      --tag-name-filter 'echo "my-module-$(cat)"' \
      --prune-empty -- --all
  git clone file://$(pwd) newcopy
  cd newcopy
  git for-each-ref --format="delete %(refname)" refs/tags/ \
      | grep -v refs/tags/my-module- \
      | git update-ref --stdin
  git gc --prune=now
```

However, this comes with caveats. Commit messages aren't rewritten, OS specific issues, and potentially slow speed are all issues.

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

Caveats include operating on entire fast-export streams and assumes all filenames are ASCII characters.

## Design rationale behind filter-repo

filter-repo was designed to overcome the limitations of existing tools, focusing on features like:

1.  [Starting report] Analysis of your repo
2.  [Keep vs. remove] Keep certain paths
3.  [Renaming] Easier renaming of paths
4.  [More intelligent safety] Encouraging fresh clone use.
5.  [Auto shrink] Removing old cruft after filtering
6.  [Clean separation] Avoid mixing old repo and rewritten repo
7.  [Versatility] Ability to extend and create tools from existing features.
8.  [Old commit references] Way to use old commit IDs with new repo
9.  [Commit message consistency] Rewrite commit messages
10. [Become-empty pruning] Prune empty commits
11. [Become-degenerate pruning] Prune merge commits with no file changes.
12. [Speed] Reasonably fast filtering

## How do I contribute?

See the [contributing guidelines](Documentation/Contributing.md).

## Is there a Code of Conduct?

The project adheres to the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

filter-repo has driven significant improvements to core Git's `fast-export` and `fast-import` commands.