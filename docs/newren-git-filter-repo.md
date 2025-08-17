# git-filter-repo: The Modern Tool for Rewriting Git History

Tired of slow and error-prone git history rewriting? **git-filter-repo** is the recommended tool by the Git project for efficient and versatile history manipulation, offering superior performance and advanced capabilities compared to `git filter-branch`.  Learn more about git-filter-repo at the [original repo](https://github.com/newren/git-filter-repo).

## Key Features

*   **Superior Performance:** Dramatically faster than `git filter-branch`, especially for complex repositories.
*   **Comprehensive Rewriting:** Handle complex tasks like path filtering, renaming, and tag manipulation with ease.
*   **Built-in Safety:** Designed to avoid common pitfalls that can corrupt your repository's history.
*   **User-Friendly:**  Simple command-line interface with extensive documentation and examples.
*   **Extensible:** Built as a library for creating new history rewriting tools.
*   **Automated Cleanup:** Automatically shrinks and repacks your repository after filtering.
*   **Commit Message Rewriting:**  Rewrites commit messages to correctly reference new commit IDs.
*   **Empty Commit Pruning:**  Removes commits that become empty due to filtering, preserving the integrity of your repository.
*   **Become-Degenerate Pruning:** Correctly handles topology changes in merge commits.
*   **Upstream Improvements:** Drives improvements in core Git commands, leading to a better Git experience.

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

Installation is straightforward. The main logic is in a single Python script named `git-filter-repo`. Just place it in your system's `$PATH`. For more advanced installation or specific situations (e.g., custom Python executables), consult the [INSTALL.md](INSTALL.md) file.

## How do I use it?

*   **Comprehensive Documentation:** Refer to the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html).  Alternative formatting is also available on external sites.
*   **Learning by Example:**  Check out the [cheat sheet for converting filter-branch commands](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) and the [cheat sheet for converting BFG Repo Cleaner commands](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg).  Also, see the [simple example](#simple-example-with-comparisons) below, the extensive [examples section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES) in the user manual, and [example filterings based on user-filed issues](Documentation/examples-from-user-filed-issues.md).
*   **FAQ:** The [Frequently Answered Questions](Documentation/FAQ.md) document may also be helpful.

## Why filter-repo instead of other alternatives?

See the [Git Rev News article on filter-repo](https://git.github.io/rev_news/2019/08/21/edition-54/#an-introduction-to-git-filter-repo--written-by-elijah-newren) for more details, but here's a summary:

### filter-branch

*   Extremely slow, especially for non-trivial repositories.
*   Riddled with potential pitfalls that can silently corrupt your rewrite.
*   Difficult to use for non-trivial rewrites.
*   The Git project recommends against using it due to these issues.
*   Consider [filter-lamely](contrib/filter-repo-demos/filter-lamely) (a.k.a. [filter-branch-ish](contrib/filter-repo-demos/filter-branch-ish)) for a reimplementation of filter-branch using filter-repo, which is more performant.
*   A [cheat sheet](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) helps convert filter-branch commands.

### BFG Repo Cleaner

*   A good tool, but limited to a few types of rewrites.
*   Its architecture isn't easily extended.
*   May have shortcomings and bugs even for its intended use cases.
*   Consider [bfg-ish](contrib/filter-repo-demos/bfg-ish), a reimplementation of bfg using filter-repo.
*   A [cheat sheet](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg) helps convert BFG Repo Cleaner commands.

## Simple example, with comparisons

Let's say we want to extract a directory `src/` from a repo:

*   Extract history of `src/`
*   Rename all files to `my-module/`
*   Rename any tags to have a `my-module-` prefix

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

(Plus several potential caveats - see original README)

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

(Plus several potential caveats and limitations - see original README)

## Design rationale behind filter-repo

The tool was created to address limitations in existing tools. The [original README](#design-rationale-behind-filter-repo) details the core principles behind the design of git-filter-repo, including its commitment to:

1.  Starting report
2.  Keep vs. Remove
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

Yes, participants in the filter-repo community are expected to follow the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

Work on filter-repo has led to numerous improvements in core Git's fast-export and fast-import commands, with links to relevant commits.