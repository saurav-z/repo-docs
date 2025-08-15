# Git Filter-Repo: Rewrite Git History with Power and Precision

**Tired of slow, error-prone history rewriting?** [Git filter-repo](https://github.com/newren/git-filter-repo) offers a fast, reliable, and feature-rich alternative to `git filter-branch` and other tools, providing unparalleled control over your Git repository's history.

## Key Features:

*   **Fast Performance:** Significantly faster than `git filter-branch` for complex rewriting tasks.
*   **Comprehensive Rewriting:** Handle a wide range of history manipulation scenarios, including path filtering, renaming, and more.
*   **Safe and Reliable:** Designed to avoid common pitfalls and data corruption issues associated with older tools.
*   **User-Friendly:** Simple command-line interface with extensive documentation and examples.
*   **Extensible:** Built as a library, allowing developers to create custom history rewriting tools.
*   **Advanced Features:** Includes features like automatic shrinking, commit message rewriting, and intelligent pruning.
*   **Conversion Guides:** Provides cheat sheets to convert commands from `filter-branch` and `BFG Repo Cleaner`.
*   **Upstream Improvements:** Drives improvements to core Git commands.

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

Installation is straightforward; simply place the single-file Python script `git-filter-repo` into your system's `$PATH`.  Consult [INSTALL.md](INSTALL.md) for more advanced installation scenarios.

## How do I use it?

Comprehensive documentation, including the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html), example conversion guides, and a [FAQ](Documentation/FAQ.md), are available to help you get started.

## Why filter-repo instead of other alternatives?

filter-repo excels where other tools like `filter-branch` and `BFG Repo Cleaner` fall short. See below for a detailed comparison:

### filter-branch

*   Slow performance for non-trivial repos.
*   Prone to data corruption and other issues.
*   Difficult to use for complex rewrites.
*   Deprecated in the Git project.
*   See [cheat sheet](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) for converting commands.

### BFG Repo Cleaner

*   Limited rewriting capabilities.
*   Limited by architecture for handling more types of rewrites.
*   See [cheat sheet](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg) for converting commands.

## Simple example, with comparisons

**Scenario:** Extract a directory (e.g., `src/`) into a separate repository, renaming files and tags.

### Solving this with filter-repo

```shell
git filter-repo --path src/ --to-subdirectory-filter my-module --tag-rename '':'my-module-'
```

### Solving this with BFG Repo Cleaner

BFG Repo Cleaner is not capable of performing this action.

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

*Requires several steps and has significant drawbacks, including performance issues, potential data corruption, and OS-specific commands.*

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

*Has caveats and limitations, including potential for corrupting data and missing commits.*

## Design rationale behind filter-repo

filter-repo's design prioritizes performance, safety, and versatility, addressing shortcomings in existing tools. See the original README for details.

## How do I contribute?

Review the [contributing guidelines](Documentation/Contributing.md).

## Is there a Code of Conduct?

The project adheres to the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

filter-repo has driven numerous improvements to core Git commands.