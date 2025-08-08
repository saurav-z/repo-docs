# Git Filter-Repo: Rewrite Your Git History with Ease

**Tired of slow and error-prone history rewriting?** Git filter-repo is a powerful, efficient, and modern tool recommended by the Git project for rewriting your repository's history, surpassing the limitations of `git filter-branch`. [Explore the Git filter-repo project](https://github.com/newren/git-filter-repo).

**Key Features:**

*   **Performance:** Significantly faster than `git filter-branch`, especially for complex rewrites.
*   **Comprehensive Capabilities:** Handles a wide range of history rewriting tasks, including path filtering, renaming, and more.
*   **Safety:** Designed to prevent common pitfalls and data corruption issues present in `git filter-branch`.
*   **Extensibility:** Provides a core library for building custom history rewriting tools tailored to specific needs.
*   **User-Friendly:** Offers clear documentation, examples, and cheat sheets to simplify the rewriting process.
*   **Automatic Cleanup:** Automatically removes old objects and repacks the repository for a clean, efficient result.
*   **Commit Message Rewriting:** Updates commit messages to reflect changes in commit IDs.
*   **Become-Empty and Become-Degenerate Pruning:** Efficiently removes unnecessary commits that result from filtering.
*   **Upstream Improvements:** Has driven many upstream improvements to the `fast-export` and `fast-import` git commands, enabling advanced history manipulation.

## Table of Contents

*   [Prerequisites](#prerequisites)
*   [How to Install](#how-do-i-install-it)
*   [How to Use](#how-do-i-use-it)
*   [Why Filter-Repo?](#why-filter-repo-instead-of-other-alternatives)
    *   [filter-branch](#filter-branch)
    *   [BFG Repo Cleaner](#bfg-repo-cleaner)
*   [Simple Example and Comparisons](#simple-example-with-comparisons)
    *   [Solving with filter-repo](#solving-this-with-filter-repo)
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

Installation is easy; just place the single-file python script `git-filter-repo` into your `$PATH`. For detailed instructions and advanced usage scenarios, see [INSTALL.md](INSTALL.md).

## How to Use

For comprehensive documentation, see the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html). You can also find alternative formatting on sites like [mankier.com](https://www.mankier.com/1/git-filter-repo).

If you prefer learning from examples, explore these resources:

*   [Cheat sheet for converting filter-branch commands](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage)
*   [Cheat sheet for converting BFG Repo Cleaner commands](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg)
*   The [simple example](#simple-example-with-comparisons) below
*   Extensive [examples section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES) in the user manual
*   [Example filterings based on user-filed issues](Documentation/examples-from-user-filed-issues.md)
*   [Frequently Answered Questions](Documentation/FAQ.md)

## Why Filter-Repo instead of Other Alternatives?

See the [Git Rev News article on filter-repo](https://git.github.io/rev_news/2019/08/21/edition-54/#an-introduction-to-git-filter-repo--written-by-elijah-newren) for a deeper dive.

### filter-branch

*   Extremely slow and can be unusable for non-trivial repositories.
*   Prone to data corruption and safety issues.
*   Onerous and complex to use for more than simple rewrites.
*   The Git project recommends against using `filter-branch`.

### BFG Repo Cleaner

*   A good tool for its time, but limited in the types of rewrites it can handle.
*   Its architecture restricts the types of rewrites it can handle.
*   May present shortcomings and bugs for its intended use cases.

## Simple Example, with Comparisons

Let's extract a single directory with these requirements:
*   Extract the history of a single directory `src/`.
*   Rename files to have a new leading directory, `my-module/`.
*   Rename any tags to have a `my-module-` prefix.

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
*(This command requires additional steps to clean up, rewrite commit messages, and more. This is still slow, and can cause various errors.)*

### Solving this with fast-export/fast-import

*(This is even more complex and error-prone, and requires more workarounds.)*

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

## Design Rationale behind filter-repo

filter-repo was built to address shortcomings of existing tools, with a focus on features such as:
*   Reporting and Analysis
*   Keeping and Removing
*   Renaming
*   More intelligent safety
*   Auto shrink
*   Clean separation
*   Versatility
*   Old commit references
*   Commit message consistency
*   Become-empty and Become-degenerate pruning
*   Speed

## How to Contribute

See the [contributing guidelines](Documentation/Contributing.md).

## Code of Conduct

The project follows the [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md).

## Upstream Improvements

filter-repo has contributed to many improvements to `fast-export` and `fast-import` in core Git. A list of these commits can be seen in the original readme.