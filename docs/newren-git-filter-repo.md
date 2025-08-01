# Git Filter-Repo: The Powerful History Rewriting Tool

Tired of slow, error-prone history rewriting? **Git filter-repo** is your go-to solution, providing a faster, more reliable, and feature-rich alternative to `git filter-branch`. Visit the [original repo](https://github.com/newren/git-filter-repo) for the latest updates.

**Key Features:**

*   **Significantly Faster:** Outperforms `git filter-branch` by orders of magnitude, especially on large repositories.
*   **More Capabilities:** Offers a wider range of rewriting options not found in other tools.
*   **Safer & More Reliable:** Designed to avoid common pitfalls and data corruption issues.
*   **Intuitive Command-Line Interface:** Easy to use, with a focus on simplicity and clarity.
*   **Extensible Library:** Provides a foundation for building custom history rewriting tools.
*   **Comprehensive Documentation:** Includes a user manual, cheat sheets, and examples to guide you.
*   **Automated Repository Shrinking:** Simplifies the process and avoids potential issues.
*   **Handles Commit Message Rewriting:** Ensures commit messages are updated to reflect the new history.

**Table of Contents**

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

Installation is straightforward: simply place the single-file Python script `git-filter-repo` into your system's `$PATH`.

For more advanced installation options, refer to [INSTALL.md](INSTALL.md). This includes instructions for:

*   Non-standard Python3 executable names
*   Installing documentation
*   Running the `contrib` examples
*   Developing custom filtering scripts

## How do I use it?

For detailed information and guidance, see the [user manual](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html).

You can also find helpful resources:

*   [Cheat sheet for converting filter-branch commands](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage)
*   [Cheat sheet for converting BFG Repo Cleaner commands](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg)
*   [Simple example](#simple-example-with-comparisons) below.
*   [Examples section](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html#EXAMPLES) in the user manual.
*   [Examples based on user-filed issues](Documentation/examples-from-user-filed-issues.md).
*   [Frequently Answered Questions](Documentation/FAQ.md).

## Why filter-repo instead of other alternatives?

Git filter-repo addresses the shortcomings of alternative tools like `filter-branch` and BFG Repo Cleaner. The [Git Rev News article on filter-repo](https://git.github.io/rev_news/2019/08/21/edition-54/#an-introduction-to-git-filter-repo--written-by-elijah-newren) provides further details.

### filter-branch

*   Significantly slower than filter-repo, especially for complex repositories.
*   Prone to data corruption and other errors.
*   Difficult to use for even moderately complex rewriting tasks.
*   The Git project recommends against using `filter-branch` due to its limitations.
*   [filter-lamely](contrib/filter-repo-demos/filter-lamely), a reimplementation of filter-branch, based on filter-repo, offers better performance (but is still slower than filter-repo).
*   A [cheat sheet](Documentation/converting-from-filter-branch.md#cheat-sheet-conversion-of-examples-from-the-filter-branch-manpage) is available for converting `filter-branch` commands.

### BFG Repo Cleaner

*   Limited to a few specific types of rewrites.
*   Its architecture makes it hard to handle more complex rewrites.
*   Contains shortcomings and bugs.
*   [bfg-ish](contrib/filter-repo-demos/bfg-ish), a reimplementation of bfg, offers new features and bugfixes.
*   A [cheat sheet](Documentation/converting-from-bfg-repo-cleaner.md#cheat-sheet-conversion-of-examples-from-bfg) is available for converting BFG commands.

## Simple example, with comparisons

Let's extract a directory `src/` to merge it into another repository:

*   Extract the history of the `src/` directory only.
*   Rename files to `my-module/` (e.g., `src/foo.c` becomes `my-module/src/foo.c`).
*   Rename tags to `my-module-` (e.g., tag `v1.0` becomes `my-module-v1.0`).

### Solving this with filter-repo

Use the following command:
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

*   This is slow and complex.
*   Requires extra steps to clean up old objects.
*   Commit messages are not rewritten.
*   The `--prune-empty` flag can miss commits.
*   OS-specific shell commands can cause portability issues.

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

*   Difficult to maintain and can easily corrupt data.
*   Limited support for special filenames.
*   Difficult to prune empty commits.
*   Commit messages are not rewritten.

## Design rationale behind filter-repo

`filter-repo` addresses several shortcomings of existing tools, including:

1.  **Starting report:** Provides an analysis of the repository.
2.  **Keep vs. remove:** Allows specifying paths to *keep* rather than just remove.
3.  **Renaming:** Easy to rename paths and handle conflicts.
4.  **More intelligent safety:** Encourages using a fresh clone for recovery.
5.  **Auto shrink:** Automatically removes cruft and repacks.
6.  **Clean separation:** Avoids mixing old and rewritten history.
7.  **Versatility:** Provides extensibility for custom tools.
8.  **Old commit references:** Includes the ability to map old IDs.
9.  **Commit message consistency:** Rewrites commit messages.
10. **Become-empty pruning:** Prunes commits that become empty.
11. **Become-degenerate pruning:** Prunes degenerate merges.
12. **Speed:** Designed for speed.

## How do I contribute?

Refer to the [contributing guidelines](Documentation/Contributing.md).

## Is there a Code of Conduct?

The [git Code of Conduct](https://git.kernel.org/pub/scm/git/git.git/tree/CODE_OF_CONDUCT.md) applies.

## Upstream Improvements

Work on `filter-repo` has driven numerous improvements to `fast-export` and `fast-import` and occasionally other commands in core git.