# 🚀 Securely Backup Your GitHub Account with `github-backup`

Effortlessly safeguard your GitHub data by backing up your organizations, repositories, user accounts, and more with `github-backup`!  [View the original repository](https://github.com/josegonzalez/python-github-backup)

## Key Features

*   **Comprehensive Backup:** Back up entire organizations, repositories, or user accounts.
*   **Data Types:** Supports backing up starred repos, issues, wikis, gists, and more.
*   **Flexible Authentication:** Supports various authentication methods, including classic tokens, fine-grained tokens, and app authentication.
*   **Incremental Backups:**  Optionally perform incremental backups to save time and resources.
*   **Customization:** Offers options for specifying output directories, including/excluding specific repositories, languages, and more.
*   **GitHub API Throttling:** Implements rate limiting to avoid API exhaustion.
*   **Docker Support:** Easily run the tool within a Docker container.

## Installation

Install `github-backup` using pip:

```bash
pip install github-backup
```

or install the latest version from GitHub:

```bash
pip install git+https://github.com/josegonzalez/python-github-backup.git#egg=github-backup
```

**Note for Python Newcomers:**
The `github-backup` script may not be in your `$PATH` by default. Add the Python installation path to your `$PATH` or call the script directly, for example, using `~/.local/bin/github-backup`.

## Usage

View the CLI help output:

```bash
github-backup -h
```

Key CLI options:

*   `-u USERNAME`: GitHub username for basic auth.
*   `-p PASSWORD`: Password for basic auth.
*   `-t TOKEN_CLASSIC`: Classic personal access token or path to token.
*   `-f TOKEN_FINE`: Fine-grained personal access token or path to token. (**Recommended**)
*   `-o OUTPUT_DIRECTORY`: Directory for the backup.
*   `-i`: Enable incremental backup.
*   `--all`: Include everything in the backup.
*   `USER`:  Your GitHub username.

### Authentication

**Password-based Authentication:** This method is deprecated.

**Classic Tokens:**  Less secure, but convenient. Generate a personal access token on GitHub.

**Fine-grained Tokens:**  **Recommended** for enhanced security.  Generate a token with specific repository permissions under Settings -> Developer Settings -> Personal access tokens -> Fine-grained Tokens on GitHub. Recommended permissions: User (Read access to followers, starring, and watching) and Repository (Read access to contents, issues, metadata, pull requests, and webhooks).

### Important Notes

*   **Rate Limiting:**  GitHub API limits are 5000 calls per hour. `github-backup` automatically throttles to respect these limits.  You can adjust throttling settings with `--throttle-limit` and `--throttle-pause`.
*   **Git LFS:** If using the `--lfs` option, you must have Git LFS installed.
*   **"All" Argument Limitations:**  The `--all` argument does *not* include private repos, forks, starred repos, pull details, LFS repos, gists, or starred gists by default.
*   **Incremental Backup Considerations:**  Use `-i` with caution.  Errors during a run can cause data loss in subsequent incremental backups.
*   **`--skip-existing`:**  Skips backups if a directory exists, even if the backup was incomplete.

## Examples

Backup all repositories, including private ones using a classic token:

```bash
export ACCESS_TOKEN=SOME-GITHUB-TOKEN
github-backup WhiteHouse --token $ACCESS_TOKEN --organization --output-directory /tmp/white-house --repositories --private
```

Backup all repositories using a fine-grained access token, a single organization repository:

```bash
export FINE_ACCESS_TOKEN=SOME-GITHUB-TOKEN
ORGANIZATION=docker
REPO=cli
github-backup $ORGANIZATION -P -f $FINE_ACCESS_TOKEN -o . --all -O -R $REPO
```

Quietly and incrementally backup useful Github user data:

```bash
export FINE_ACCESS_TOKEN=SOME-GITHUB-TOKEN
GH_USER=YOUR-GITHUB-USER

github-backup -f $FINE_ACCESS_TOKEN --prefer-ssh -o ~/github-backup/ -l error -P -i --all-starred --starred --watched --followers --following --issues --issue-comments --issue-events --pulls --pull-comments --pull-commits --labels --milestones --repositories --wikis --releases --assets --pull-details --gists --starred-gists $GH_USER
```

## Development

Contributions are welcome! See the original repo.

## Contributors

Thank you to all contributors!

[![Contributors](https://contrib.rocks/image?repo=josegonzalez/python-github-backup)](https://github.com/josegonzalez/python-github-backup/graphs/contributors)

## Testing

To run linting:

```bash
pip install flake8
flake8 --ignore=E501