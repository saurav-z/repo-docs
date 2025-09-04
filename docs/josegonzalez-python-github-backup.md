# 🛡️ GitHub Backup: Secure Your Repositories and Data with Ease

**Easily back up your GitHub organization, user account, and all associated data, ensuring your code and contributions are safe and accessible.** ([GitHub Repository](https://github.com/josegonzalez/python-github-backup))

## Key Features

*   **Comprehensive Backup:** Backs up repositories, starred repos, issues, wikis, and more.
*   **Flexible Options:** Supports various backup types, including full, incremental, and selective backups.
*   **Authentication Support:** Works with password-based authentication, classic tokens, and fine-grained access tokens.
*   **Repository Cloning:** Clones repositories, including support for bare and LFS repositories.
*   **Data Formats:** Saves data in appropriate formats (clones for wikis, JSON files for issues, etc.).
*   **Rate Limiting:** Automatically throttles API requests to avoid hitting GitHub's rate limits.
*   **Docker Support:** Easily run the backup tool within a Docker container.

## Installation

Install using pip:

```bash
pip install github-backup
```

For the latest version:

```bash
pip install git+https://github.com/josegonzalez/python-github-backup.git#egg=github-backup
```

*Note for new Python users: You may need to add the Python installation path to your system's PATH or call the script directly (e.g., `~/.local/bin/github-backup`) to run it from the terminal.*

## Usage

### Basic Help

```bash
github-backup -h
```

### Authentication

*   **Password-based authentication** is deprecated and will fail if you have two-factor authentication enabled.
*   **Classic tokens** are less secure and provide very coarse-grained permissions.
*   **Fine-grained personal access tokens** are recommended for long-running backups.

#### Fine Tokens

1.  Generate a new token in your GitHub settings.
2.  Customize permissions based on your needs (User and Repository permissions are required).

#### SSH

Use `--prefer-ssh` to clone repositories using SSH. You will need SSH authentication configured for your GitHub account.

#### Using the Keychain on Mac OSX

1.  Open Keychain Access and add a new password item.
2.  Provide the "Keychain Item Name" and "Account Name" to `github-backup` using `--keychain-name` and `--keychain-account` arguments.

### Github Rate-limit and Throttling

*   The tool will throttle API requests automatically.
*   You can override the throttle with `--throttle-limit` and `--throttle-pause`.

### Git LFS

Install Git LFS if you use the `--lfs` option: [https://git-lfs.github.com](https://git-lfs.github.com)

### Run in Docker Container

```bash
sudo docker run --rm -v /path/to/backup:/data --name github-backup ghcr.io/josegonzalez/python-github-backup -o /data $OPTIONS $USER
```

## Examples

### Backup All Repositories with a Classic Token
```bash
export ACCESS_TOKEN=SOME-GITHUB-TOKEN
github-backup WhiteHouse --token $ACCESS_TOKEN --organization --output-directory /tmp/white-house --repositories --private
```

### Use a fine-grained access token to backup a single organization repository with everything else
```bash
export FINE_ACCESS_TOKEN=SOME-GITHUB-TOKEN
ORGANIZATION=docker
REPO=cli
github-backup $ORGANIZATION -P -f $FINE_ACCESS_TOKEN -o . --all -O -R $REPO
```

### Quietly and incrementally backup useful Github user data
```bash
export FINE_ACCESS_TOKEN=SOME-GITHUB-TOKEN
GH_USER=YOUR-GITHUB-USER
github-backup -f $FINE_ACCESS_TOKEN --prefer-ssh -o ~/github-backup/ -l error -P -i --all-starred --starred --watched --followers --following --issues --issue-comments --issue-events --pulls --pull-comments --pull-commits --labels --milestones --repositories --wikis --releases --assets --pull-details --gists --starred-gists $GH_USER
```

## Gotchas / Known Issues

*   **Limitations of `--all`:** Does not include cloning private repos (`-P`), forks (`-F`), starred repositories (`--all-starred`),  `--pull-details`, LFS repositories (`--lfs`), gists (`--gists`), or starred gist repos (`--starred-gists`).
*   **Starred Repositories Storage:** Backing up all starred repositories may use a large amount of storage space,  consider just storing links to starred repos in JSON format with `--starred`.
*   **Incremental Backup:** Incremental backups only request new data since the last run (successful or not), which can lead to data loss if previous runs have errors.
*   **Blocking Errors:** Errors, like 403 Forbidden, can block backup runs, which can result in unexpected missing data in an incremental backup.
*   **Starred Public Repo Hooks Blocking:** Using `--all` and `--all-starred` can cause errors when cloning starred public repositories, as the backup will likely block.
*   **"bare" is actually "mirror"**: The `--bare` argument calls git's `clone --mirror` command, which has subtle differences from `bare` cloning.
*   **Starred gists vs starred repo behaviour:**  Starred gists are stored within the same directory as the users own gists. All gist repo directory names are IDs not the gist's name.
*   **Skip existing on incomplete backups:** The `--skip-existing` argument will skip a backup if the directory already exists, even if the backup failed, which may result in unexpected missing data.

## Development

Feel free to contribute with pull requests if you'd like a bugfix or enhancement.

## Testing

This project has no unit tests at this time. To run linting:

```bash
pip install flake8
flake8 --ignore=E501