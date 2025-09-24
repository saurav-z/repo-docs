# Enhance Your Code Security with AI-Powered Reviews: Claude Code Security Reviewer

**Tired of manual code reviews and struggling to catch security vulnerabilities?** Automate your security analysis with the Claude Code Security Reviewer, a GitHub Action powered by Anthropic's Claude, that provides intelligent, context-aware security analysis for pull requests.  [Learn more about the original repo](https://github.com/anthropics/claude-code-security-review).

## Key Features:

*   **AI-Powered Analysis:** Leverages Claude's advanced reasoning for deep semantic understanding of code and accurate vulnerability detection.
*   **Diff-Aware Scanning:** Focuses on changed files in pull requests for faster and more efficient analysis.
*   **Direct PR Comments:** Automatically posts security findings directly within pull requests, highlighting specific lines of code.
*   **Contextual Understanding:** Goes beyond pattern matching to understand the intent and context of your code.
*   **Language Agnostic:** Works seamlessly with any programming language.
*   **Reduced Noise:** Includes advanced filtering to minimize false positives and focus on critical vulnerabilities.

## Getting Started: Quick Installation

Integrate the Claude Code Security Reviewer into your GitHub Actions workflow with ease:

1.  Add the following to your repository's `.github/workflows/security.yml`:

```yaml
name: Security Review

permissions:
  pull-requests: write  # Needed for leaving PR comments
  contents: read

on:
  pull_request:

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha || github.sha }}
          fetch-depth: 2
      
      - uses: anthropics/claude-code-security-review@main
        with:
          comment-pr: true
          claude-api-key: ${{ secrets.CLAUDE_API_KEY }}
```

2.  Ensure you have a valid Anthropic Claude API key and configure it as a GitHub secret ( `CLAUDE_API_KEY`).

## Configuration Options: Tailor the Action

Customize the behavior of the security reviewer to fit your needs:

### Action Inputs

| Input                      | Description                                                                                                                                 | Default                     | Required |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------|----------|
| `claude-api-key`           | Anthropic Claude API key for security analysis. *Note*: This API key needs to be enabled for both the Claude API and Claude Code usage.      | None                        | Yes      |
| `comment-pr`               | Whether to comment on PRs with findings                                                                                                     | `true`                      | No       |
| `upload-results`           | Whether to upload results as artifacts                                                                                                      | `true`                      | No       |
| `exclude-directories`      | Comma-separated list of directories to exclude from scanning                                                                              | None                        | No       |
| `claude-model`             | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use. Defaults to Opus 4.1.                | `claude-opus-4-1-20250805` | No       |
| `claudecode-timeout`       | Timeout for ClaudeCode analysis in minutes                                                                                                   | `20`                        | No       |
| `run-every-commit`         | Run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits.                           | `false`                     | No       |
| `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file | None | No |
| `custom-security-scan-instructions` | Path to custom security scan instructions text file to append to audit prompt | None | No |

### Action Outputs

| Output            | Description                          |
|-------------------|--------------------------------------|
| `findings-count`  | Total number of security findings    |
| `results-file`    | Path to the results JSON file        |

## How It Works:

1.  **PR Analysis:**  The action analyzes the changes in a pull request.
2.  **Contextual Review:** Claude examines the code changes, considering context.
3.  **Finding Generation:** Identifies security issues with explanations and severity ratings.
4.  **False Positive Filtering:** Advanced filtering to reduce noise.
5.  **PR Comments:** Findings are posted as comments within the PR.

## Security Analysis Capabilities:

### Vulnerabilities Detected:

*   Injection Attacks (SQL, Command, LDAP, XPath, NoSQL, XXE)
*   Authentication & Authorization Flaws
*   Data Exposure Risks
*   Cryptographic Issues
*   Input Validation Problems
*   Business Logic Flaws
*   Configuration Security Issues
*   Supply Chain Vulnerabilities
*   Code Execution Risks
*   Cross-Site Scripting (XSS)

### False Positive Filtering:

Automatically excludes:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation
*   Open redirect vulnerabilities

## Claude Code Integration: /security-review Command

Use the `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) for security analysis directly in your Claude Code environment. Customize the command by copying and editing `security-review.md` in your `.claude/commands/` folder.

## Custom Scanning Configuration

Customize the scanning instructions and false-positive filtering by following the instructions in the `/docs/` directory.

## Testing:

Run the test suite:

```bash
cd claude-code-security-review
pytest claudecode -v
```

## Support:

For any issues or questions:

*   Open an issue in this repository
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history)

## License:

MIT License - see [LICENSE](LICENSE) file.