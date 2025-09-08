# AI-Powered Code Security Reviews with Claude Code

**Enhance your code security with the Anthropic Claude Code Security Reviewer, a GitHub Action that intelligently analyzes code changes for vulnerabilities.** [View the original repository](https://github.com/anthropics/claude-code-security-review).

## Key Features

*   **AI-Powered Security Analysis:** Leverages Claude's advanced reasoning to identify security vulnerabilities with deep semantic understanding.
*   **Diff-Aware Scanning:** Analyzes only the changed files in pull requests for faster and more efficient reviews.
*   **Automated PR Comments:** Automatically comments on pull requests with security findings, providing clear explanations and remediation guidance.
*   **Contextual Understanding:** Goes beyond pattern matching, understanding code semantics and intent.
*   **Language Agnostic:** Works seamlessly with any programming language.
*   **False Positive Filtering:** Reduces noise and focuses on real vulnerabilities with advanced filtering.

## Getting Started

Integrate the security review action into your repository's `.github/workflows/security.yml`:

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

## Configuration Options

### Action Inputs

| Input                     | Description                                                                                                                              | Default                     | Required |
| :------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------- | :------- |
| `claude-api-key`          | Your Anthropic Claude API key. *Note*: This API key needs to be enabled for both the Claude API and Claude Code usage.                   | None                        | Yes      |
| `comment-pr`              | Whether to comment on PRs with findings                                                                                                   | `true`                      | No       |
| `upload-results`          | Whether to upload results as artifacts                                                                                                  | `true`                      | No       |
| `exclude-directories`     | Comma-separated list of directories to exclude from scanning                                                                             | None                        | No       |
| `claude-model`            | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use.                                   | `claude-opus-4-1-20250805` | No       |
| `claudecode-timeout`      | Timeout for ClaudeCode analysis in minutes                                                                                                | `20`                        | No       |
| `run-every-commit`        | Run ClaudeCode on every commit (skips cache check).  *Warning:* May increase false positives on PRs with many commits.                      | `false`                     | No       |
| `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file                                                                   | None                        | No       |
| `custom-security-scan-instructions`  | Path to custom security scan instructions text file to append to audit prompt                                                     | None                        | No       |

### Action Outputs

| Output          | Description                                |
| :-------------- | :----------------------------------------- |
| `findings-count` | Total number of security findings          |
| `results-file`  | Path to the results JSON file            |

## How It Works

1.  **PR Analysis:** Analyzes the pull request diff.
2.  **Contextual Review:** Examines code changes in context.
3.  **Finding Generation:** Identifies security issues with explanations and remediation guidance.
4.  **False Positive Filtering:** Removes low-impact findings.
5.  **PR Comments:** Posts findings as comments on the relevant lines of code.

## Security Analysis Capabilities

### Types of Vulnerabilities Detected

*   Injection Attacks (SQL, command, LDAP, XPath, NoSQL, XXE)
*   Authentication & Authorization (Broken authentication, privilege escalation, etc.)
*   Data Exposure (Hardcoded secrets, sensitive data logging, etc.)
*   Cryptographic Issues (Weak algorithms, improper key management, etc.)
*   Input Validation (Missing validation, improper sanitization, etc.)
*   Business Logic Flaws (Race conditions, TOCTOU issues)
*   Configuration Security (Insecure defaults, missing security headers, etc.)
*   Supply Chain (Vulnerable dependencies, typosquatting risks)
*   Code Execution (RCE via deserialization, pickle injection, eval injection)
*   Cross-Site Scripting (XSS)

### False Positive Filtering

The tool automatically excludes the following types of findings:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

## Benefits Over Traditional SAST

*   **Contextual Understanding:** Understands code semantics and intent.
*   **Lower False Positives:** Reduces noise with AI-powered analysis.
*   **Detailed Explanations:** Provides clear explanations and how to fix issues.
*   **Adaptive Learning:** Customizable for organization-specific security requirements.

## Installation & Setup

Follow the Quick Start guide to integrate the action into your GitHub workflow.

## Claude Code Integration: /security-review Command

The `/security-review` slash command provides the same security analysis as the GitHub Action, directly in your Claude Code development environment. Run `/security-review` to review all pending changes.

### Customizing the Command

1.  Copy `security-review.md` from this repository's `.claude/commands/` folder to your project's `.claude/commands/` folder.
2.  Edit `security-review.md` to customize the security analysis.

## Custom Scanning Configuration

Configure custom scanning and false positive filtering instructions in the [`docs/`](docs/) folder.

## Testing

Run the test suite to validate functionality:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

*   Open an issue in this repository
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history)

## License

MIT License - see [LICENSE](LICENSE) file for details.