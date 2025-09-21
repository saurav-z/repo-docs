# Secure Your Code with AI: Claude Code Security Reviewer

**Enhance your code's security effortlessly with the AI-powered Claude Code Security Reviewer, which analyzes code changes and identifies vulnerabilities directly within your pull requests.** [Learn More](https://github.com/anthropics/claude-code-security-review)

## Key Features

*   **AI-Powered Analysis:** Leverages Anthropic's Claude to detect vulnerabilities through deep semantic understanding.
*   **Diff-Aware Scanning:** Focuses on changed files within pull requests for efficient analysis.
*   **Automated PR Comments:** Automatically posts security findings as comments directly in your pull requests.
*   **Contextual Understanding:** Analyzes code semantics, going beyond simple pattern matching.
*   **Language Agnostic:** Supports security analysis across various programming languages.
*   **False Positive Filtering:** Reduces noise with advanced filtering to prioritize real vulnerabilities.

## How it Works

1.  **PR Analysis:** When a pull request is opened, Claude analyzes the diff to understand what changed.
2.  **Contextual Review:** Claude examines the code changes in context, understanding the purpose and potential security implications.
3.  **Finding Generation:** Security issues are identified with detailed explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering:** Advanced filtering removes low-impact or false positive prone findings to reduce noise.
5.  **PR Comments:** Findings are posted as review comments on the specific lines of code.

## Quick Start

Integrate the security review action into your GitHub workflow:

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

Customize the action's behavior using the following inputs:

| Input                         | Description                                                                                                                           | Default                      | Required |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | -------- |
| `claude-api-key`              | Anthropic Claude API key for security analysis. *Note*: This API key needs to be enabled for both the Claude API and Claude Code usage. | None                         | Yes      |
| `comment-pr`                  | Whether to comment on PRs with findings                                                                                              | `true`                       | No       |
| `upload-results`              | Whether to upload results as artifacts                                                                                                | `true`                       | No       |
| `exclude-directories`         | Comma-separated list of directories to exclude from scanning                                                                         | None                         | No       |
| `claude-model`                | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use.  | `claude-opus-4-1-20250805` | No       |
| `claudecode-timeout`          | Timeout for ClaudeCode analysis in minutes                                                                                             | `20`                         | No       |
| `run-every-commit`            | Run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits.                  | `false`                      | No       |
| `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file                                                                | None                         | No       |
| `custom-security-scan-instructions` | Path to custom security scan instructions text file to append to audit prompt                                                       | None                         | No       |

**Outputs:**

*   `findings-count`: Total number of security findings.
*   `results-file`: Path to the results JSON file.

## Security Analysis Capabilities

The Claude Code Security Reviewer is designed to identify a wide range of vulnerabilities:

*   **Injection Attacks:** SQL, command, LDAP, XPath, NoSQL, XXE
*   **Authentication & Authorization:** Broken authentication, privilege escalation, insecure direct object references, bypass logic, session flaws
*   **Data Exposure:** Hardcoded secrets, sensitive data logging, information disclosure, PII handling violations
*   **Cryptographic Issues:** Weak algorithms, improper key management, insecure random number generation
*   **Input Validation:** Missing validation, improper sanitization, buffer overflows
*   **Business Logic Flaws:** Race conditions, time-of-check-time-of-use (TOCTOU) issues
*   **Configuration Security:** Insecure defaults, missing security headers, permissive CORS
*   **Supply Chain:** Vulnerable dependencies, typosquatting risks
*   **Code Execution:** RCE via deserialization, pickle injection, eval injection
*   **Cross-Site Scripting (XSS):** Reflected, stored, and DOM-based XSS

**False Positive Filtering:** The tool automatically excludes low-impact findings to focus on high-impact vulnerabilities.

## Claude Code Integration: `/security-review` Command

Integrate the `/security-review` slash command directly into your Claude Code development environment to run security reviews.

### Customizing the Command

1.  Copy [`security-review.md`](https://github.com/anthropics/claude-code-security-review/blob/main/.claude/commands/security-review.md?plain=1) to your project's `.claude/commands/` folder.
2.  Edit `security-review.md` to customize security analysis.

## Custom Scanning Configuration

Configure custom scanning and false positive filtering instructions by referring to the information in the [`docs/`](docs/) folder.

## Testing

Run the test suite:

```bash
cd claude-code-security-review
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.