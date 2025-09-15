# AI-Powered Code Security Reviews with Claude Code

**Enhance your code security with the power of AI, automatically identifying and explaining vulnerabilities with the Anthropic Claude Code Security Review GitHub Action.** ([Original Repository](https://github.com/anthropics/claude-code-security-review))

## Key Features

*   **Intelligent Analysis:** Leverages Claude's advanced reasoning for in-depth security vulnerability detection.
*   **Diff-Aware Scanning:** Focuses on changes in pull requests, improving efficiency.
*   **Automated PR Comments:** Posts findings directly in pull requests for immediate feedback.
*   **Contextual Understanding:** Analyzes code semantics, going beyond simple pattern matching.
*   **Language Agnostic:** Works with any programming language supported by Claude.
*   **Reduced Noise:** Advanced false positive filtering to focus on critical issues.

## Installation & Setup

### GitHub Actions

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

## Configuration Options

### Action Inputs

| Input | Description | Default | Required |
|---|---|---|---|
| `claude-api-key` | Anthropic Claude API key for security analysis. *Note*: This API key needs to be enabled for both the Claude API and Claude Code usage. | None | Yes |
| `comment-pr` | Whether to comment on PRs with findings | `true` | No |
| `upload-results` | Whether to upload results as artifacts | `true` | No |
| `exclude-directories` | Comma-separated list of directories to exclude from scanning | None | No |
| `claude-model` | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use. Defaults to Opus 4.1. | `claude-opus-4-1-20250805` | No |
| `claudecode-timeout` | Timeout for ClaudeCode analysis in minutes | `20` | No |
| `run-every-commit` | Run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits. | `false` | No |
| `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file | None | No |
| `custom-security-scan-instructions` | Path to custom security scan instructions text file to append to audit prompt | None | No |

### Action Outputs

| Output | Description |
|---|---|
| `findings-count` | Total number of security findings |
| `results-file` | Path to the results JSON file |

## How It Works

### Workflow

1.  **PR Analysis:** Analyzes the diff of a pull request.
2.  **Contextual Review:** Examines code changes in context.
3.  **Finding Generation:** Identifies security issues with explanations and remediation guidance.
4.  **False Positive Filtering:** Filters out noise to reduce false positives.
5.  **PR Comments:** Posts findings as review comments on relevant lines of code.

## Security Analysis Capabilities

### Vulnerabilities Detected

*   **Injection Attacks:** SQL, command, LDAP, XPath, NoSQL, XXE
*   **Authentication & Authorization:** Broken authentication, privilege escalation, insecure direct object references, bypass logic, session flaws.
*   **Data Exposure:** Hardcoded secrets, sensitive data logging, information disclosure, PII handling violations.
*   **Cryptographic Issues:** Weak algorithms, improper key management, insecure random number generation.
*   **Input Validation:** Missing validation, improper sanitization, buffer overflows.
*   **Business Logic Flaws:** Race conditions, TOCTOU issues.
*   **Configuration Security:** Insecure defaults, missing security headers, permissive CORS.
*   **Supply Chain:** Vulnerable dependencies, typosquatting risks.
*   **Code Execution:** RCE via deserialization, pickle injection, eval injection.
*   **Cross-Site Scripting (XSS):** Reflected, stored, and DOM-based XSS.

### False Positive Filtering

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

## Claude Code Integration: `/security-review` Command

Leverage the `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) within your Claude Code development environment for on-demand security analysis of pending changes.

### Customizing the Command

1.  Copy the [`security-review.md`](https://github.com/anthropics/claude-code-security-review/blob/main/.claude/commands/security-review.md?plain=1) file to your project's `.claude/commands/` folder.
2.  Edit `security-review.md` to customize the security analysis (e.g., add organization-specific directions).

## Custom Scanning Configuration

Configure custom scanning and false positive filtering instructions using the documentation in the [`docs/`](docs/) folder.

## Testing

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history).

## License

MIT License - see [LICENSE](LICENSE) file for details.