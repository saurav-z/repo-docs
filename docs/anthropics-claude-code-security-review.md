# Enhance Your Code Security with AI-Powered Reviews

**Proactively identify and address security vulnerabilities in your code with the Anthropic Claude Code Security Review GitHub Action.**  [Learn more at the original repo](https://github.com/anthropics/claude-code-security-review).

## Key Features

*   **AI-Powered Analysis:** Leverages Anthropic's Claude Code for deep semantic understanding and vulnerability detection.
*   **Diff-Aware Scanning:** Analyzes only changed files within pull requests for efficient reviews.
*   **Automated PR Comments:**  Provides direct feedback on pull requests, highlighting security findings.
*   **Contextual Understanding:** Goes beyond pattern matching to understand the code's purpose and potential risks.
*   **Language Agnostic:** Works with code written in any programming language.
*   **Reduced Noise:** Advanced false positive filtering minimizes distractions, focusing on critical vulnerabilities.

## Getting Started

Integrate the Claude Code Security Review into your workflow by adding the following snippet to your repository's `.github/workflows/security.yml`:

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

| Input                       | Description                                                                                                                            | Default                         | Required |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | -------- |
| `claude-api-key`            | Your Anthropic Claude API key.  *Note:* The key must be enabled for both the Claude API and Claude Code usage.                                    | None                            | Yes      |
| `comment-pr`                | Whether to comment on PRs with findings.                                                                                              | `true`                          | No       |
| `upload-results`            | Whether to upload results as artifacts.                                                                                                 | `true`                          | No       |
| `exclude-directories`       | Comma-separated list of directories to exclude from scanning.                                                                           | None                            | No       |
| `claude-model`              | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use. Defaults to Opus 4.1.                      | `claude-opus-4-1-20250805` | No       |
| `claudecode-timeout`        | Timeout for ClaudeCode analysis in minutes.                                                                                             | `20`                            | No       |
| `run-every-commit`          | Run ClaudeCode on every commit (skips cache check). *Warning:* May increase false positives on PRs with many commits.                        | `false`                         | No       |
| `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file.                                                           | None                            | No       |
| `custom-security-scan-instructions`      | Path to custom security scan instructions text file to append to audit prompt.                                           | None                            | No       |

### Action Outputs

| Output           | Description                                        |
| ---------------- | -------------------------------------------------- |
| `findings-count` | Total number of security findings.                 |
| `results-file`   | Path to the results JSON file.                    |

## How It Works

1.  **Pull Request Analysis:** Analyzes the changes in the pull request.
2.  **Contextual Review:** Claude Code examines the code changes in context.
3.  **Finding Generation:** Identifies security issues with explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering:** Advanced filtering removes low-impact or false positive prone findings.
5.  **PR Comments:** Findings are posted as review comments on the specific lines of code.

## Security Analysis Capabilities

### Types of Vulnerabilities Detected

*   Injection Attacks (SQL, Command, LDAP, XPath, NoSQL, XXE)
*   Authentication & Authorization (Broken authentication, privilege escalation, insecure direct object references, bypass logic, session flaws)
*   Data Exposure (Hardcoded secrets, sensitive data logging, information disclosure, PII handling violations)
*   Cryptographic Issues (Weak algorithms, improper key management, insecure random number generation)
*   Input Validation (Missing validation, improper sanitization, buffer overflows)
*   Business Logic Flaws (Race conditions, TOCTOU)
*   Configuration Security (Insecure defaults, missing security headers, permissive CORS)
*   Supply Chain (Vulnerable dependencies, typosquatting risks)
*   Code Execution (RCE via deserialization, pickle injection, eval injection)
*   Cross-Site Scripting (XSS)

### False Positive Filtering

The tool automatically excludes a variety of low-impact and false positive prone findings, including:
- Denial of Service vulnerabilities
- Rate limiting concerns
- Memory/CPU exhaustion issues
- Generic input validation without proven impact
- Open redirect vulnerabilities

Customize filtering as needed for your security needs.

### Benefits Over Traditional SAST

*   **Contextual Understanding:** Understands code semantics and intent.
*   **Lower False Positives:** AI-powered analysis reduces noise.
*   **Detailed Explanations:** Provides clear explanations of vulnerabilities and solutions.
*   **Adaptive Learning:** Customizable to meet your organization's specific security requirements.

## Claude Code Integration: `/security-review` Command

Use the `/security-review` command directly within Claude Code for instant security analysis of your code.

### Customizing the Command

Customize the security review by editing the `security-review.md` file in your project's `.claude/commands/` folder.

## Custom Scanning Configuration

Customize scanning and false positive filtering instructions by following the documentation in the [`docs/`](docs/) folder.

## Testing

Run the test suite:

```bash
cd claude-code-security-review
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging.

## License

MIT License - see [LICENSE](LICENSE) file for details.