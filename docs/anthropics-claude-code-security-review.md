# Enhance Your Code Security with AI-Powered Reviews: Claude Code Security Reviewer

**Proactively identify and address security vulnerabilities in your code with the AI-driven Claude Code Security Reviewer.**  ([Original Repo](https://github.com/anthropics/claude-code-security-review))

## Key Features

*   **AI-Powered Security Analysis:** Leverages Anthropic's Claude to deeply analyze code for vulnerabilities.
*   **Differential Scanning:** Focuses on changes within pull requests, streamlining the review process.
*   **Automated Pull Request Comments:**  Provides direct feedback on identified security issues within the pull request.
*   **Contextual Code Understanding:** Goes beyond simple pattern matching, interpreting code semantics for accurate vulnerability detection.
*   **Language Agnostic:** Compatible with a wide range of programming languages.
*   **Intelligent False Positive Reduction:** Advanced filtering minimizes noise, highlighting genuine security threats.

## Getting Started

Integrate the Claude Code Security Reviewer into your GitHub Actions workflow with these simple steps:

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

2.  Configure the necessary `claude-api-key` using your Anthropic Claude API key.

## Configuration Options

Customize the behavior of the security scanner with these inputs:

| Input                          | Description                                                                                                                  | Default                  | Required |
| :----------------------------- | :--------------------------------------------------------------------------------------------------------------------------- | :----------------------- | :------- |
| `claude-api-key`               | Your Anthropic Claude API key. Requires both Claude API and Claude Code usage enabled.                                       | None                     | Yes      |
| `comment-pr`                   | Enables or disables commenting on pull requests with findings.                                                              | `true`                   | No       |
| `upload-results`               | Enables or disables uploading results as artifacts.                                                                         | `true`                   | No       |
| `exclude-directories`          | Comma-separated list of directories to exclude from scanning.                                                              | None                     | No       |
| `claude-model`                 | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use.                    | `claude-opus-4-1-20250805` | No       |
| `claudecode-timeout`           | Timeout for ClaudeCode analysis in minutes.                                                                                   | `20`                     | No       |
| `run-every-commit`             | Enables running ClaudeCode on every commit (skips cache). Use with caution on PRs with many commits.                         | `false`                  | No       |
| `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file.                                                     | None                     | No       |
| `custom-security-scan-instructions` | Path to custom security scan instructions text file to append to audit prompt.                                           | None                     | No       |

### Outputs

| Output          | Description                         |
| :-------------- | :---------------------------------- |
| `findings-count` | Total number of security findings.  |
| `results-file`   | Path to the results JSON file.      |

## How It Works

1.  **Pull Request Analysis:** On pull request creation, the action analyzes the code changes.
2.  **Contextual Review:** Claude examines changes within context.
3.  **Finding Generation:** Security issues are identified with explanations, severity, and remediation guidance.
4.  **False Positive Filtering:**  Low-impact issues are filtered out to reduce noise.
5.  **Pull Request Comments:** Findings are posted as comments on the relevant code lines.

## Security Analysis Capabilities

### Vulnerabilities Detected

*   Injection Attacks (SQL, Command, LDAP, XPath, NoSQL, XXE)
*   Authentication & Authorization (Broken Auth, Privilege Escalation, etc.)
*   Data Exposure (Hardcoded Secrets, Information Disclosure, PII Handling)
*   Cryptographic Issues (Weak Algorithms, Key Management, RNG)
*   Input Validation (Missing Validation, Buffer Overflows)
*   Business Logic Flaws (Race Conditions, TOCTOU)
*   Configuration Security (Insecure Defaults, Missing Headers)
*   Supply Chain (Vulnerable Dependencies, Typosquatting)
*   Code Execution (RCE via Deserialization, Pickle Injection)
*   Cross-Site Scripting (XSS)

### False Positive Filtering

The tool automatically excludes:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

## Benefits over Traditional SAST

*   **Contextual Understanding:** Analyzes code semantics and intent.
*   **Reduced False Positives:** AI-powered analysis reduces noise.
*   **Detailed Explanations:** Clear explanations and remediation guidance.
*   **Customizable:** Adaptable to your organization's security needs.

## Claude Code Integration: /security-review Command

The tool ships with a `/security-review` command that allows you to perform security reviews within your Claude Code development environment. Simply type `/security-review` to analyze all pending changes.

### Customizing the Command

1.  Copy the [`security-review.md`](https://github.com/anthropics/claude-code-security-review/blob/main/.claude/commands/security-review.md?plain=1) file to your project's `.claude/commands/` folder.
2.  Edit `security-review.md` to add custom directions.

## Custom Scanning Configuration

Customize scan instructions and false positive filtering. See the [`docs/`](docs/) folder for details.

## Testing

Validate the tool's functionality with:

```bash
cd claude-code-security-review
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history).

## License

MIT License - See the [LICENSE](LICENSE) file.