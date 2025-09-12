# Secure Your Code with AI: Claude Code Security Reviewer

**Automate and enhance your code security with the Claude Code Security Reviewer, an AI-powered GitHub Action that identifies vulnerabilities with unmatched accuracy.**  [View the original repository](https://github.com/anthropics/claude-code-security-review).

## Key Features:

*   **AI-Powered Analysis:** Leverages Anthropic's Claude to deeply understand code semantics and identify security flaws.
*   **Diff-Aware Scanning:** Focuses on changes within pull requests for efficient vulnerability detection.
*   **Automated PR Comments:** Directly comments on pull requests, highlighting security findings and offering guidance.
*   **Contextual Understanding:** Goes beyond pattern matching to understand the purpose and potential security implications of the code.
*   **Language Agnostic:** Compatible with any programming language.
*   **Reduced Noise:** Advanced filtering minimizes false positives, focusing on genuine security risks.

## Get Started: Quick Installation

Integrate the Claude Code Security Reviewer into your GitHub Actions workflow with these simple steps:

1.  Add the following configuration to your repository's `.github/workflows/security.yml`:

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

2.  Configure the following action inputs within the `with:` block in the YAML. Ensure the `claude-api-key` input is set to your Anthropic Claude API key.

## Configuration Options

### Action Inputs

| Input | Description | Default | Required |
|-------|-------------|---------|----------|
| `claude-api-key` | Anthropic Claude API key for security analysis. <br>*Note*: This API key needs to be enabled for both the Claude API and Claude Code usage. | None | Yes |
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
|--------|-------------|
| `findings-count` | Total number of security findings |
| `results-file` | Path to the results JSON file |

## How it Works:

1.  **PR Analysis:**  Analyzes the code changes in pull requests.
2.  **Contextual Review:** Evaluates code changes within their context, understanding potential security risks.
3.  **Finding Generation:** Identifies security issues with explanations, severity ratings, and remediation steps.
4.  **False Positive Filtering:** Reduces noise by filtering out low-impact findings.
5.  **PR Comments:** Posts findings as comments on the relevant lines of code.

## Security Analysis Capabilities

*   **Injection Attacks:** SQLi, Command Injection, etc.
*   **Authentication & Authorization:** Broken auth, privilege escalation, etc.
*   **Data Exposure:** Hardcoded secrets, sensitive data logging, etc.
*   **Cryptographic Issues:** Weak algorithms, improper key management.
*   **Input Validation:** Missing validation, improper sanitization.
*   **Business Logic Flaws:** Race conditions, TOCTOU issues.
*   **Configuration Security:** Insecure defaults.
*   **Supply Chain:** Vulnerable dependencies.
*   **Code Execution:** RCE via deserialization, pickle injection, eval injection
*   **Cross-Site Scripting (XSS):** Reflected, stored, and DOM-based XSS

### False Positive Filtering

The tool automatically excludes these types of vulnerabilities:
- Denial of Service vulnerabilities
- Rate limiting concerns
- Memory/CPU exhaustion issues
- Generic input validation without proven impact
- Open redirect vulnerabilities

## Benefits:

*   **Contextual Understanding:** Deep semantic analysis of the code.
*   **Lower False Positives:** Reduces noise by filtering out irrelevant findings.
*   **Detailed Explanations:** Clear explanations of vulnerabilities and how to fix them.
*   **Customization:** Adaptable to your organization's specific security needs.

## Claude Code Integration: /security-review Command

Claude Code ships a `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) that provides the same security analysis capabilities as the GitHub Action workflow, but integrated directly into your Claude Code development environment. To use this, simply run `/security-review` to perform a comprehensive security review of all pending changes.

### Customizing the Command

The default `/security-review` command is designed to work well in most cases, but it can also be customized based on your specific security needs. To do so: 

1.  Copy the [`security-review.md`](https://github.com/anthropics/claude-code-security-review/blob/main/.claude/commands/security-review.md?plain=1) file from this repository to your project's `.claude/commands/` folder. 
2.  Edit `security-review.md` to customize the security analysis. For example, you could add additional organization-specific directions to the false positive filtering instructions. 

## Custom Scanning Configuration

It is also possible to configure custom scanning and false positive filtering instructions, see the [`docs/`](docs/) folder for more details.  

## Testing

To run the test suite, execute:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

For assistance, please:

*   Open an issue in this repository.
*   Consult the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history).

## License

MIT License - see the [LICENSE](LICENSE) file for details.