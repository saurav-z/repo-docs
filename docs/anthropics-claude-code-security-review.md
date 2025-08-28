# Enhance Your Code Security with AI: Claude Code Security Reviewer

**Automatically identify and address security vulnerabilities in your code with the power of AI. This GitHub Action utilizes Anthropic's Claude Code to provide intelligent, context-aware security analysis for your pull requests.** Learn more at the [original repository](https://github.com/anthropics/claude-code-security-review).

## Key Features

*   **AI-Powered Security Analysis:** Leverages Claude's advanced reasoning to detect vulnerabilities with deep semantic understanding.
*   **Diff-Aware Scanning:** Focuses on changes within your pull requests for efficiency.
*   **Automated PR Comments:** Provides direct feedback on your pull requests with actionable security findings.
*   **Contextual Understanding:** Goes beyond pattern matching, interpreting code semantics to minimize false positives.
*   **Language Agnostic:** Compatible with all programming languages.
*   **False Positive Filtering:** Reduces noise and focuses on real vulnerabilities through advanced filtering.

## Getting Started

Integrate this action into your repository's `.github/workflows/security.yml` with the following code snippet:

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

**Important:** You'll need to provide your Anthropic Claude API key, ensuring it's enabled for both the Claude API and Claude Code usage.

## Configuration Options

Customize the action to fit your needs with these inputs:

**Action Inputs**

| Input | Description | Default | Required |
|---|---|---|---|
| `claude-api-key` | Your Anthropic Claude API key. This API key needs to be enabled for both the Claude API and Claude Code usage.  | None | Yes |
| `comment-pr` | Whether to comment on PRs with findings | `true` | No |
| `upload-results` | Whether to upload results as artifacts | `true` | No |
| `exclude-directories` | Comma-separated directories to exclude from scanning | None | No |
| `claude-model` | Claude model name. Defaults to Opus 4.1. | `claude-opus-4-1-20250805` | No |
| `claudecode-timeout` | Timeout for ClaudeCode analysis in minutes | `20` | No |
| `run-every-commit` | Run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits. | `false` | No |
| `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file | None | No |
| `custom-security-scan-instructions` | Path to custom security scan instructions text file to append to audit prompt | None | No |

**Action Outputs**

| Output | Description |
|---|---|
| `findings-count` | Total number of security findings |
| `results-file` | Path to the results JSON file |

## How It Works

1.  **PR Analysis:** Analyzes the code changes within a pull request.
2.  **Contextual Review:** Claude examines the changes, considering their purpose and potential security impact.
3.  **Finding Generation:** Identifies security issues with detailed explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering:** Uses advanced filtering to remove low-impact findings and reduce noise.
5.  **PR Comments:** Presents findings directly on the lines of code within the pull request.

## Security Analysis Capabilities

### Vulnerabilities Detected
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

### False Positive Filtering

The tool automatically excludes the following:
*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

You can tune false positive filtering for your security needs.

### Advantages Over Traditional SAST

*   **Contextual Understanding:** Analyzes the code's intent, not just patterns.
*   **Reduced False Positives:** AI-powered analysis reduces noise and focuses on true vulnerabilities.
*   **Detailed Explanations:** Provides clear explanations of issues and how to fix them.
*   **Customizable:** Adaptable to your organization's specific security requirements.

## Integration with Claude Code: `/security-review` Command

Use the `/security-review` slash command within your Claude Code development environment to get the same security analysis as the GitHub Action. Customize this command by modifying the `security-review.md` file located in your project's `.claude/commands/` folder.

## Custom Scanning Configuration

Customize your scan with custom security instructions, more details can be found in the `docs/` folder in the repository.

## Testing

Test the suite with this command:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

MIT License - see [LICENSE](LICENSE) file for details.