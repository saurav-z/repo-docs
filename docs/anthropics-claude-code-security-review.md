# Enhance Your Code Security with AI-Powered Reviews using Claude Code

**Automatically identify and address security vulnerabilities in your code with the Anthropic Claude Code Security Reviewer, a powerful GitHub Action that uses AI to provide intelligent, context-aware analysis.** Learn more about the technology behind this tool at the original repository: [anthropics/claude-code-security-review](https://github.com/anthropics/claude-code-security-review)

## Key Features

*   🛡️ **AI-Powered Analysis:** Leverages Claude's advanced reasoning for deep semantic understanding of code, detecting vulnerabilities beyond simple pattern matching.
*   🔍 **Diff-Aware Scanning:** Focuses on changed files within pull requests, optimizing analysis and reducing review time.
*   💬 **Automated PR Comments:** Provides direct feedback within your pull requests, highlighting security findings and offering remediation guidance.
*   🧠 **Contextual Understanding:** Analyzes code semantics to understand the purpose and potential security implications of code.
*   🌐 **Language Agnostic:** Supports security analysis across various programming languages, ensuring broad compatibility.
*   ✅ **False Positive Filtering:** Advanced filtering mechanisms reduce noise and focus on critical vulnerabilities.

## Quick Start: Integrate into Your GitHub Workflow

Get started quickly by adding the following snippet to your repository's `.github/workflows/security.yml` file:

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

Customize the action to fit your project's needs.

### Action Inputs

| Input | Description | Default | Required |
|-------|-------------|---------|----------|
| `claude-api-key` | Your Anthropic Claude API key | None | Yes |
| `comment-pr` | Whether to comment on PRs with findings | `true` | No |
| `upload-results` | Whether to upload results as artifacts | `true` | No |
| `exclude-directories` | Comma-separated list of directories to exclude | None | No |
| `claude-model` | Claude model name (see Anthropic's [model names](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names)) | `claude-opus-4-1-20250805` | No |
| `claudecode-timeout` | Timeout for ClaudeCode analysis (minutes) | `20` | No |
| `run-every-commit` | Run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits. | `false` | No |
| `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file | None | No |
| `custom-security-scan-instructions` | Path to custom security scan instructions text file to append to audit prompt | None | No |

### Action Outputs

| Output | Description |
|--------|-------------|
| `findings-count` | Total number of security findings |
| `results-file` | Path to the results JSON file |

## How It Works

1.  **PR Analysis:** The tool analyzes the pull request diff to identify changes.
2.  **Contextual Review:** Claude examines the code changes, understanding their purpose and potential security implications.
3.  **Finding Generation:** Security issues are identified with detailed explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering:** Advanced filtering removes low-impact findings to reduce noise.
5.  **PR Comments:** Findings are posted as review comments on the specific lines of code.

## Security Analysis Capabilities

### Vulnerabilities Detected

*   Injection Attacks: SQL, Command, LDAP, XPath, NoSQL, XXE
*   Authentication & Authorization: Broken authentication, privilege escalation, insecure direct object references, bypass logic, session flaws
*   Data Exposure: Hardcoded secrets, sensitive data logging, information disclosure, PII handling violations
*   Cryptographic Issues: Weak algorithms, improper key management, insecure random number generation
*   Input Validation: Missing validation, improper sanitization, buffer overflows
*   Business Logic Flaws: Race conditions, time-of-check-time-of-use (TOCTOU) issues
*   Configuration Security: Insecure defaults, missing security headers, permissive CORS
*   Supply Chain: Vulnerable dependencies, typosquatting risks
*   Code Execution: RCE via deserialization, pickle injection, eval injection
*   Cross-Site Scripting (XSS): Reflected, stored, and DOM-based XSS

### False Positive Filtering

The tool automatically excludes the following to prioritize high-impact vulnerabilities:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

Customize this filtering as needed for your project.

### Benefits Over Traditional SAST

*   **Contextual Understanding:** Understands code semantics and intent, not just patterns
*   **Lower False Positives:** AI-powered analysis reduces noise
*   **Detailed Explanations:** Provides clear explanations of vulnerabilities
*   **Adaptive Learning:** Customizable with your organization-specific security requirements

## Installation & Setup

Follow the Quick Start guide for GitHub Actions integration.  For local development, see the [evaluation framework documentation](claudecode/evals/README.md).

## Claude Code Integration: /security-review Command

Use the `/security-review` command within your Claude Code environment for in-line security analysis of your code.  

To customize:

1.  Copy the [`security-review.md`](https://github.com/anthropics/claude-code-security-review/blob/main/.claude/commands/security-review.md?plain=1) file to your project's `.claude/commands/` folder.
2.  Edit `security-review.md` to customize the analysis.

## Custom Scanning Configuration

Configure custom scanning and false positive filtering instructions - see the [`docs/`](docs/) folder for more information.

## Testing

To validate the functionality, run the test suite:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

For issues or questions, open an issue in this repository or check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

MIT License - see [LICENSE](LICENSE) file for details.