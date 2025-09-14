# Enhance Your Code Security with AI-Powered Reviews

**Automate your code security with Anthropic's Claude Code Security Reviewer, a GitHub Action that identifies vulnerabilities directly within your pull requests.** [Learn more about the project](https://github.com/anthropics/claude-code-security-review).

## Key Features

*   **AI-Powered Analysis:** Leverages Anthropic's Claude to deeply understand code semantics and identify potential security risks.
*   **Diff-Aware Scanning:** Focuses analysis on the specific code changes within your pull requests, saving time and resources.
*   **Automated PR Comments:**  Provides clear, actionable feedback directly within your pull requests, highlighting vulnerabilities and offering remediation guidance.
*   **Contextual Understanding:** Analyzes code within its context, reducing false positives by understanding the intent of the code.
*   **Language Agnostic:** Compatible with any programming language, providing broad coverage for your projects.
*   **Reduced Noise:** Advanced false positive filtering to focus your attention on the most critical vulnerabilities.

## Getting Started: Integrate in Minutes

Add the following to your repository's `.github/workflows/security.yml`:

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

Customize the behavior of the security review with these inputs:

**Action Inputs:**

*   `claude-api-key`: (Required) Your Anthropic Claude API key. *Note: This API key needs to be enabled for both the Claude API and Claude Code usage.*
*   `comment-pr`:  Whether to comment on PRs with findings (defaults to `true`).
*   `upload-results`: Whether to upload results as artifacts (defaults to `true`).
*   `exclude-directories`: Comma-separated list of directories to exclude from scanning.
*   `claude-model`: Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use. Defaults to Opus 4.1.
*   `claudecode-timeout`: Timeout for ClaudeCode analysis in minutes (defaults to `20`).
*   `run-every-commit`: Run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits.
*   `false-positive-filtering-instructions`: Path to custom false positive filtering instructions text file.
*   `custom-security-scan-instructions`: Path to custom security scan instructions text file to append to audit prompt.

**Action Outputs:**

*   `findings-count`: Total number of security findings.
*   `results-file`: Path to the results JSON file.

## How It Works

1.  **PR Analysis:** The action analyzes the pull request diff to identify changed code.
2.  **Contextual Review:** Claude examines the changed code, understanding its purpose and potential security implications.
3.  **Finding Generation:** Security issues are identified with detailed explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering:** Advanced filtering reduces noise by eliminating less critical findings.
5.  **PR Comments:** Findings are posted as review comments directly on the affected lines of code.

## Security Analysis Capabilities

The Claude Code Security Reviewer detects a wide range of vulnerabilities:

*   **Injection Attacks:** SQL, command, LDAP, XPath, NoSQL, XXE.
*   **Authentication & Authorization:** Broken authentication, privilege escalation, insecure object references, bypass logic, session flaws.
*   **Data Exposure:** Hardcoded secrets, sensitive data logging, information disclosure, PII handling violations.
*   **Cryptographic Issues:** Weak algorithms, improper key management, insecure random number generation.
*   **Input Validation:** Missing validation, improper sanitization, buffer overflows.
*   **Business Logic Flaws:** Race conditions, TOCTOU issues.
*   **Configuration Security:** Insecure defaults, missing security headers, permissive CORS.
*   **Supply Chain:** Vulnerable dependencies, typosquatting risks.
*   **Code Execution:** RCE via deserialization, pickle injection, eval injection.
*   **Cross-Site Scripting (XSS):** Reflected, stored, and DOM-based XSS.

### False Positive Filtering

The tool automatically excludes common false positives to focus on high-impact vulnerabilities, including:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

You can customize false positive filtering instructions as needed.

### Benefits Over Traditional SAST

*   **Contextual Understanding:** Goes beyond pattern matching to understand code semantics and intent.
*   **Lower False Positives:** AI-powered analysis reduces noise by understanding when code is actually vulnerable.
*   **Detailed Explanations:** Provides clear explanations of why something is a vulnerability and how to fix it.
*   **Adaptive Learning:** Customizable for your specific security requirements.

## Claude Code Integration: `/security-review` Command

Use the `/security-review` slash command within your Claude Code development environment for instant security analysis of your code changes.

### Customizing the Command

1.  Copy the [`security-review.md`](https://github.com/anthropics/claude-code-security-review/blob/main/.claude/commands/security-review.md?plain=1) file to your project's `.claude/commands/` folder.
2.  Edit `security-review.md` to customize your security analysis.

## Custom Scanning Configuration

For more details on custom scanning and false positive filtering, please refer to the [`docs/`](docs/) folder.

## Testing

Run the test suite:

```bash
cd claude-code-security-review
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

MIT License - see [LICENSE](LICENSE) file for details.