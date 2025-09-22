# Enhance Your Code Security with AI-Powered Reviews

**Automatically identify and remediate security vulnerabilities in your code with the Claude Code Security Review GitHub Action.** [See the original repo](https://github.com/anthropics/claude-code-security-review) for more details.

## Key Features

*   **AI-Powered Analysis**: Leverages Anthropic's Claude's advanced reasoning for deep semantic understanding of code.
*   **Diff-Aware Scanning**: Focuses on changed files in pull requests, optimizing analysis.
*   **Automated PR Comments**: Provides direct feedback on pull requests, highlighting vulnerabilities.
*   **Contextual Understanding**: Analyzes code in context, going beyond simple pattern matching.
*   **Language Agnostic**: Compatible with all programming languages.
*   **Reduced Noise**: Advanced false positive filtering minimizes irrelevant findings.

## Get Started Quickly

Integrate the Claude Code Security Review into your workflow by adding the following to your `.github/workflows/security.yml` file:

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

*   `claude-api-key`: Your Anthropic Claude API key (required).
*   `comment-pr`:  Whether to comment on PRs (default: `true`).
*   `upload-results`: Whether to upload results as artifacts (default: `true`).
*   `exclude-directories`: Comma-separated list of directories to exclude.
*   `claude-model`: Claude model to use (default: `claude-opus-4-1-20250805`).
*   `claudecode-timeout`: Timeout for ClaudeCode analysis in minutes (default: `20`).
*   `run-every-commit`: Run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits (default: `false`).
*   `false-positive-filtering-instructions`: Path to custom false positive filtering instructions text file.
*   `custom-security-scan-instructions`: Path to custom security scan instructions text file to append to audit prompt.

### Action Outputs

*   `findings-count`: Total number of security findings.
*   `results-file`: Path to the results JSON file.

## How It Works

1.  **PR Analysis**: Analyzes the changes in a pull request.
2.  **Contextual Review**: Examines code changes within their context.
3.  **Finding Generation**: Identifies security issues with explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering**: Removes low-impact and common false positives.
5.  **PR Comments**: Findings are posted as comments directly on the code.

## Security Analysis Capabilities

### Vulnerabilities Detected

*   **Injection Attacks:** SQL, command, LDAP, XPath, NoSQL, XXE.
*   **Authentication & Authorization:** Broken authentication, privilege escalation, etc.
*   **Data Exposure:** Hardcoded secrets, information disclosure, PII violations.
*   **Cryptographic Issues:** Weak algorithms, key management issues.
*   **Input Validation:** Missing or improper sanitization, buffer overflows.
*   **Business Logic Flaws:** Race conditions, TOCTOU issues.
*   **Configuration Security:** Insecure defaults, missing security headers, CORS.
*   **Supply Chain:** Vulnerable dependencies, typosquatting risks.
*   **Code Execution:** RCE via deserialization, pickle injection, eval injection.
*   **Cross-Site Scripting (XSS):** Reflected, stored, and DOM-based XSS.

### False Positive Filtering

The tool automatically filters a variety of findings to reduce noise, including:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

Customizable filtering allows tailoring to project-specific needs.

### Benefits Over Traditional SAST

*   **Contextual Understanding:** Understands code semantics and intent.
*   **Lower False Positives:** AI-powered analysis minimizes noise.
*   **Detailed Explanations:** Provides clear explanations and fix recommendations.
*   **Adaptive Learning:** Customizable for organization-specific requirements.

## Advanced Integration

### Claude Code Integration: `/security-review` Command

Claude Code provides a `/security-review` slash command for on-the-fly security analysis. Run the command to perform a comprehensive security review of all pending changes within your Claude Code development environment.

### Custom Scanning Configuration

Customize the security analysis by editing `.claude/commands/security-review.md`. Advanced configuration is possible, see the `docs/` folder for more details.

## Testing

Run the test suite with:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

MIT License - see [LICENSE](LICENSE) file.