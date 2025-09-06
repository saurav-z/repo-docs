# Enhance Your Code Security with AI-Powered Reviews: Claude Code Security Reviewer

**Protect your codebase with automated security analysis powered by Anthropic's Claude, identifying vulnerabilities directly in your pull requests.** ([View the original repository](https://github.com/anthropics/claude-code-security-review))

## Key Features

*   ✅ **AI-Powered Analysis:** Leverages Claude's advanced reasoning for deep semantic security analysis.
*   🔍 **Diff-Aware Scanning:** Analyzes only changed files in pull requests, saving time.
*   💬 **Automated PR Comments:** Provides clear security findings and guidance directly in your pull requests.
*   🧠 **Contextual Understanding:** Goes beyond pattern matching to understand the code's purpose and intent.
*   🌐 **Language Agnostic:** Works seamlessly with any programming language.
*   🚫 **Reduced Noise:** Advanced false positive filtering to focus on real vulnerabilities.

## Getting Started

Integrate the Claude Code Security Reviewer into your GitHub Actions workflow with these simple steps:

1.  Add the following to your `.github/workflows/security.yml` file:

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
2.  Configure your Anthropic Claude API Key in your repository secrets.

## Configuration Options

Customize the behavior of the security reviewer with the following options:

### Action Inputs

| Input                         | Description                                                                                                                                                                                          | Default                                 | Required |
| :---------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------- | :------- |
| `claude-api-key`              | Anthropic Claude API key.  Ensure your API key is enabled for both the Claude API and Claude Code usage.                                                                                                | None                                    | Yes      |
| `comment-pr`                  | Whether to comment on PRs with findings.                                                                                                                                                               | `true`                                  | No       |
| `upload-results`              | Whether to upload results as artifacts.                                                                                                                                                                | `true`                                  | No       |
| `exclude-directories`         | Comma-separated list of directories to exclude from scanning.                                                                                                                                          | None                                    | No       |
| `claude-model`                | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use.  Defaults to Opus 4.1.                                                                                   | `claude-opus-4-1-20250805`               | No       |
| `claudecode-timeout`          | Timeout for ClaudeCode analysis in minutes.                                                                                                                                                           | `20`                                    | No       |
| `run-every-commit`            | Run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits.                                                                                    | `false`                                 | No       |
| `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file.                                                                                                                              | None                                    | No       |
| `custom-security-scan-instructions` | Path to custom security scan instructions text file to append to audit prompt.                                                                                                              | None                                    | No       |

### Action Outputs

| Output           | Description                                    |
| :--------------- | :--------------------------------------------- |
| `findings-count` | Total number of security findings.           |
| `results-file`   | Path to the results JSON file.                |

## How It Works

### Architecture

```
claudecode/
├── github_action_audit.py  # Main audit script for GitHub Actions
├── prompts.py              # Security audit prompt templates
├── findings_filter.py      # False positive filtering logic
├── claude_api_client.py    # Claude API client for false positive filtering
├── json_parser.py          # Robust JSON parsing utilities
├── requirements.txt        # Python dependencies
├── test_*.py               # Test suites
└── evals/                  # Eval tooling to test CC on arbitrary PRs
```

### Workflow

1.  **PR Analysis:**  Claude analyzes the pull request diff.
2.  **Contextual Review:** Claude examines the code changes, understanding purpose and potential security implications.
3.  **Finding Generation:** Security issues are identified with explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering:** Advanced filtering reduces noise.
5.  **PR Comments:** Findings are posted as review comments on specific lines of code.

## Security Analysis Capabilities

### Types of Vulnerabilities Detected

*   **Injection Attacks:** SQL, command, LDAP, XPath, NoSQL, XXE
*   **Authentication & Authorization:** Broken authentication, privilege escalation, insecure object references, session flaws
*   **Data Exposure:** Hardcoded secrets, sensitive data logging, information disclosure, PII handling violations
*   **Cryptographic Issues:** Weak algorithms, improper key management, insecure random number generation
*   **Input Validation:** Missing validation, improper sanitization, buffer overflows
*   **Business Logic Flaws:** Race conditions, TOCTOU issues
*   **Configuration Security:** Insecure defaults, missing security headers, permissive CORS
*   **Supply Chain:** Vulnerable dependencies, typosquatting risks
*   **Code Execution:** RCE via deserialization, pickle injection, eval injection
*   **Cross-Site Scripting (XSS):** Reflected, stored, and DOM-based XSS

### False Positive Filtering

The tool automatically filters out low-impact findings, including:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

You can further customize false positive filtering to meet your project's security goals.

### Benefits Over Traditional SAST

*   **Contextual Understanding:** Understands code semantics and intent.
*   **Lower False Positives:**  AI-powered analysis reduces noise.
*   **Detailed Explanations:** Provides clear explanations and fixes.
*   **Adaptive Learning:** Customizable to project-specific requirements.

## Claude Code Integration: `/security-review` Command

Use the `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) directly within your Claude Code environment for on-demand security analysis.

### Customizing the Command

1.  Copy `security-review.md` from the repository to your project's `.claude/commands/` folder.
2.  Edit `security-review.md` to customize the analysis and filtering.

## Custom Scanning Configuration

Explore custom scanning and false positive filtering options in the [`docs/`](docs/) folder.

## Testing

Run the test suite to validate functionality:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging.

## License

MIT License - see [LICENSE](LICENSE) file.