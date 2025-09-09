# Enhance Your Code Security with AI-Powered Reviews

**Automatically identify and address security vulnerabilities in your code with the Claude Code Security Review GitHub Action, powered by Anthropic's advanced AI.** ([Original Repo](https://github.com/anthropics/claude-code-security-review))

## Key Features

*   **AI-Powered Analysis:** Leverages Claude's deep semantic understanding to detect security flaws.
*   **Diff-Aware Scanning:** Focuses on code changes within pull requests for efficient analysis.
*   **Automated PR Comments:** Provides direct feedback on identified vulnerabilities within your pull requests.
*   **Contextual Understanding:** Analyzes code in context, going beyond simple pattern matching.
*   **Language Agnostic:** Works seamlessly with various programming languages.
*   **Reduced Noise:** Advanced filtering minimizes false positives, highlighting critical issues.

## Getting Started

Integrate the security review into your GitHub workflow with the following steps:

1.  Add the following code snippet to your repository's `.github/workflows/security.yml` file:

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

2.  Ensure you have a valid Anthropic Claude API key and store it as a GitHub secret named `CLAUDE_API_KEY`.

## Configuration Options

### Action Inputs

Customize the behavior of the security review with the following input options:

*   `claude-api-key`: Your Anthropic Claude API key (required).
*   `comment-pr`: Enable/disable PR comments (default: `true`).
*   `upload-results`: Enable/disable result artifact upload (default: `true`).
*   `exclude-directories`: Comma-separated list of directories to exclude.
*   `claude-model`: Specify the Claude model (default: `claude-opus-4-1-20250805`).
*   `claudecode-timeout`: Set timeout for ClaudeCode analysis (minutes, default: `20`).
*   `run-every-commit`: Run on every commit (skips cache check, may increase false positives).
*   `false-positive-filtering-instructions`: Path to a custom false positive filtering instructions text file.
*   `custom-security-scan-instructions`: Path to custom security scan instructions text file.

### Action Outputs

Retrieve results and insights:

*   `findings-count`: Total number of security findings.
*   `results-file`: Path to the results JSON file.

## How It Works

1.  **PR Analysis:** Analyzes pull request diffs.
2.  **Contextual Review:** Examines code changes in context.
3.  **Finding Generation:** Identifies security issues with detailed explanations.
4.  **False Positive Filtering:** Reduces noise by removing low-impact findings.
5.  **PR Comments:** Posts findings as review comments.

## Security Analysis Capabilities

### Vulnerabilities Detected

The tool detects a wide range of security vulnerabilities, including:

*   Injection Attacks (SQLi, Command Injection, etc.)
*   Authentication & Authorization Flaws
*   Data Exposure Risks
*   Cryptographic Issues
*   Input Validation Problems
*   Business Logic Errors
*   Configuration Security Weaknesses
*   Supply Chain Vulnerabilities
*   Code Execution Risks (RCE)
*   Cross-Site Scripting (XSS)

### False Positive Filtering

The tool automatically filters out a variety of low-impact and false positive prone findings, focusing on high-impact vulnerabilities, such as:
* Denial of Service vulnerabilities
* Rate limiting concerns
* Memory/CPU exhaustion issues
* Generic input validation without proven impact
* Open redirect vulnerabilities

## Benefits Over Traditional SAST

*   **Semantic Understanding:** Understands code semantics and intent.
*   **Reduced Noise:** AI-powered analysis lowers false positives.
*   **Detailed Explanations:** Provides actionable vulnerability explanations.
*   **Customization:** Customizable with organization-specific requirements.

## Claude Code Integration: /security-review Command

Utilize the `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) within your Claude Code environment for direct security analysis. Customize the command by editing the `security-review.md` file located in your project's `.claude/commands/` folder to tailor the security analysis to your specific needs.

## Custom Scanning Configuration

Customize scanning by using custom security scanning and false positive filtering instructions. Explore the [`docs/`](docs/) folder for more details.

## Testing

Run the test suite to validate functionality:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

Report issues or ask questions:

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.