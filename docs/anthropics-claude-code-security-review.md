<!--
  This README.md is optimized for SEO and readability, focusing on key features and benefits.
  It includes relevant keywords and a clear call to action (using the GitHub Action as an example).
-->

# AI-Powered Code Security Review with Claude Code

**Enhance your code's security with the power of AI; automatically identify and address vulnerabilities directly within your GitHub pull requests.**  [View the original repository](https://github.com/anthropics/claude-code-security-review).

This GitHub Action leverages Anthropic's Claude Code to provide intelligent, context-aware security analysis for your code. It goes beyond traditional static analysis tools by understanding the *semantics* of your code, leading to more accurate and actionable results.

## Key Features

*   **AI-Driven Analysis:** Employs Claude's advanced reasoning capabilities to detect security vulnerabilities with deep semantic understanding.
*   **Diff-Aware Scanning:** Analyzes only the changed files within a pull request for faster and more efficient security reviews.
*   **Automated PR Comments:** Automatically posts detailed comments on your pull requests, highlighting vulnerabilities and providing remediation guidance.
*   **Contextual Understanding:**  Analyzes code within its context, going beyond simple pattern matching to understand the intent and potential impact of changes.
*   **Language Agnostic:** Compatible with any programming language, ensuring comprehensive security coverage across your projects.
*   **Reduced Noise with Advanced Filtering:**  Minimizes false positives, focusing your attention on the most critical security issues.

##  How It Works

1.  **PR Analysis:** When a pull request is opened, Claude analyzes the code changes.
2.  **Contextual Review:** Claude examines changes, understanding their purpose and security implications.
3.  **Finding Generation:** Security issues are identified with explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering:** Advanced filtering removes noise.
5.  **PR Comments:** Findings are posted as comments on the specific lines of code.

## Types of Vulnerabilities Detected

This tool identifies a wide range of vulnerabilities, including:

*   **Injection Attacks:** SQL injection, command injection, etc.
*   **Authentication & Authorization:** Broken authentication, privilege escalation, etc.
*   **Data Exposure:** Hardcoded secrets, sensitive data logging, etc.
*   **Cryptographic Issues:** Weak algorithms, improper key management, etc.
*   **Input Validation:** Missing validation, improper sanitization, etc.
*   **Business Logic Flaws:** Race conditions, TOCTOU issues, etc.
*   **Configuration Security:** Insecure defaults, missing security headers, etc.
*   **Supply Chain:** Vulnerable dependencies, typosquatting risks.
*   **Code Execution:** RCE via deserialization, pickle injection, eval injection.
*   **Cross-Site Scripting (XSS)**: Reflected, stored, and DOM-based XSS

## Benefits Over Traditional SAST (Static Analysis Security Testing)

*   **Contextual Understanding:** Grasping code semantics and intent, not just matching patterns.
*   **Lower False Positives:** AI-powered analysis reduces noise.
*   **Detailed Explanations:** Providing clear explanations of why something is a vulnerability and how to fix it.
*   **Adaptive Learning:** Customization with organization-specific security requirements.

## Getting Started (GitHub Actions)

Integrate this powerful security review tool into your GitHub workflow with ease.

1.  Add the following snippet to your repository's `.github/workflows/security.yml` file:

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
2.  Replace `{{ secrets.CLAUDE_API_KEY }}` with your Anthropic Claude API key, stored as a GitHub secret.

For more configuration options, see the original [README](https://github.com/anthropics/claude-code-security-review).

## Claude Code Integration: /security-review Command

The tool also offers a `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) within Claude Code, providing similar security analysis capabilities directly in your development environment. Run `/security-review` to perform a security review of all pending changes.

### Customizing the Command

Customize security analysis by:

1.  Copying the [`security-review.md`](https://github.com/anthropics/claude-code-security-review/blob/main/.claude/commands/security-review.md?plain=1) file to your project's `.claude/commands/` folder.
2.  Editing `security-review.md` to modify analysis, such as adding organization-specific directions to false positive filtering.

## Support & Resources

*   For support, open an issue in the [repository](https://github.com/anthropics/claude-code-security-review).
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.
*   MIT License - see the [LICENSE](LICENSE) file for details.