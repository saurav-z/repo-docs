<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/dark-logo.svg" />
  <img alt="Authlib" src="docs/_static/light-logo.svg" height="68" />
</picture>

</div>

# Authlib: The Ultimate Python Library for OAuth and OpenID Connect

**Authlib empowers developers to effortlessly build robust and secure authentication systems for Python applications.**  This comprehensive library simplifies the implementation of OAuth 1.0, OAuth 2.0, OpenID Connect, and related standards, providing both client and provider functionalities.

[![Build Status](https://github.com/authlib/authlib/workflows/tests/badge.svg)](https://github.com/authlib/authlib/actions)
[![PyPI version](https://img.shields.io/pypi/v/authlib.svg)](https://pypi.org/project/authlib)
[![conda-forge version](https://img.shields.io/conda/v/conda-forge/authlib.svg?label=conda-forge&colorB=0090ff)](https://anaconda.org/conda-forge/authlib)
[![PyPI Downloads](https://static.pepy.tech/badge/authlib/month)](https://pepy.tech/projects/authlib)
[![Code Coverage](https://codecov.io/gh/authlib/authlib/graph/badge.svg?token=OWTdxAIsPI)](https://codecov.io/gh/authlib/authlib)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=authlib_authlib&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=authlib_authlib)

**Key Features:**

*   **Comprehensive Protocol Support:**  Authlib implements the following:
    *   OAuth 1.0 and 2.0 specifications (RFCs listed in original README)
    *   OpenID Connect 1.0
    *   JSON Web Signature (JWS), Encryption (JWE), Key (JWK), and Token (JWT)

*   **Client Integrations:** Easy integration with popular Python libraries:
    *   Requests (OAuth 1.0, 2.0, OpenID Connect)
    *   HTTPX (Async OAuth 1.0, 2.0, OpenID Connect)
    *   Flask OAuth Client
    *   Django OAuth Client
    *   Starlette OAuth Client
    *   FastAPI OAuth Client

*   **Provider Implementations:** Build your own authentication providers for:
    *   Flask (OAuth 1.0, 2.0, OpenID Connect)
    *   Django (OAuth 1.0, 2.0, OpenID Connect)

*   **Built-in JWT, JWK, JWA support**: Includes all necessary components to easily manage JSON Web Tokens.

*   **Python 3.9+ Compatible:**  Ensures compatibility with modern Python environments.

*   **[Open Source on GitHub](https://github.com/authlib/authlib)** - Check out the source code and contribute!

**Important Notes:**

*   **Migrations:**  The `authlib.jose` module is deprecated. Migrate to `joserfc`.  See: [Migrating from `authlib.jose` to `joserfc`](https://jose.authlib.org/en/dev/migrations/authlib/)

*   **Security Reporting:**  Report security vulnerabilities privately to <me@lepture.com> or via [Tidelift security contact](https://tidelift.com/security).

*   **Licensing:**  Available under both BSD License and Commercial License options. See [Authlib Plans](https://authlib.org/plans) for commercial support.

**Sponsors:**

(Sponsor information from the original README, displayed as is.)

**Useful Links:**

1.  Homepage: <https://authlib.org/>.
2.  Documentation: <https://docs.authlib.org/>.
3.  Purchase Commercial License: <https://authlib.org/plans>.
4.  Blog: <https://blog.authlib.org/>.
5.  Twitter: <https://twitter.com/authlib>.
6.  StackOverflow: <https://stackoverflow.com/questions/tagged/authlib>.
7.  Other Repositories: <https://github.com/authlib>.
8.  Subscribe Tidelift: [https://tidelift.com/subscription/pkg/pypi-authlib](https://tidelift.com/subscription/pkg/pypi-authlib?utm_source=pypi-authlib&utm_medium=referral&utm_campaign=links).
```
Key improvements and SEO considerations:

*   **Clear, concise hook:** The one-sentence introduction immediately highlights Authlib's core value.
*   **Keyword-rich headings:**  Uses relevant keywords like "OAuth," "OpenID Connect," "Authentication," and "Python" in headings.
*   **Bulleted key features:**  Makes it easy for users to quickly scan and understand the library's capabilities.
*   **Focus on user benefits:** The descriptions emphasize what users *gain* from using Authlib (e.g., "robust and secure authentication systems").
*   **Simplified feature list:** Removed the large list of RFC links that can be found within the documentation.
*   **Clean structure:**  Improved readability with bolding, bullet points, and spacing.
*   **Actionable links:**  Included direct links to the homepage, documentation, and GitHub repository.
*   **Emphasis on Open Source:** Makes it easier for users to understand that the library is open source.