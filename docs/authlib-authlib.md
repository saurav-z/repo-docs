<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/dark-logo.svg" />
  <img alt="Authlib" src="docs/_static/light-logo.svg" height="68" />
</picture>

</div>

# Authlib: The Ultimate Python Library for OAuth, OpenID Connect, and JWT

**Authlib is a comprehensive Python library empowering developers to build robust and secure authentication and authorization solutions.** This library provides a wide range of features to implement clients and providers for OAuth 1.0, OAuth 2.0, OpenID Connect, and JSON Web Tokens (JWT).  [Explore the original repo](https://github.com/authlib/authlib) to learn more.

[![Build Status](https://github.com/authlib/authlib/workflows/tests/badge.svg)](https://github.com/authlib/authlib/actions)
[![PyPI version](https://img.shields.io/pypi/v/authlib.svg)](https://pypi.org/project/authlib)
[![conda-forge version](https://img.shields.io/conda/v/conda-forge/authlib.svg?label=conda-forge&colorB=0090ff)](https://anaconda.org/conda-forge/authlib)
[![PyPI Downloads](https://static.pepy.tech/badge/authlib/month)](https://pepy.tech/projects/authlib)
[![Code Coverage](https://codecov.io/gh/authlib/authlib/graph/badge.svg?token=OWTdxAIsPI)](https://codecov.io/gh/authlib/authlib)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=authlib_authlib&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=authlib_authlib)

## Key Features

Authlib offers extensive support for building both clients and providers:

*   **Comprehensive Protocol Support:**
    *   OAuth 1.0, including RFC5849.
    *   OAuth 2.0, adhering to numerous RFCs (6749, 6750, 7009, 7523, 7591, 7592, 7636, 7662, 8414, 8628, 9068, 9101, 9207).
    *   OpenID Connect 1.0.
*   **JSON Web Token (JWT) and Related Standards:**
    *   JWS, JWK, JWA, JWT compliant with RFC7515, RFC7516, RFC7517, RFC7518, RFC7519, RFC7638, RFC8037.
*   **Built-in Client Integrations:**
    *   Requests
    *   HTTPX
    *   Flask, Django, Starlette, and FastAPI OAuth clients.
*   **Framework-Specific Provider Implementations:**
    *   Flask
    *   Django
*   **Security and Standards Compliance:** Authlib is designed to adhere to all relevant RFCs and security best practices.

## Migrations

Please note the deprecation of the `authlib.jose` module.  Refer to the migration guide: [Migrating from `authlib.jose` to `joserfc`](https://jose.authlib.org/en/dev/migrations/authlib/)

## Sponsors

<table>
<tr>
<td><img align="middle" width="48" src="https://cdn.auth0.com/website/website/favicons/auth0-favicon.svg"></td>
<td>If you want to quickly add secure token-based authentication to Python projects, feel free to check Auth0's Python SDK and free plan at <a href="https://auth0.com/overview?utm_source=GHsponsor&utm_medium=GHsponsor&utm_campaign=authlib&utm_content=auth">auth0.com/overview</a>.</td>
</tr>
<tr>
<td><img align="middle" width="48" src="https://typlog.com/assets/icon-white.svg"></td>
<td>A blogging and podcast hosting platform with minimal design but powerful features. Host your blog and Podcast with <a href="https://typlog.com/">Typlog.com</a>.
</td>
</tr>
</table>

[**Fund Authlib to access additional features**](https://docs.authlib.org/en/latest/community/funding.html)

## Useful Links

1.  [Homepage](https://authlib.org/).
2.  [Documentation](https://docs.authlib.org/).
3.  [Commercial License](https://authlib.org/plans).
4.  [Blog](https://blog.authlib.org/).
5.  [Twitter](https://twitter.com/authlib).
6.  [StackOverflow](https://stackoverflow.com/questions/tagged/authlib).
7.  [Other Repositories](https://github.com/authlib).
8.  [Tidelift Subscription](https://tidelift.com/subscription/pkg/pypi-authlib?utm_source=pypi-authlib&utm_medium=referral&utm_campaign=links).

## Security Reporting

Report security vulnerabilities directly via email to <me@lepture.com> or through Tidelift for coordinated disclosure.  PGP key fingerprint provided in original README.

## License

Authlib is available under both a BSD license and a commercial license.