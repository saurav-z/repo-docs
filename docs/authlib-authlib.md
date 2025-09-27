<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/dark-logo.svg" />
  <img alt="Authlib" src="docs/_static/light-logo.svg" height="68" />
</picture>

</div>

# Authlib: The Ultimate Python Library for OAuth and OpenID Connect

**Authlib** is a powerful and versatile Python library that simplifies the implementation of OAuth and OpenID Connect client and server applications.  [Check out the original repo](https://github.com/authlib/authlib).

[![Build Status](https://github.com/authlib/authlib/workflows/tests/badge.svg)](https://github.com/authlib/authlib/actions)
[![PyPI version](https://img.shields.io/pypi/v/authlib.svg)](https://pypi.org/project/authlib)
[![conda-forge version](https://img.shields.io/conda/v/conda-forge/authlib.svg?label=conda-forge&colorB=0090ff)](https://anaconda.org/conda-forge/authlib)
[![PyPI Downloads](https://static.pepy.tech/badge/authlib/month)](https://pepy.tech/projects/authlib)
[![Code Coverage](https://codecov.io/gh/authlib/authlib/graph/badge.svg?token=OWTdxAIsPI)](https://codecov.io/gh/authlib/authlib)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=authlib_authlib&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=authlib_authlib)

## Key Features

Authlib provides a comprehensive set of features for building secure and compliant authentication solutions:

*   **Supports Multiple Protocols:**
    *   OAuth 1.0 & 2.0
    *   OpenID Connect 1.0
    *   JSON Web Signature (JWS)
    *   JSON Web Encryption (JWE)
    *   JSON Web Key (JWK)
    *   JSON Web Token (JWT)
*   **Spec-Compliant Implementation:**  Adheres to relevant RFC specifications for OAuth, OpenID Connect, and JOSE.
*   **Built-in Client Integrations:**
    *   Requests (OAuth1Session, OAuth2Session, OpenID Connect, AssertionSession)
    *   HTTPX (AsyncOAuth1Client, AsyncOAuth2Client, OpenID Connect, AsyncAssertionClient)
    *   Flask, Django, Starlette, and FastAPI OAuth clients
*   **Provider Support:**
    *   Flask OAuth 1.0, 2.0, and OpenID Connect providers
    *   Django OAuth 1.0, 2.0, and OpenID Connect providers
*   **JWS, JWK, JWA, JWT included** Easy-to-use cryptography tools.
*   **Python 3.9+ Compatible.**

## Migrations

Authlib will deprecate `authlib.jose` module, please read:

- [Migrating from `authlib.jose` to `joserfc`](https://jose.authlib.org/en/dev/migrations/authlib/)

## Sponsors

[Sponsor information remains as is]

[**Fund Authlib to access additional features**](https://docs.authlib.org/en/latest/community/funding.html)

## Useful Links

1.  Homepage: <https://authlib.org/>.
2.  Documentation: <https://docs.authlib.org/>.
3.  Purchase Commercial License: <https://authlib.org/plans>.
4.  Blog: <https://blog.authlib.org/>.
5.  Twitter: <https://twitter.com/authlib>.
6.  StackOverflow: <https://stackoverflow.com/questions/tagged/authlib>.
7.  Other Repositories: <https://github.com/authlib>.
8.  Subscribe Tidelift: [https://tidelift.com/subscription/pkg/pypi-authlib](https://tidelift.com/subscription/pkg/pypi-authlib?utm_source=pypi-authlib&utm_medium=referral&utm_campaign=links).

## Security Reporting

[Security Reporting information remains as is]

## License

[License information remains as is]