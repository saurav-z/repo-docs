<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/dark-logo.svg" />
  <img alt="Authlib" src="docs/_static/light-logo.svg" height="68" />
</picture>

[![Build Status](https://github.com/authlib/authlib/workflows/tests/badge.svg)](https://github.com/authlib/authlib/actions)
[![PyPI version](https://img.shields.io/pypi/v/authlib.svg)](https://pypi.org/project/authlib)
[![conda-forge version](https://img.shields.io/conda/v/conda-forge/authlib.svg?label=conda-forge&colorB=0090ff)](https://anaconda.org/conda-forge/authlib)
[![PyPI Downloads](https://static.pepy.tech/badge/authlib/month)](https://pepy.tech/projects/authlib)
[![Code Coverage](https://codecov.io/gh/authlib/authlib/graph/badge.svg?token=OWTdxAIsPI)](https://codecov.io/gh/authlib/authlib)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=authlib_authlib&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=authlib_authlib)

</div>

# Authlib: Build Secure OAuth and OpenID Connect Solutions in Python

**Authlib empowers developers to implement robust authentication and authorization, offering comprehensive support for OAuth, OpenID Connect, and related standards.** ([View the original repo](https://github.com/authlib/authlib))

Authlib is a versatile Python library designed for building both clients and servers adhering to modern web authentication standards. It includes implementations for JWS, JWK, JWA, and JWT. Authlib is compatible with Python 3.9+ and provides a wide range of features for secure authentication and authorization.

## Key Features

*   **Comprehensive Protocol Support:**
    *   OAuth 1.0 and 2.0
    *   OpenID Connect 1.0
    *   JSON Web Signature (JWS), JSON Web Encryption (JWE), JSON Web Key (JWK), JSON Web Token (JWT)
    *   Extensive support for related RFCs (listed in detail in the original README)
*   **Client Integrations:**
    *   Requests and HTTPX clients for easy integration with third-party OAuth providers.
    *   Support for Flask, Django, Starlette, and FastAPI frameworks.
*   **Provider Implementation:**
    *   Build custom OAuth 1.0, OAuth 2.0, and OpenID Connect providers.
    *   Framework-specific implementations for Flask and Django.

## Migrations

*   Authlib will deprecate the `authlib.jose` module.  Please read:
    *   [Migrating from `authlib.jose` to `joserfc`](https://jose.authlib.org/en/dev/migrations/authlib/)

## Sponsors

[Sponsor information from original README - include details here.]

## Useful Links

1.  Homepage: <https://authlib.org/>
2.  Documentation: <https://docs.authlib.org/>
3.  Purchase Commercial License: <https://authlib.org/plans>
4.  Blog: <https://blog.authlib.org/>
5.  Twitter: <https://twitter.com/authlib>
6.  StackOverflow: <https://stackoverflow.com/questions/tagged/authlib>
7.  Other Repositories: <https://github.com/authlib>
8.  Subscribe Tidelift: [https://tidelift.com/subscription/pkg/pypi-authlib](https://tidelift.com/subscription/pkg/pypi-authlib?utm_source=pypi-authlib&utm_medium=referral&utm_campaign=links).

## Security Reporting

If you discover any security vulnerabilities, please report them privately to <me@lepture.com> with a patch if possible.  You can also use the [Tidelift security contact](https://tidelift.com/security).

## License

Authlib is available under two licenses:

1.  BSD LICENSE
2.  COMMERCIAL LICENSE

The BSD license is available for any project. For commercial support, purchase a commercial license at [Authlib Plans](https://authlib.org/plans).