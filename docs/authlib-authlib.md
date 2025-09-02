<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/dark-logo.svg" />
  <img alt="Authlib" src="docs/_static/light-logo.svg" height="68" />
</picture>

</div>

# Authlib: The Ultimate Python Library for OAuth and OpenID Connect

**Authlib empowers developers to easily implement and integrate robust authentication and authorization solutions in their Python applications.** [See the original repo](https://github.com/authlib/authlib)

[![Build Status](https://github.com/authlib/authlib/workflows/tests/badge.svg)](https://github.com/authlib/authlib/actions)
[![PyPI version](https://img.shields.io/pypi/v/authlib.svg)](https://pypi.org/project/authlib)
[![conda-forge version](https://img.shields.io/conda/v/conda-forge/authlib.svg?label=conda-forge&colorB=0090ff)](https://anaconda.org/conda-forge/authlib)
[![PyPI Downloads](https://static.pepy.tech/badge/authlib/month)](https://pepy.tech/projects/authlib)
[![Code Coverage](https://codecov.io/gh/authlib/authlib/graph/badge.svg?token=OWTdxAIsPI)](https://codecov.io/gh/authlib/authlib)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=authlib_authlib&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=authlib_authlib)

Authlib is a comprehensive library designed for building both OAuth and OpenID Connect clients and providers, with support for modern security standards like JWS, JWK, JWA, and JWT.  Compatible with Python 3.9+, Authlib simplifies the complex world of authentication, saving developers valuable time and resources.

## Key Features:

*   **Comprehensive Protocol Support:**
    *   OAuth 1.0 and 2.0
    *   OpenID Connect 1.0
    *   JSON Web Signature (JWS), Encryption (JWE), Key (JWK), and Token (JWT) standards.
*   **Ready-to-Use Client Integrations:** Seamlessly connect to third-party OAuth providers.
    *   Requests
    *   HTTPX
    *   Flask, Django, Starlette, and FastAPI OAuth clients
*   **Provider Building Capabilities:** Build your own OAuth and OpenID Connect servers.
    *   Flask
    *   Django
*   **Security and Standards Compliance:** Built with security best practices and adheres to relevant RFC specifications.

## Useful Links

1.  **Homepage:** <https://authlib.org/>
2.  **Documentation:** <https://docs.authlib.org/>
3.  **Commercial License:** <https://authlib.org/plans>
4.  **Blog:** <https://blog.authlib.org/>
5.  **Twitter:** <https://twitter.com/authlib>
6.  **StackOverflow:** <https://stackoverflow.com/questions/tagged/authlib>
7.  **Other Repositories:** <https://github.com/authlib>
8.  **Subscribe Tidelift:** <https://tidelift.com/subscription/pkg/pypi-authlib?utm_source=pypi-authlib&utm_medium=referral&utm_campaign=links>

## Migrations

Please read the following for migrating from `authlib.jose` module:

-   [Migrating from `authlib.jose` to `joserfc`](https://jose.authlib.org/en/dev/migrations/authlib/)

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

## Security Reporting

Please report security vulnerabilities privately via email to <me@lepture.com> with an optional patch.

PGP Key fingerprint:
```
72F8 E895 A70C EBDF 4F2A DFE0 7E55 E3E0 118B 2B4C
```

Alternatively, use the [Tidelift security contact](https://tidelift.com/security).

## License

Authlib is available under two licenses:

1.  BSD LICENSE
2.  COMMERCIAL-LICENSE

Open-source and closed-source projects are free to use the BSD license. Commercial licenses are also available at [Authlib Plans](https://authlib.org/plans) with more info at <https://authlib.org/support>.