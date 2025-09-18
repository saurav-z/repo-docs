<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/dark-logo.svg" />
  <img alt="Authlib" src="docs/_static/light-logo.svg" height="68" />
</picture>

</div>

# Authlib: The Ultimate Python Library for OAuth and OpenID Connect

**Authlib** is a powerful and versatile Python library designed to simplify building OAuth and OpenID Connect clients and servers, providing robust security and compliance. ([View on GitHub](https://github.com/authlib/authlib))

[![Build Status](https://github.com/authlib/authlib/workflows/tests/badge.svg)](https://github.com/authlib/authlib/actions)
[![PyPI version](https://img.shields.io/pypi/v/authlib.svg)](https://pypi.org/project/authlib)
[![conda-forge version](https://img.shields.io/conda/v/conda-forge/authlib.svg?label=conda-forge&colorB=0090ff)](https://anaconda.org/conda-forge/authlib)
[![PyPI Downloads](https://static.pepy.tech/badge/authlib/month)](https://pepy.tech/projects/authlib)
[![Code Coverage](https://codecov.io/gh/authlib/authlib/graph/badge.svg?token=OWTdxAIsPI)](https://codecov.io/gh/authlib/authlib)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=authlib_authlib&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=authlib_authlib)

Authlib supports Python 3.9+.

## Key Features

*   **Comprehensive Protocol Support:**
    *   OAuth 1.0
    *   OAuth 2.0 (including RFCs 6749, 6750, 7009, 7523, 7591, 7592, 7636, 7662, 8414, 8628, 9068, 9101, 9207)
    *   JSON Web Signature (JWS)
    *   JSON Web Key (JWK)
    *   JSON Web Encryption (JWE)
    *   JSON Web Token (JWT)
    *   OpenID Connect 1.0 (Core, Discovery, Dynamic Client Registration)
*   **Built-in Client Integrations:**
    *   Requests (OAuth 1.0, OAuth 2.0, OpenID Connect, Assertion)
    *   HTTPX (Async OAuth 1.0, Async OAuth 2.0, Async OpenID Connect, Async Assertion)
    *   Flask OAuth Client
    *   Django OAuth Client
    *   Starlette OAuth Client
    *   FastAPI OAuth Client
*   **Flexible Provider Support:**
    *   Flask OAuth 1.0 & 2.0 Provider
    *   Flask OpenID Connect 1.0 Provider
    *   Django OAuth 1.0 & 2.0 Provider
    *   Django OpenID Connect 1.0 Provider
*   **JWT, JWK, JWA, and JWS Support:** Includes components for working with JSON Web Standards.

## Migrations

Authlib is deprecating `authlib.jose`. See the migration guide: [Migrating from `authlib.jose` to `joserfc`](https://jose.authlib.org/en/dev/migrations/authlib/)

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

*   **Homepage:** <https://authlib.org/>
*   **Documentation:** <https://docs.authlib.org/>
*   **Commercial License:** <https://authlib.org/plans>
*   **Blog:** <https://blog.authlib.org/>
*   **Twitter:** <https://twitter.com/authlib>
*   **StackOverflow:** <https://stackoverflow.com/questions/tagged/authlib>
*   **Other Repositories:** <https://github.com/authlib>
*   **Tidelift Subscription:** [https://tidelift.com/subscription/pkg/pypi-authlib](https://tidelift.com/subscription/pkg/pypi-authlib?utm_source=pypi-authlib&utm_medium=referral&utm_campaign=links)

## Security Reporting

Report security bugs by emailing <me@lepture.com> with a PGP key (fingerprint: `72F8 E895 A70C EBDF 4F2A DFE0 7E55 E3E0 118B 2B4C`) or use the [Tidelift security contact](https://tidelift.com/security).

## License

Authlib is available under the BSD license. Commercial licenses are available through [Authlib Plans](https://authlib.org/plans).