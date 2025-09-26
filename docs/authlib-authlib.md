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

# Authlib: Python Library for OAuth and OpenID Connect 

**Authlib is your all-in-one Python solution for building robust and secure OAuth and OpenID Connect clients and servers.** (See the original repo: [https://github.com/authlib/authlib](https://github.com/authlib/authlib))

Authlib simplifies the implementation of authentication and authorization protocols, enabling developers to easily integrate secure token-based authentication into their Python projects. This versatile library supports a wide range of specifications, including OAuth 1.0, OAuth 2.0, OpenID Connect, and more, while also providing comprehensive support for JWS, JWK, JWA, and JWT.

Authlib is compatible with Python 3.9+.

## Key Features

*   **Comprehensive Protocol Support:**
    *   OAuth 1.0 and 2.0
    *   OpenID Connect 1.0
    *   JSON Web Signature (JWS), Encryption (JWE), Key (JWK), and Token (JWT)
    *   Supports all relevant RFC specifications.
*   **Client Integrations:**
    *   Built-in clients for Requests, HTTPX, Flask, Django, Starlette, and FastAPI.
    *   Simplifies integration with third-party OAuth providers.
*   **Server Implementations:**
    *   Providers for Flask and Django, enabling you to build your own OAuth and OpenID Connect servers.
*   **Security-Focused:**
    *   Provides tools to manage JWTs, ensuring secure token handling.
*   **Flexible Licensing:** Offers both BSD and Commercial Licenses for flexibility.

## Migrations

Authlib will deprecate `authlib.jose` module, please read:

- [Migrating from `authlib.jose` to `joserfc`](https://jose.authlib.org/en/dev/migrations/authlib/)

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

*   Homepage: <https://authlib.org/>
*   Documentation: <https://docs.authlib.org/>
*   Commercial License: <https://authlib.org/plans>
*   Blog: <https://blog.authlib.org/>
*   Twitter: <https://twitter.com/authlib>
*   StackOverflow: <https://stackoverflow.com/questions/tagged/authlib>
*   Other Repositories: <https://github.com/authlib>
*   Subscribe Tidelift: [https://tidelift.com/subscription/pkg/pypi-authlib](https://tidelift.com/subscription/pkg/pypi-authlib?utm_source=pypi-authlib&utm_medium=referral&utm_campaign=links)

## Security Reporting

If you found security bugs, please do not send a public issue or patch.
You can send me email at <me@lepture.com>. Attachment with patch is welcome.
My PGP Key fingerprint is:

```
72F8 E895 A70C EBDF 4F2A DFE0 7E55 E3E0 118B 2B4C
```

Or, you can use the [Tidelift security contact](https://tidelift.com/security).
Tidelift will coordinate the fix and disclosure.

## License

Authlib offers two licenses:

1.  BSD LICENSE
2.  COMMERCIAL-LICENSE

Any project, open or closed source, can use the BSD license.
If your company needs commercial support, you can purchase a commercial license at
[Authlib Plans](https://authlib.org/plans). You can find more information at
<https://authlib.org/support>.