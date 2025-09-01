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

# Authlib: Your Go-To Python Library for OAuth, OpenID Connect, and JWT

**Authlib simplifies building secure and robust authentication and authorization systems in your Python applications.**  ([View on GitHub](https://github.com/authlib/authlib))

Authlib provides a comprehensive and spec-compliant toolkit for implementing OAuth and OpenID Connect clients and providers, along with robust support for JSON Web Tokens (JWT) and related standards.  It supports Python 3.9+.

## Key Features

*   **Comprehensive Protocol Support:**
    *   OAuth 1.0 and 2.0
    *   OpenID Connect 1.0
    *   JWT (JWS, JWK, JWA, JWT) and related RFCs
*   **Client Integrations:** Seamlessly integrate with popular frameworks using built-in client integrations:
    *   `requests`
    *   `httpx` (async)
    *   Flask, Django, Starlette, and FastAPI
*   **Provider Implementations:** Build your own OAuth and OpenID Connect providers for:
    *   Flask
    *   Django
*   **Security Focused:**  Robust implementation of security standards.
*   **Easy to Use:** Provides a clean and intuitive API for both client and provider implementations.
*   **Well-Documented:** Extensive documentation to get you started quickly.

## Migrations

Authlib will deprecate `authlib.jose` module. Please see [Migrating from `authlib.jose` to `joserfc`](https://jose.authlib.org/en/dev/migrations/authlib/).

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
*   **Tidelift Subscription:** [https://tidelift.com/subscription/pkg/pypi-authlib](https://tidelift.com/subscription/pkg/pypi-authlib?utm_source=pypi-authlib&utm_medium=referral&utm_campaign=links).

## Security Reporting

If you discover a security vulnerability, please report it privately to <me@lepture.com>.  Include a patch if possible.  My PGP Key fingerprint is:

```
72F8 E895 A70C EBDF 4F2A DFE0 7E55 E3E0 118B 2B4C
```

Alternatively, you can use the [Tidelift security contact](https://tidelift.com/security).

## License

Authlib offers two licenses:

1.  BSD LICENSE
2.  COMMERCIAL-LICENSE

Open-source projects and commercial projects can use the BSD license.
For commercial support, purchase a commercial license at [Authlib Plans](https://authlib.org/plans). Learn more at <https://authlib.org/support>.