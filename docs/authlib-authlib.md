<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/dark-logo.svg" />
  <img alt="Authlib" src="docs/_static/light-logo.svg" height="68" />
</picture>

</div>

# Authlib: The Ultimate Python Library for OAuth and OpenID Connect

**Authlib** simplifies building secure authentication and authorization systems in Python, providing robust support for OAuth, OpenID Connect, and related security standards. (**[View the original repo](https://github.com/authlib/authlib)**)

[![Build Status](https://github.com/authlib/authlib/workflows/tests/badge.svg)](https://github.com/authlib/authlib/actions)
[![PyPI version](https://img.shields.io/pypi/v/authlib.svg)](https://pypi.org/project/authlib)
[![conda-forge version](https://img.shields.io/conda/v/conda-forge/authlib.svg?label=conda-forge&colorB=0090ff)](https://anaconda.org/conda-forge/authlib)
[![PyPI Downloads](https://static.pepy.tech/badge/authlib/month)](https://pepy.tech/projects/authlib)
[![Code Coverage](https://codecov.io/gh/authlib/authlib/graph/badge.svg?token=OWTdxAIsPI)](https://codecov.io/gh/authlib/authlib)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=authlib_authlib&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=authlib_authlib)

## Key Features

Authlib is a comprehensive library for implementing various authentication and authorization protocols. Here are some of its key features:

*   **Comprehensive Protocol Support:**
    *   OAuth 1.0 and 2.0
    *   OpenID Connect 1.0
    *   JSON Web Signature (JWS), Encryption (JWE), Key (JWK), and Token (JWT)
    *   Supports a wide range of OAuth 2.0 specifications, including token revocation, dynamic client registration, and more.
*   **Built-in Client Integrations:**
    *   Seamless integration with popular Python web frameworks and libraries including:
        *   Requests
        *   HTTPX
        *   Flask
        *   Django
        *   Starlette
        *   FastAPI
*   **Flexible Provider Development:**
    *   Build your own OAuth 1.0, OAuth 2.0, and OpenID Connect providers.
    *   Framework-specific implementations for Flask and Django.
*   **Security Focused:**
    *   Includes features for handling JWS, JWK, JWA, and JWT for secure token handling.
    *   Provides tools for building secure authentication flows.

## Migrations

Authlib is constantly evolving, so be aware that the `authlib.jose` module is deprecated. Please refer to the [Migrating from `authlib.jose` to `joserfc`](https://jose.authlib.org/en/dev/migrations/authlib/) documentation for details on the migration path.

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
*   **Tidelift:** [https://tidelift.com/subscription/pkg/pypi-authlib](https://tidelift.com/subscription/pkg/pypi-authlib?utm_source=pypi-authlib&utm_medium=referral&utm_campaign=links)

## Security Reporting

Report security bugs privately to <me@lepture.com> or use the [Tidelift security contact](https://tidelift.com/security).

## License

Authlib is available under the BSD License and a commercial license.  For more information, please see [Authlib Plans](https://authlib.org/plans) and <https://authlib.org/support>.