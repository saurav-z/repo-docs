<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/dark-logo.svg" />
    <img alt="Authlib" src="docs/_static/light-logo.svg" height="68" />
  </picture>
</div>

# Authlib: The Ultimate Python Library for OAuth and OpenID Connect

**Authlib** is a powerful Python library designed to simplify the implementation of OAuth and OpenID Connect client and server functionality.  [View the original repo](https://github.com/authlib/authlib)

[![Build Status](https://github.com/authlib/authlib/workflows/tests/badge.svg)](https://github.com/authlib/authlib/actions)
[![PyPI version](https://img.shields.io/pypi/v/authlib.svg)](https://pypi.org/project/authlib)
[![conda-forge version](https://img.shields.io/conda/v/conda-forge/authlib.svg?label=conda-forge&colorB=0090ff)](https://anaconda.org/conda-forge/authlib)
[![PyPI Downloads](https://static.pepy.tech/badge/authlib/month)](https://pepy.tech/projects/authlib)
[![Code Coverage](https://codecov.io/gh/authlib/authlib/graph/badge.svg?token=OWTdxAIsPI)](https://codecov.io/gh/authlib/authlib)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=authlib_authlib&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=authlib_authlib)

## Key Features of Authlib

Authlib provides a comprehensive set of features for both client and server implementations, adhering to relevant specifications:

*   **OAuth 1.0 and 2.0 Support:** Full support for both OAuth 1.0 and 2.0 protocols, including all relevant RFC specifications.
*   **OpenID Connect 1.0:**  Comprehensive support for OpenID Connect 1.0, including core, discovery, and dynamic client registration.
*   **JSON Web Token (JWT) and JOSE:** Includes JWS, JWK, JWA, and JWT for secure data transmission and authentication.
*   **Built-in Client Integrations:** Easy integration with popular Python frameworks like Requests, HTTPX, Flask, Django, Starlette, and FastAPI.
*   **Flexible Provider Implementations:** Build custom OAuth 1.0, OAuth 2.0, and OpenID Connect providers for Flask and Django.
*   **Security Focused:**  Includes tools and guidance for secure implementation and reporting of vulnerabilities.
*   **Spec-Compliant:**  Adheres to industry standards for OAuth and OpenID Connect protocols, ensuring interoperability.

## Migrations

Authlib will deprecate `authlib.jose` module. Please read:

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

1.  Homepage: <https://authlib.org/>
2.  Documentation: <https://docs.authlib.org/>
3.  Purchase Commercial License: <https://authlib.org/plans>
4.  Blog: <https://blog.authlib.org/>
5.  Twitter: <https://twitter.com/authlib>
6.  StackOverflow: <https://stackoverflow.com/questions/tagged/authlib>
7.  Other Repositories: <https://github.com/authlib>
8.  Subscribe Tidelift: [https://tidelift.com/subscription/pkg/pypi-authlib](https://tidelift.com/subscription/pkg/pypi-authlib?utm_source=pypi-authlib&utm_medium=referral&utm_campaign=links)

## Security Reporting

Report security issues responsibly.  Contact `me@lepture.com` or use the [Tidelift security contact](https://tidelift.com/security).

## License

Authlib is available under both BSD and commercial licenses.  See [Authlib Plans](https://authlib.org/plans) for details.