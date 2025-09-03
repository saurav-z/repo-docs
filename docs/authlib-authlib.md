<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/dark-logo.svg" />
  <img alt="Authlib" src="docs/_static/light-logo.svg" height="68" />
</picture>

</div>

# Authlib: The Comprehensive Python Library for OAuth and OpenID Connect

**Authlib** simplifies building secure authentication and authorization solutions in your Python projects, supporting OAuth, OpenID Connect, and JSON Web Tokens (JWT).  [See the original repository](https://github.com/authlib/authlib).

[![Build Status](https://github.com/authlib/authlib/workflows/tests/badge.svg)](https://github.com/authlib/authlib/actions)
[![PyPI version](https://img.shields.io/pypi/v/authlib.svg)](https://pypi.org/project/authlib)
[![conda-forge version](https://img.shields.io/conda/v/conda-forge/authlib.svg?label=conda-forge&colorB=0090ff)](https://anaconda.org/conda-forge/authlib)
[![PyPI Downloads](https://static.pepy.tech/badge/authlib/month)](https://pepy.tech/projects/authlib)
[![Code Coverage](https://codecov.io/gh/authlib/authlib/graph/badge.svg?token=OWTdxAIsPI)](https://codecov.io/gh/authlib/authlib)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=authlib_authlib&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=authlib_authlib)


## Key Features

Authlib provides a robust set of features for implementing OAuth and OpenID Connect:

*   **Comprehensive Protocol Support:**  Implements the latest specifications for OAuth 1.0, OAuth 2.0, and OpenID Connect 1.0, including:
    *   OAuth 1.0 & 2.0 Frameworks (RFCs 5849, 6749, 6750, 7009, 7523, 7591, 7592, 7636, 7662, 8414, 8628, 9068, 9101, 9207)
    *   JSON Web Signature (JWS), Encryption (JWE), Key (JWK), and Token (JWT) specifications (RFCs 7515, 7516, 7517, 7518, 7519, 7638, 8037)
    *   OpenID Connect Core, Discovery, and Dynamic Client Registration
*   **Client Integrations:** Simplifies integration with popular Python web frameworks using the built-in client integrations.
    *   `requests`
    *   `HTTPX`
    *   Flask, Django, Starlette, and FastAPI.
*   **Provider Implementations:**  Build your own OAuth and OpenID Connect providers for:
    *   Flask
    *   Django
*   **JWT/JWS/JWK Support:** Includes implementations for JWS, JWK, JWA, and JWT.
*   **Python 3.9+ Compatibility**

## Migrations

Please read about how to migrate from `authlib.jose` module:

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

Report security bugs responsibly. Contact details are:

*   Email: `me@lepture.com` with PGP key:

```
72F8 E895 A70C EBDF 4F2A DFE0 7E55 E3E0 118B 2B4C
```

*   Tidelift Security Contact: <https://tidelift.com/security>

## License

Authlib is available under both:

1.  BSD LICENSE
2.  COMMERCIAL-LICENSE

Use the BSD license for open and closed-source projects. For commercial support, consider a commercial license. More info: <https://authlib.org/support>