<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/dark-logo.svg" />
  <img alt="Authlib" src="docs/_static/light-logo.svg" height="68" />
</picture>

</div>

# Authlib: The Ultimate Python Library for OAuth, OpenID Connect, and JWT

Authlib is a powerful and versatile Python library, providing a comprehensive toolkit for building robust and secure authentication and authorization systems.  (See the [original repo](https://github.com/authlib/authlib) for source and more details.)

[![Build Status](https://github.com/authlib/authlib/workflows/tests/badge.svg)](https://github.com/authlib/authlib/actions)
[![PyPI version](https://img.shields.io/pypi/v/authlib.svg)](https://pypi.org/project/authlib)
[![conda-forge version](https://img.shields.io/conda/v/conda-forge/authlib.svg?label=conda-forge&colorB=0090ff)](https://anaconda.org/conda-forge/authlib)
[![PyPI Downloads](https://static.pepy.tech/badge/authlib/month)](https://pepy.tech/projects/authlib)
[![Code Coverage](https://codecov.io/gh/authlib/authlib/graph/badge.svg?token=OWTdxAIsPI)](https://codecov.io/gh/authlib/authlib)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=authlib_authlib&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=authlib_authlib)

**Key Features:**

*   **Comprehensive Protocol Support:**
    *   [OAuth 1.0](https://docs.authlib.org/en/latest/basic/oauth1.html) and [OAuth 2.0](https://docs.authlib.org/en/latest/basic/oauth2.html) specifications, including RFCs.
    *   [OpenID Connect 1.0](https://docs.authlib.org/en/latest/specs/oidc.html) support.
    *   Javascript Object Signing and Encryption (JOSE) including JWS, JWK, JWA, JWT.
*   **Client Integrations:**
    *   Seamless integration with popular Python frameworks using Requests, HTTPX, Flask, Django, Starlette, and FastAPI for easy client implementation.
*   **Provider Building:**
    *   Framework-specific providers for Flask and Django, enabling you to build your own OAuth 1.0, OAuth 2.0, and OpenID Connect servers.
*   **JWT, JWK, JWA and JWS Support:**
    *   Full support for JSON Web Tokens, JSON Web Keys, JSON Web Algorithms, and JSON Web Signatures for secure data exchange.
*   **Security Focused:**
    *   Implements industry best practices for secure authentication and authorization.

Authlib is compatible with Python 3.9+.

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

1.  [Homepage](https://authlib.org/).
2.  [Documentation](https://docs.authlib.org/).
3.  [Commercial License](https://authlib.org/plans).
4.  [Blog](https://blog.authlib.org/).
5.  [Twitter](https://twitter.com/authlib).
6.  [StackOverflow](https://stackoverflow.com/questions/tagged/authlib).
7.  [Other Repositories](https://github.com/authlib).
8.  [Tidelift Subscription](https://tidelift.com/subscription/pkg/pypi-authlib?utm_source=pypi-authlib&utm_medium=referral&utm_campaign=links).

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