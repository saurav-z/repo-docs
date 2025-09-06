<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/dark-logo.svg" />
    <img alt="Authlib" src="docs/_static/light-logo.svg" height="68" />
  </picture>
</div>

[![Build Status](https://github.com/authlib/authlib/workflows/tests/badge.svg)](https://github.com/authlib/authlib/actions)
[![PyPI version](https://img.shields.io/pypi/v/authlib.svg)](https://pypi.org/project/authlib)
[![conda-forge version](https://img.shields.io/conda/v/conda-forge/authlib.svg?label=conda-forge&colorB=0090ff)](https://anaconda.org/conda-forge/authlib)
[![PyPI Downloads](https://static.pepy.tech/badge/authlib/month)](https://pepy.tech/projects/authlib)
[![Code Coverage](https://codecov.io/gh/authlib/authlib/graph/badge.svg?token=OWTdxAIsPI)](https://codecov.io/gh/authlib/authlib)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=authlib_authlib&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=authlib_authlib)

## Authlib: Your Go-To Python Library for Secure Authentication

Authlib is a powerful and versatile Python library designed to simplify the implementation of OAuth and OpenID Connect (OIDC) protocols, including JWS, JWK, JWA, and JWT.

**[Explore the Authlib GitHub Repository](https://github.com/authlib/authlib)**

**Key Features:**

*   **Comprehensive Protocol Support:**
    *   OAuth 1.0 and 2.0
    *   OpenID Connect 1.0
    *   JSON Web Signature (JWS), Encryption (JWE), Key (JWK), and Token (JWT)
*   **Client Integrations:** Seamlessly connect with various OAuth providers using built-in clients for:
    *   Requests
    *   HTTPX
    *   Flask, Django, Starlette, and FastAPI
*   **Provider Implementations:** Easily build your own OAuth and OIDC providers using:
    *   Flask
    *   Django
*   **Spec-Compliant:** Adheres to industry standards, ensuring secure and reliable authentication.
*   **Python Compatibility:** Compatible with Python 3.9+

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

1.  Homepage: <https://authlib.org/>
2.  Documentation: <https://docs.authlib.org/>
3.  Purchase Commercial License: <https://authlib.org/plans>
4.  Blog: <https://blog.authlib.org/>
5.  Twitter: <https://twitter.com/authlib>
6.  StackOverflow: <https://stackoverflow.com/questions/tagged/authlib>
7.  Other Repositories: <https://github.com/authlib>
8.  Subscribe Tidelift: [https://tidelift.com/subscription/pkg/pypi-authlib](https://tidelift.com/subscription/pkg/pypi-authlib?utm_source=pypi-authlib&utm_medium=referral&utm_campaign=links).

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