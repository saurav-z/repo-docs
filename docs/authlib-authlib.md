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

## Authlib: Your All-in-One Python Library for Secure Authentication and Authorization

Authlib is a powerful Python library providing a comprehensive solution for building OAuth and OpenID Connect clients and servers, incorporating JWS, JWK, JWA, and JWT functionalities.  [Explore the original repo](https://github.com/authlib/authlib).

**Key Features:**

*   **Comprehensive Protocol Support:**
    *   OAuth 1.0 and 2.0
    *   OpenID Connect 1.0
    *   JSON Web Signature (JWS) and Encryption (JWE)
    *   JSON Web Key (JWK) and Algorithms (JWA)
    *   JSON Web Token (JWT)
*   **Built-in Client Integrations:**  Easily connect to third-party OAuth providers using requests and HTTPX.
    *   Requests: OAuth1Session, OAuth2Session, OpenID Connect, AssertionSession
    *   HTTPX: AsyncOAuth1Client, AsyncOAuth2Client, OpenID Connect, AsyncAssertionClient
    *   Framework-Specific Clients: Flask, Django, Starlette, FastAPI
*   **Flexible Provider Implementations:** Build your own OAuth 1.0, 2.0, and OpenID Connect providers with ease using:
    *   Flask
    *   Django
*   **Spec-Compliant:** Adheres to relevant RFCs for robust and secure implementations.
*   **Python 3.9+ Compatible:**  Ensuring compatibility with modern Python versions.

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

1.  [Homepage](https://authlib.org/)
2.  [Documentation](https://docs.authlib.org/)
3.  [Purchase Commercial License](https://authlib.org/plans)
4.  [Blog](https://blog.authlib.org/)
5.  [Twitter](https://twitter.com/authlib)
6.  [StackOverflow](https://stackoverflow.com/questions/tagged/authlib)
7.  [Other Repositories](https://github.com/authlib)
8.  [Subscribe Tidelift](https://tidelift.com/subscription/pkg/pypi-authlib?utm_source=pypi-authlib&utm_medium=referral&utm_campaign=links)

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