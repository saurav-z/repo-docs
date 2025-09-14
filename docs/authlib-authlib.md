<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/dark-logo.svg" />
  <img alt="Authlib" src="docs/_static/light-logo.svg" height="68" />
</picture>

</div>

# Authlib: The Ultimate Python Library for OAuth and OpenID Connect

**Authlib empowers developers to build robust and secure authentication and authorization systems for their Python applications.** [Visit the original repository on GitHub](https://github.com/authlib/authlib)

[![Build Status](https://github.com/authlib/authlib/workflows/tests/badge.svg)](https://github.com/authlib/authlib/actions)
[![PyPI version](https://img.shields.io/pypi/v/authlib.svg)](https://pypi.org/project/authlib)
[![conda-forge version](https://img.shields.io/conda/v/conda-forge/authlib.svg?label=conda-forge&colorB=0090ff)](https://anaconda.org/conda-forge/authlib)
[![PyPI Downloads](https://static.pepy.tech/badge/authlib/month)](https://pepy.tech/projects/authlib)
[![Code Coverage](https://codecov.io/gh/authlib/authlib/graph/badge.svg?token=OWTdxAIsPI)](https://codecov.io/gh/authlib/authlib)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=authlib_authlib&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=authlib_authlib)

Authlib is a comprehensive library for implementing OAuth and OpenID Connect, supporting Python 3.9+ and featuring built-in support for JWS, JWK, JWA, and JWT.

## Key Features

*   **Comprehensive Protocol Support:**
    *   OAuth 1.0 and 2.0
    *   OpenID Connect 1.0
    *   JSON Web Signature (JWS), Encryption (JWE), Key (JWK), and Token (JWT)
*   **Client Integrations:** Easily connect to third-party OAuth providers using:
    *   Requests
    *   HTTPX
    *   Flask, Django, Starlette, and FastAPI
*   **Provider Implementations:** Build your own OAuth and OpenID Connect providers for:
    *   Flask
    *   Django
*   **Security Focused:**  Includes robust security features and adheres to industry best practices.

## Migrations
Please refer to [Migrating from `authlib.jose` to `joserfc`](https://jose.authlib.org/en/dev/migrations/authlib/) for updates.

## Sponsors
[Sponsor information and links]

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