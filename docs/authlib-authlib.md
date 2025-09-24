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

# Authlib: The Ultimate Python Library for OAuth and OpenID Connect

**Authlib simplifies secure authentication and authorization by providing a robust and spec-compliant toolkit for building OAuth and OpenID Connect clients and servers.**  Explore the [original repository](https://github.com/authlib/authlib) for more details.

## Key Features

*   **Comprehensive Protocol Support:** Authlib implements the latest OAuth 1.0, OAuth 2.0, and OpenID Connect 1.0 specifications, ensuring robust compatibility and security.
    *   OAuth 1.0 Protocol (RFC5849)
    *   OAuth 2.0 Framework (RFC6749, RFC6750, RFC7009, RFC7523, RFC7591, RFC7592, RFC7636, RFC7662, RFC8414, RFC8628, RFC9068, RFC9101, RFC9207)
    *   JSON Web Signature (JWS), Encryption (JWE), Key (JWK), Algorithms (JWA), and Token (JWT)
    *   OpenID Connect Core, Discovery, and Dynamic Client Registration 1.0
*   **Client Integrations:** Seamlessly connect to third-party OAuth providers.
    *   Requests: `OAuth1Session`, `OAuth2Session`, OpenID Connect, `AssertionSession`
    *   HTTPX: `AsyncOAuth1Client`, `AsyncOAuth2Client`, OpenID Connect, `AsyncAssertionClient`
    *   Flask, Django, Starlette, and FastAPI OAuth Client support
*   **Server-Side Capabilities:** Build your own OAuth 1.0, OAuth 2.0, and OpenID Connect providers using popular frameworks:
    *   Flask: OAuth 1.0, OAuth 2.0, and OpenID Connect 1.0 providers
    *   Django: OAuth 1.0, OAuth 2.0, and OpenID Connect 1.0 providers
*   **JWS, JWK, JWA, and JWT Support:** Includes robust support for JSON Web Signature (JWS), JSON Web Key (JWK), JSON Web Algorithms (JWA), and JSON Web Token (JWT).
*   **Python 3.9+ Compatible:**  Authlib is designed to work seamlessly with modern Python environments.

## Migrations

Please note that Authlib is deprecating the `authlib.jose` module.  Refer to the following resource for migration guidance:

*   [Migrating from `authlib.jose` to `joserfc`](https://jose.authlib.org/en/dev/migrations/authlib/)

## Sponsors

<!-- Sponsorship details remain as is -->

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

<!-- Security reporting details remain as is -->

## License

<!-- License details remain as is -->