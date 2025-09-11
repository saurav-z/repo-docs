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

# Authlib: Build Secure OAuth and OpenID Connect Applications in Python

**Authlib** is the go-to Python library for securely implementing OAuth and OpenID Connect protocols, offering robust tools for both client and server-side applications. ([See the original repository](https://github.com/authlib/authlib))

**Key Features:**

*   **Comprehensive Protocol Support:**
    *   OAuth 1.0 and 2.0 (Authorization Framework)
    *   OpenID Connect 1.0
    *   JSON Web Signature (JWS), Encryption (JWE), and Key (JWK)
    *   JSON Web Token (JWT)
*   **Built-in Client Integrations:**
    *   Requests Library:  OAuth 1.0/2.0, OpenID Connect
    *   HTTPX Library: Asynchronous OAuth 1.0/2.0 and OpenID Connect
    *   Framework Support:  Flask, Django, Starlette, and FastAPI OAuth Clients
*   **Provider Implementations:**
    *   Flask: OAuth 1.0/2.0 and OpenID Connect providers
    *   Django: OAuth 1.0/2.0 and OpenID Connect providers
*   **Security-Focused:** Provides robust cryptographic primitives and follows industry best practices for secure authentication and authorization.
*   **Spec-Compliant:**  Adheres strictly to the latest OAuth and OpenID Connect specifications.
*   **Python 3.9+ Compatible**

**Important Links:**

*   **Homepage:** <https://authlib.org/>
*   **Documentation:** <https://docs.authlib.org/>
*   **Commercial License and Support:** <https://authlib.org/plans>
*   **Blog:** <https://blog.authlib.org/>
*   **Twitter:** <https://twitter.com/authlib>
*   **StackOverflow:** <https://stackoverflow.com/questions/tagged/authlib>
*   **Other Repositories:** <https://github.com/authlib>
*   **Tidelift Subscription:** [https://tidelift.com/subscription/pkg/pypi-authlib](https://tidelift.com/subscription/pkg/pypi-authlib?utm_source=pypi-authlib&utm_medium=referral&utm_campaign=links)

**Security Reporting:**

If you discover any security vulnerabilities, please report them directly via email to <me@lepture.com>, including a patch if possible. My PGP key fingerprint is:
```
72F8 E895 A70C EBDF 4F2A DFE0 7E55 E3E0 118B 2B4C
```
Alternatively, you can use the [Tidelift security contact](https://tidelift.com/security) for coordinated disclosure.

**License:**

Authlib is available under the BSD License, suitable for both open-source and closed-source projects.  Commercial licenses are available for projects requiring additional support and features.