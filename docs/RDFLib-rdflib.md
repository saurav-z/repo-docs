# RDFLib: A Powerful Python Library for Working with RDF Data

**RDFLib is your go-to Python library for creating, parsing, and manipulating RDF (Resource Description Framework) data, enabling you to work with semantic web technologies seamlessly.** (Original Repo: [https://github.com/RDFLib/rdflib](https://github.com/RDFLib/rdflib))

[![Build Status](https://github.com/RDFLib/rdflib/actions/workflows/validate.yaml/badge.svg?branch=main)](https://github.com/RDFLib/rdflib/actions?query=branch%3Amain)
[![Documentation Status](https://readthedocs.org/projects/rdflib/badge/?version=latest)](https://rdflib.readthedocs.io/en/latest/?badge=latest)
[![Coveralls branch](https://img.shields.io/coveralls/RDFLib/rdflib/main.svg)](https://coveralls.io/r/RDFLib/rdflib?branch=main)

[![GitHub stars](https://img.shields.io/github/stars/RDFLib/rdflib.svg)](https://github.com/RDFLib/rdflib/stargazers)
[![Downloads](https://pepy.tech/badge/rdflib/week)](https://pepy.tech/project/rdflib)
[![PyPI](https://img.shields.io/pypi/v/rdflib.svg)](https://pypi.python.org/pypi/rdflib)
[![PyPI](https://img.shields.io/pypi/pyversions/rdflib.svg)](https://pypi.python.org/pypi/rdflib)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6845245.svg)](https://doi.org/10.5281/zenodo.6845245)

[![Contribute with Gitpod](https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod)](https://gitpod.io/#https://github.com/RDFLib/rdflib)
[![Gitter](https://badges.gitter.im/RDFLib/rdflib.svg)](https://gitter.im/RDFLib/rdflib?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Matrix](https://img.shields.io/matrix/rdflib:matrix.org?label=matrix.org%20chat)](https://matrix.to/#/#RDFLib_rdflib:gitter.im)

## Key Features

*   **Comprehensive RDF Support:** Parsers and serializers for various RDF formats like RDF/XML, N3, NTriples, Turtle, JSON-LD, and more.
*   **Graph Interface:** A flexible `Graph` interface for managing RDF data.
*   **Multiple Store Implementations:** Supports in-memory, persistent (Berkeley DB), and remote SPARQL endpoint storage.
*   **SPARQL 1.1 Compliance:** Includes a SPARQL 1.1 implementation for querying and updating RDF data.
*   **Extensible:** Plugin-based architecture for custom store implementations and SPARQL function extensions.

## RDFLib Family of Packages

The RDFLib community maintains a collection of related Python packages. Explore them here:  [https://github.com/RDFLib](https://github.com/RDFLib)

## Versions & Releases

* `main` branch is the current unstable release - version 8 alpha
* `7.1.4` tidy-up release, possibly last 7.x release
* `7.1.3` current stable release, small improvements to 7.1.1
* `7.1.2` previously deleted release
* `7.1.1` previous stable release
    * see <https://github.com/RDFLib/rdflib/releases/tag/7.1.1>
* `7.0.0` previous stable release, supports Python 3.8.1+ only.
    * see [Releases](https://github.com/RDFLib/rdflib/releases)
* `6.x.y` supports Python 3.7+ only. Many improvements over 5.0.0
    * see <https://github.com/RDFLib/rdflib/releases/tag/6.3.2>
* `5.x.y` supports Python 2.7 and 3.4+ and is [mostly backwards compatible with 4.2.2](https://rdflib.readthedocs.io/en/stable/upgrade4to5.html).
  * * see <https://github.com/RDFLib/rdflib/releases/tag/5.0.0>

See <https://github.com/RDFLib/rdflib/releases/> for the release details.

## Documentation

Find detailed documentation at: [https://rdflib.readthedocs.io](https://rdflib.readthedocs.io)

## Installation

Install the stable release using pip:

```bash
pip install rdflib
```

Install optional dependencies with extras:

```bash
pip install rdflib[berkeleydb,networkx,html,lxml,orjson]
```

## Getting Started

Example of creating a graph and loading it with RDF data from DBPedia then printing the results:

```python
from rdflib import Graph
g = Graph()
g.parse('http://dbpedia.org/resource/Semantic_Web')

for s, p, o in g:
    print(s, p, o)
```

## Examples

For examples of how to use RDFLib: [./examples](./examples)

## Testing

### Running the tests on the host

```shell
poetry install
poetry run pytest
```

### Running test coverage on the host with coverage report

```shell
poetry run pytest --cov
```

### Viewing test coverage

```shell
poetry run pytest --cov --cov-report term --cov-report html
python -m http.server --directory=htmlcov
```

## Contributing

Contributions are welcome!  Please read the [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and [developers guide](https://rdflib.readthedocs.io/en/latest/developers/) .

*   [Pull Requests](https://github.com/RDFLib/rdflib/pulls)
*   [Issues](https://github.com/RDFLib/rdflib/issues)

## Support & Contacts

*   **For general questions:** Use [Stack Overflow](https://stackoverflow.com/questions/tagged/rdflib) with the tag `rdflib`.
*   **rdflib-dev mailing list:** <https://groups.google.com/group/rdflib-dev>
*   **Chat:**  [gitter](https://gitter.im/RDFLib/rdflib) or via matrix [#RDFLib_rdflib:gitter.im](https://matrix.to/#/#RDFLib_rdflib:gitter.im)