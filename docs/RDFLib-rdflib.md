# RDFLib: Your Python Toolkit for Working with RDF Data

**RDFLib is a powerful and versatile Python library that empowers you to parse, manipulate, and serialize RDF data, the foundation of the Semantic Web.**  Explore the original repository: [https://github.com/RDFLib/rdflib](https://github.com/RDFLib/rdflib)

![RDFLib Logo](docs/_static/RDFlib.png)

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


## Key Features:

*   **Comprehensive RDF Support:** Parses and serializes RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, Trig, JSON-LD, and even HexTuples.
*   **Flexible Graph Interface:** Provides a Graph interface backed by various Store implementations, including in-memory, persistent (Berkeley DB), and remote SPARQL endpoints.
*   **SPARQL 1.1 Implementation:** Offers robust support for SPARQL 1.1 queries and update statements.
*   **Extensible:** Includes SPARQL function extension mechanisms and the ability to add custom Store implementations via plugins.
*   **Pythonic API:** Designed to be intuitive and easy to use within the Python ecosystem.

## The RDFLib Family of Packages

The RDFLib community maintains a family of related Python packages for various RDF-related tasks:

*   [rdflib](https://github.com/RDFLib/rdflib) - The core RDFLib library.
*   [sparqlwrapper](https://github.com/RDFLib/sparqlwrapper) - A simple wrapper for SPARQL services.
*   [pyLODE](https://github.com/RDFLib/pyLODE) - An OWL ontology documentation tool.
*   [pyrdfa3](https://github.com/RDFLib/pyrdfa3) - RDFa 1.1 distiller/parser library.
*   [pymicrodata](https://github.com/RDFLib/pymicrodata) - A module to extract RDF from HTML5 microdata.
*   [pySHACL](https://github.com/RDFLib/pySHACL) - SHACL validation for RDF graphs.
*   [OWL-RL](https://github.com/RDFLib/OWL-RL) - OWL2 RL Profile implementation.

Explore the complete list of packages: <https://github.com/RDFLib>

## Versions & Releases

*   `main`: Current unstable release (version 8 alpha)
*   `7.1.4`: Tidy-up release, potentially the last 7.x release.
*   `7.1.3`: Current stable release.
*   `7.1.1`: Previous stable release.
*   `7.0.0`: Supports Python 3.8.1+ only.
*   `6.x.y`: Supports Python 3.7+ only.
*   `5.x.y`: Supports Python 2.7 and 3.4+.

See <https://github.com/RDFLib/rdflib/releases/> for release details.

## Documentation

Comprehensive documentation is available at: <https://rdflib.readthedocs.io>

## Installation

Install the stable release using *pip*:

    $ pip install rdflib

Install optional dependencies with extras:

    $ pip install rdflib[berkeleydb,networkx,html,lxml,orjson]

Or, install the package from PyPI:  https://pypi.python.org/pypi/rdflib

### Installation of the current main branch (for developers)

Install from the Git repository:

    $ pip install git+https://github.com/rdflib/rdflib@main

or

    $ pip install -e git+https://github.com/rdflib/rdflib@main#egg=rdflib

or

    $ poetry install  # installs into a poetry-managed venv

or

    $ pip install -e .

## Getting Started

RDFLib makes working with RDF data in Python easy and natural.  The core data structure is the `Graph`:

```python
from rdflib import Graph
g = Graph()
g.parse('http://dbpedia.org/resource/Semantic_Web')

for s, p, o in g:
    print(s, p, o)
```

## Example: Using Namespaces

```python
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, XSD

g = Graph()
semweb = URIRef('http://dbpedia.org/resource/Semantic_Web')
type = g.value(semweb, RDFS.label)
```

And adding a triple:

```python
from rdflib.namespace import FOAF
g.add((
    URIRef("http://example.com/person/nick"),
    FOAF.givenName,
    Literal("Nick", datatype=XSD.string)
))
```

## Features

*   Parsers and serializers for various RDF formats (RDF/XML, N3, Turtle, JSON-LD, etc.).
*   Graph interface with multiple Store implementations.
*   SPARQL 1.1 support.
*   Open-source and maintained on GitHub.
*   Available on PyPI.
*   Integration with a wider "family" of related projects (see above).

## Running Tests

Run tests:
```shell
poetry install
poetry run pytest
```

Run tests and generate a coverage report:
```shell
poetry run pytest --cov
```

View coverage report:
```shell
poetry run pytest --cov --cov-report term --cov-report html
python -m http.server --directory=htmlcov
```

## Contributing

Contribute to RDFLib:  Read the [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and [developers guide](https://rdflib.readthedocs.io/en/latest/developers/).
Submit pull requests:

*   <https://github.com/RDFLib/rdflib/pulls>

Raise issues:

*   <https://github.com/RDFLib/rdflib/issues>

Use Gitpod or Google Cloud Shell for development:

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/RDFLib/rdflib)
[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FRDFLib%2Frdflib&cloudshell_git_branch=main&cloudshell_open_in_editor=README.md)

## Support & Contacts

For general questions:  Use Stack Overflow and tag with `rdflib`.
Existing questions: <https://stackoverflow.com/questions/tagged/rdflib>

Contact maintainers:

*   rdflib-dev mailing list: <https://groups.google.com/group/rdflib-dev>
*   Gitter: [https://gitter.im/RDFLib/rdflib](https://gitter.im/RDFLib/rdflib) or Matrix: [#RDFLib_rdflib:gitter.im](https://matrix.to/#/#RDFLib_rdflib:gitter.im)