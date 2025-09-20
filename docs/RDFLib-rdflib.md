# RDFLib: Your Python Toolkit for Working with RDF Data

**RDFLib is a powerful and versatile Python library that simplifies working with Resource Description Framework (RDF) data, making it easier to build semantic web applications and manage knowledge graphs.**  ([View on GitHub](https://github.com/RDFLib/rdflib))

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

*   **Comprehensive RDF Support:** Parsers and serializers for a wide range of RDF formats, including RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, Trig, JSON-LD, and more.
*   **Graph Interface:**  Provides a consistent `Graph` interface for working with RDF data, enabling easy manipulation and querying.
*   **Flexible Storage Options:** Supports in-memory, persistent (Berkeley DB), and remote SPARQL endpoint storage implementations.  Extensible with plugin support for additional stores.
*   **SPARQL 1.1 Implementation:**  Includes a robust SPARQL 1.1 implementation that supports both queries and update statements, empowering you to interact with knowledge graphs.
*   **Extensible:** Includes SPARQL function extension mechanisms.
*   **Pythonic API:** Designed to be easy to use and integrate with Python projects.

## Getting Started

RDFLib centers around the `Graph` object, which represents a collection of RDF triples (subject, predicate, object).

Here's a quick example:

```python
from rdflib import Graph
g = Graph()
g.parse('http://dbpedia.org/resource/Semantic_Web')

for s, p, o in g:
    print(s, p, o)
```

You can also use namespaces and add triples to a graph:

```python
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, XSD, FOAF

g = Graph()
semweb = URIRef('http://dbpedia.org/resource/Semantic_Web')
type = g.value(semweb, RDFS.label)

g.add((
    URIRef("http://example.com/person/nick"),
    FOAF.givenName,
    Literal("Nick", datatype=XSD.string)
))
```

## RDFlib Family of Packages

The RDFLib community maintains a suite of related Python packages for various RDF-related tasks:

*   [rdflib](https://github.com/RDFLib/rdflib) - The core RDFLib library.
*   [sparqlwrapper](https://github.com/RDFLib/sparqlwrapper) - A SPARQL service wrapper.
*   [pyLODE](https://github.com/RDFLib/pyLODE) - OWL ontology documentation.
*   [pyrdfa3](https://github.com/RDFLib/pyrdfa3) - RDFa parser.
*   [pymicrodata](https://github.com/RDFLib/pymicrodata) - Microdata extraction.
*   [pySHACL](https://github.com/RDFLib/pySHACL) - SHACL validation.
*   [OWL-RL](https://github.com/RDFLib/OWL-RL) - OWL RL profile implementation.

See the complete list: <https://github.com/RDFLib>

## Installation

Install the stable release using `pip`:

```bash
pip install rdflib
```

Install optional dependencies:

```bash
pip install rdflib[berkeleydb,networkx,html,lxml,orjson]
```

Or, install from the Git repository:

```bash
pip install git+https://github.com/rdflib/rdflib@main
```

## Documentation

Find comprehensive documentation at: <https://rdflib.readthedocs.io>

## Versions & Releases

* `main` branch: Current unstable release - version 8 alpha
* `7.2.1`:  Tiny clean up release.
* `7.2.0`:  General fixes and usability improvements.
* `7.1.4`: Tidy-up release.
* `7.1.3`: Current stable release.
* `7.1.1`: Previous stable release.
* `7.0.0`: Previous stable release, supports Python 3.8.1+ only.
* `6.x.y`: Supports Python 3.7+ only.
* `5.x.y`: Supports Python 2.7 and 3.4+.

See <https://github.com/RDFLib/rdflib/releases/> for release details.

## Running Tests

Use `pytest` to run tests:

```bash
poetry install
poetry run pytest
```

Generate a coverage report:

```bash
poetry run pytest --cov --cov-report term --cov-report html
python -m http.server --directory=htmlcov
```

## Contributing

Contributions are welcome!  Please review the [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and [developers guide](https://rdflib.readthedocs.io/en/latest/developers/)

Submit pull requests: <https://github.com/RDFLib/rdflib/pulls>

File issues: <https://github.com/RDFLib/rdflib/issues>

## Support & Contacts

*   **Stack Overflow:** Use the tag `rdflib` for "how do I" questions: <https://stackoverflow.com/questions/tagged/rdflib>
*   **Mailing List:** rdflib-dev mailing list: <https://groups.google.com/group/rdflib-dev>
*   **Chat:** Gitter: <https://gitter.im/RDFLib/rdflib> or Matrix: [#RDFLib_rdflib:gitter.im](https://matrix.to/#/#RDFLib_rdflib:gitter.im)