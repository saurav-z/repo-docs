# RDFLib: A Powerful Python Library for Working with RDF Data

**RDFLib is your go-to Python library for parsing, manipulating, and serializing RDF (Resource Description Framework) data, empowering you to build semantic web applications with ease.**  [Explore the original repository](https://github.com/RDFLib/rdflib).

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

*   **Comprehensive RDF Support:** Parses and serializes RDF data in various formats, including RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, Trig, JSON-LD, and HexTuples.
*   **Flexible Graph Interface:** Provides a `Graph` interface that can be backed by multiple store implementations.
*   **Versatile Store Implementations:** Offers store implementations for in-memory storage, persistent disk storage (Berkeley DB), and remote SPARQL endpoints. Supports additional stores via plugins.
*   **SPARQL 1.1 Compliance:**  Includes a SPARQL 1.1 implementation, supporting both queries and update statements.
*   **SPARQL Extension:** Provides SPARQL function extension mechanisms.

## RDFLib Family of Packages

The RDFLib community maintains various related Python packages:

*   [rdflib](https://github.com/RDFLib/rdflib) - The core RDFLib library.
*   [sparqlwrapper](https://github.com/RDFLib/sparqlwrapper) - A SPARQL service wrapper.
*   [pyLODE](https://github.com/RDFLib/pyLODE) - An OWL ontology documentation tool.
*   [pyrdfa3](https://github.com/RDFLib/pyrdfa3) - RDFa 1.1 distiller/parser.
*   [pymicrodata](https://github.com/RDFLib/pymicrodata) - Microdata extraction from HTML5.
*   [pySHACL](https://github.com/RDFLib/pySHACL) - SHACL validation in Python.
*   [OWL-RL](https://github.com/RDFLib/OWL-RL) - OWL2 RL Profile implementation.

Explore all packages: <https://github.com/RDFLib>

## Versions & Releases

*   `main`: Current unstable release (version 8 alpha).
*   `7.1.4`: Tidy-up release.
*   `7.1.3`: Current stable release.
*   `7.1.1` Previous stable release.
*   `7.0.0`: Previous stable release, supports Python 3.8.1+.
*   `6.x.y`: Supports Python 3.7+ only.
*   `5.x.y`: Supports Python 2.7 and 3.4+.

See [Releases](https://github.com/RDFLib/rdflib/releases/) for details.

## Documentation

Access comprehensive documentation at <https://rdflib.readthedocs.io>.

## Installation

Install the stable release using *pip*:

```bash
pip install rdflib
```

Install optional dependencies using *pip* extras:

```bash
pip install rdflib[berkeleydb,networkx,html,lxml,orjson]
```

Alternatively, download from PyPI:  <https://pypi.python.org/pypi/rdflib>

### Installing from the Current `main` Branch (for developers)

```bash
pip install git+https://github.com/rdflib/rdflib@main
```

or

```bash
pip install -e git+https://github.com/rdflib/rdflib@main#egg=rdflib
```

or from your locally cloned repository

```bash
poetry install  # installs into a poetry-managed venv
```

or

```bash
pip install -e .
```

## Getting Started

RDFLib uses a Pythonic API.  The core data object is the `Graph`, a collection of RDF triples (Subject, Predicate, Object).

Example: Creating a graph, loading data, and printing results:

```python
from rdflib import Graph
g = Graph()
g.parse('http://dbpedia.org/resource/Semantic_Web')

for s, p, o in g:
    print(s, p, o)
```

Triples consist of URIs and Literals.  Common namespaces are available:

```python
from rdflib.namespace import DC, DCTERMS, DOAP, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD
```

Example:  Using namespaces:

```python
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, XSD

g = Graph()
semweb = URIRef('http://dbpedia.org/resource/Semantic_Web')
type = g.value(semweb, RDFS.label)
```

Example: Adding a triple:

```python
g.add((
    URIRef("http://example.com/person/nick"),
    FOAF.givenName,
    Literal("Nick", datatype=XSD.string)
))
```

Bind namespaces for shorter URIs in serialization:

```python
g.bind("foaf", FOAF)
g.bind("xsd", XSD)

print(g.serialize(format="turtle"))
```

New namespaces can also be defined.

```python
dbpedia = Namespace('http://dbpedia.org/ontology/')

abstracts = list(x for x in g.objects(semweb, dbpedia['abstract']) if x.language=='en')
```

See [./examples](./examples) for more.

## Features

*   Parsers and serializers for various RDF formats.
*   `Graph` interface with multiple `Store` implementations.
*   In-memory and persistent storage options.
*   SPARQL 1.1 implementation for queries and updates.

## Running Tests

### Run tests on host

```bash
poetry install
poetry run pytest
```

### Test Coverage

```bash
poetry run pytest --cov
```

### View test coverage

```bash
poetry run pytest --cov --cov-report term --cov-report html
python -m http.server --directory=htmlcov
```

## Contributing

Contributions are welcome! Read the [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and [developers guide](https://rdflib.readthedocs.io/en/latest/developers/).

Submit Pull Requests:  <https://github.com/RDFLib/rdflib/pulls>

Use Gitpod or Google Cloud Shell for development.

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/RDFLib/rdflib)
[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FRDFLib%2Frdflib&cloudshell_git_branch=main&cloudshell_open_in_editor=README.md)

Raise issues: <https://github.com/RDFLib/rdflib/issues>

## Support & Contacts

For "how do I..." questions, use Stack Overflow with the tag `rdflib`:  <https://stackoverflow.com/questions/tagged/rdflib>

Contact maintainers via:

*   rdflib-dev mailing list: <https://groups.google.com/group/rdflib-dev>
*   Gitter or Matrix: [gitter](https://gitter.im/RDFLib/rdflib) or [#RDFLib_rdflib:gitter.im](https://matrix.to/#/#RDFLib_rdflib:gitter.im)