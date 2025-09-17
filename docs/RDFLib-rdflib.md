<div align="center">
<img src="docs/_static/RDFlib.png" alt="RDFLib Logo" width="200"/>
</div>

# RDFLib: A Powerful Python Library for Working with RDF

**RDFLib is your go-to Python library for creating, parsing, querying, and manipulating RDF (Resource Description Framework) data.**  Access the original repo: [https://github.com/RDFLib/rdflib](https://github.com/RDFLib/rdflib)

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

*   **Versatile Format Support:** Parses and serializes RDF data in various formats, including RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, Trig, JSON-LD, and HexTuples.
*   **Graph Interface:** Offers a flexible Graph interface that can be backed by various store implementations for in-memory, persistent storage (Berkeley DB), and remote SPARQL endpoints.
*   **SPARQL 1.1 Implementation:** Provides a complete SPARQL 1.1 implementation, including support for queries and update statements, along with extension mechanisms for SPARQL functions.
*   **Extensible Architecture:**  Supports plugin-based store implementations, allowing you to extend RDFLib's capabilities.
*   **Pythonic API:** Designed with a user-friendly, Python-idiomatic API to simplify RDF manipulation.

## RDFLib Family of Packages

The RDFLib community maintains a suite of related Python packages:

*   [rdflib](https://github.com/RDFLib/rdflib) - The core RDFLib library.
*   [sparqlwrapper](https://github.com/RDFLib/sparqlwrapper) - A SPARQL service wrapper.
*   [pyLODE](https://github.com/RDFLib/pyLODE) - An OWL ontology documentation tool.
*   [pyrdfa3](https://github.com/RDFLib/pyrdfa3) - RDFa 1.1 distiller/parser library.
*   [pymicrodata](https://github.com/RDFLib/pymicrodata) - Extract RDF from HTML5 microdata.
*   [pySHACL](https://github.com/RDFLib/pySHACL) - Validate RDF graphs against SHACL graphs.
*   [OWL-RL](https://github.com/RDFLib/OWL-RL) - OWL2 RL Profile implementation.

Explore the complete list of packages at: <https://github.com/RDFLib>

## Versions & Releases

*   `main` branch - Current unstable release (version 8 alpha).
*   `7.1.4` - Tidy-up release.
*   `7.1.3` - Current stable release.
*   `7.1.1` - Previous stable release (see <https://github.com/RDFLib/rdflib/releases/tag/7.1.1>).
*   `7.0.0` - Previous stable release, supports Python 3.8.1+ only.
    (see [Releases](https://github.com/RDFLib/rdflib/releases))
*   `6.x.y` - Supports Python 3.7+ only (see <https://github.com/RDFLib/rdflib/releases/tag/6.3.2>).
*   `5.x.y` - Supports Python 2.7 and 3.4+ and is [mostly backwards compatible with 4.2.2](https://rdflib.readthedocs.io/en/stable/upgrade4to5.html).
  (see <https://github.com/RDFLib/rdflib/releases/tag/5.0.0>)

More details on releases are available at: <https://github.com/RDFLib/rdflib/releases/>

## Documentation

Comprehensive documentation is available at: <https://rdflib.readthedocs.io>

## Installation

Install the stable release of RDFLib using *pip*:

```bash
pip install rdflib
```

Install optional dependencies:

```bash
pip install rdflib[berkeleydb,networkx,html,lxml,orjson]
```

Alternatively, download from PyPI: [https://pypi.python.org/pypi/rdflib](https://pypi.python.org/pypi/rdflib)

### Installation of the current main branch (for developers)

Install from the Git repository using *pip*:

```bash
pip install git+https://github.com/rdflib/rdflib@main
```

or

```bash
pip install -e git+https://github.com/rdflib/rdflib@main#egg=rdflib
```

or from a local repository install using:

```bash
poetry install  # installs into a poetry-managed venv
```

or

```bash
pip install -e .
```

## Getting Started

RDFLib's core object is the `Graph`, a collection of RDF *Subject, Predicate, Object* Triples.

Example:

```python
from rdflib import Graph
g = Graph()
g.parse('http://dbpedia.org/resource/Semantic_Web')

for s, p, o in g:
    print(s, p, o)
```

URIs can be grouped by *namespace*:

```python
from rdflib.namespace import DC, DCTERMS, DOAP, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD
```

Use them like this:

```python
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, XSD

g = Graph()
semweb = URIRef('http://dbpedia.org/resource/Semantic_Web')
type = g.value(semweb, RDFS.label)
```

Adding a triple:

```python
g.add((
    URIRef("http://example.com/person/nick"),
    FOAF.givenName,
    Literal("Nick", datatype=XSD.string)
))
```

Bind namespaces for serialization:

```python
g.bind("foaf", FOAF)
g.bind("xsd", XSD)
```

```python
print(g.serialize(format="turtle"))
```

Result:

```turtle
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

<http://example.com/person/nick> foaf:givenName "Nick"^^xsd:string .
```

Define new Namespaces:

```python
dbpedia = Namespace('http://dbpedia.org/ontology/')

abstracts = list(x for x in g.objects(semweb, dbpedia['abstract']) if x.language=='en')
```

See also [./examples](./examples)

## Features Summary

*   Parsers and serializers for RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, JSON-LD, RDFa and Microdata.
*   Graph interface with various store implementations.
*   SPARQL 1.1 support (Queries and Updates).
*   Open Source.

## Running tests

### Running the tests on the host

Run the test suite with `pytest`.
```shell
poetry install
poetry run pytest
```

### Running test coverage on the host with coverage report

Run the test suite and generate a HTML coverage report with `pytest` and `pytest-cov`.
```shell
poetry run pytest --cov
```

### Viewing test coverage

Once tests have produced HTML output of the coverage report, view it by running:
```shell
poetry run pytest --cov --cov-report term --cov-report html
python -m http.server --directory=htmlcov
```

## Contributing

We welcome your contributions! See our [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and [developers guide](https://rdflib.readthedocs.io/en/latest/developers/) to get started.

Submit Pull Requests:

* <https://github.com/RDFLib/rdflib/pulls>

Use Gitpod or Google Cloud Shell for development:

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/RDFLib/rdflib)
[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FRDFLib%2Frdflib&cloudshell_git_branch=main&cloudshell_open_in_editor=README.md)

Report issues:

* <https://github.com/RDFLib/rdflib/issues>

## Support & Contacts

For "how do I..." questions, please use Stack Overflow with the tag `rdflib`:

*   <https://stackoverflow.com/questions/tagged/rdflib>

Contact maintainers:

*   rdflib-dev mailing list: <https://groups.google.com/group/rdflib-dev>
*   Chat (Gitter): [https://gitter.im/RDFLib/rdflib](https://gitter.im/RDFLib/rdflib) or via matrix [#RDFLib_rdflib:gitter.im](https://matrix.to/#/#RDFLib_rdflib:gitter.im)