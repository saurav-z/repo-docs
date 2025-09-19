# RDFLib: Your Python Toolkit for Working with RDF Data

**RDFLib is a powerful and versatile Python library that simplifies working with Resource Description Framework (RDF) data, enabling you to parse, serialize, query, and manipulate semantic web information.**  [Explore the project on GitHub](https://github.com/RDFLib/rdflib).

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

*   **Comprehensive RDF Support:** Parsers and serializers for various RDF formats, including RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, Trig, JSON-LD, and HexTuples.
*   **Flexible Graph Interface:** A `Graph` interface that can be backed by a variety of store implementations, including in-memory, persistent (Berkeley DB), and remote SPARQL endpoints.  Additional stores can be integrated via plugins.
*   **SPARQL 1.1 Implementation:** Full support for SPARQL 1.1 Queries and Update statements.
*   **SPARQL Function Extensions:** Mechanism to extend SPARQL functionality.

## RDFLib Family of Packages

The RDFLib community has a wider set of packages:

*   [rdflib](https://github.com/RDFLib/rdflib) - the RDFLib core
*   [sparqlwrapper](https://github.com/RDFLib/sparqlwrapper) - a simple Python wrapper around a SPARQL service to remotely execute your queries
*   [pyLODE](https://github.com/RDFLib/pyLODE) - An OWL ontology documentation tool using Python and templating, based on LODE.
*   [pyrdfa3](https://github.com/RDFLib/pyrdfa3) - RDFa 1.1 distiller/parser library: can extract RDFa 1.1/1.0 from (X)HTML, SVG, or XML in general.
*   [pymicrodata](https://github.com/RDFLib/pymicrodata) - A module to extract RDF from an HTML5 page annotated with microdata.
*   [pySHACL](https://github.com/RDFLib/pySHACL) - A pure Python module which allows for the validation of RDF graphs against SHACL graphs.
*   [OWL-RL](https://github.com/RDFLib/OWL-RL) - A simple implementation of the OWL2 RL Profile which expands the graph with all possible triples that OWL RL defines.

Full list: <https://github.com/RDFLib>

## Versions & Releases

*   `main` branch in this repository is the current unstable release - version 8 alpha
*   `7.1.4` tidy-up release, possibly last 7.x release
*   `7.1.3` current stable release, small improvements to 7.1.1
*   `7.1.2` previously deleted release
*   `7.1.1` previous stable release
    * see <https://github.com/RDFLib/rdflib/releases/tag/7.1.1>
*   `7.0.0` previous stable release, supports Python 3.8.1+ only.
    * see [Releases](https://github.com/RDFLib/rdflib/releases)
*   `6.x.y` supports Python 3.7+ only. Many improvements over 5.0.0
    * see <https://github.com/RDFLib/rdflib/releases/tag/6.3.2>
*   `5.x.y` supports Python 2.7 and 3.4+ and is [mostly backwards compatible with 4.2.2](https://rdflib.readthedocs.io/en/stable/upgrade4to5.html).
  * * see <https://github.com/RDFLib/rdflib/releases/tag/5.0.0>

See <https://github.com/RDFLib/rdflib/releases/> for the release details.

## Documentation

Comprehensive documentation is available at: <https://rdflib.readthedocs.io>

## Installation

Install the latest stable release using *pip*:

```bash
pip install rdflib
```

Install optional dependencies using *pip* extras:

```bash
pip install rdflib[berkeleydb,networkx,html,lxml,orjson]
```

Alternatively, download from PyPI:  https://pypi.python.org/pypi/rdflib

### Installing the Current Main Branch (for Developers)

Install from the git repository using *pip*:

```bash
pip install git+https://github.com/rdflib/rdflib@main
```
or
```bash
pip install -e git+https://github.com/rdflib/rdflib@main#egg=rdflib
```

Or install from a local clone:

```bash
poetry install  # installs into a poetry-managed venv
```

or

```bash
pip install -e .
```

## Getting Started

RDFLib provides a Pythonic API for interacting with RDF data, centered around the `Graph` object, a collection of *Subject, Predicate, Object* triples.

Example: Create a graph, load from DBPedia, and print triples:

```python
from rdflib import Graph
g = Graph()
g.parse('http://dbpedia.org/resource/Semantic_Web')

for s, p, o in g:
    print(s, p, o)
```

Triples consist of URIs (resources) and Literals (values). Namespaces are included:

```python
from rdflib.namespace import DC, DCTERMS, DOAP, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD
```

Example: Using namespaces:

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

Bind namespaces for concise serialization:

```python
g.bind("foaf", FOAF)
g.bind("xsd", XSD)
```

Then serialize:

```python
print(g.serialize(format="turtle"))
```

New namespaces can also be defined.

See also [./examples](./examples)

## Features

*   Parsers and serializers for multiple RDF formats (RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, JSON-LD, RDFa, and Microdata).
*   Graph interface with various store implementations.
*   In-memory and Berkeley DB persistent storage options.
*   SPARQL 1.1 support.

## Running Tests

### Running Tests on the Host

```shell
poetry install
poetry run pytest
```

### Running Test Coverage on the Host

```shell
poetry run pytest --cov
```

### Viewing Test Coverage

```shell
poetry run pytest --cov --cov-report term --cov-report html
python -m http.server --directory=htmlcov
```

## Contributing

Contributions are welcomed!  Please read the [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and [developers guide](https://rdflib.readthedocs.io/en/latest/developers/).

Submit pull requests here: <https://github.com/RDFLib/rdflib/pulls>

Use Gitpod or Google Cloud Shell for development.

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/RDFLib/rdflib)
[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FRDFLib%2Frdflib&cloudshell_git_branch=main&cloudshell_open_in_editor=README.md)

Report issues here: <https://github.com/RDFLib/rdflib/issues>

## Support & Contacts

For "how do I..." questions, use Stack Overflow with the tag `rdflib`:  <https://stackoverflow.com/questions/tagged/rdflib>

Contact maintainers:

*   rdflib-dev mailing list: <https://groups.google.com/group/rdflib-dev>
*   Chat: [gitter](https://gitter.im/RDFLib/rdflib) or matrix [#RDFLib_rdflib:gitter.im](https://matrix.to/#/#RDFLib_rdflib:gitter.im)