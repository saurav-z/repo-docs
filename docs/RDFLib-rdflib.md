# RDFLib: A Powerful Python Library for Working with RDF Data

**RDFLib empowers developers to work with Resource Description Framework (RDF) data in Python, making it easy to parse, manipulate, and serialize semantic web data.**

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

RDFLib is a versatile Python library designed for working with RDF, a standard model for data interchange on the Semantic Web. It provides comprehensive tools for handling RDF data, including parsing, serialization, querying, and storage.

**Key Features:**

*   **Extensive Format Support:** Parsers and serializers for RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, Trig, JSON-LD, and HexTuples.
*   **Flexible Graph Interface:** A Graph interface that can be backed by various Store implementations.
*   **Diverse Storage Options:**  Store implementations for in-memory, persistent on-disk (Berkeley DB), and remote SPARQL endpoints. Supports plugin stores.
*   **SPARQL 1.1 Compliance:**  Includes a SPARQL 1.1 implementation, supporting both queries and update statements.
*   **SPARQL Function Extensions:** Supports SPARQL function extension mechanisms.
*   **Easy to Use:** Pythonic API for easy data manipulation and access.

## RDFLib Family of Packages

The RDFLib community maintains several related Python packages, including:

*   [rdflib](https://github.com/RDFLib/rdflib) - The core RDFLib package.
*   [sparqlwrapper](https://github.com/RDFLib/sparqlwrapper) - A simple Python wrapper for SPARQL services.
*   [pyLODE](https://github.com/RDFLib/pyLODE) - An OWL ontology documentation tool.
*   [pyrdfa3](https://github.com/RDFLib/pyrdfa3) - RDFa distiller/parser library.
*   [pymicrodata](https://github.com/RDFLib/pymicrodata) - Extracts RDF from HTML5 microdata.
*   [pySHACL](https://github.com/RDFLib/pySHACL) - Validates RDF graphs against SHACL graphs.
*   [OWL-RL](https://github.com/RDFLib/OWL-RL) - Simple implementation of the OWL2 RL Profile.

Explore the full list of packages at [https://github.com/RDFLib](https://github.com/RDFLib).

## Versions & Releases

* `main` branch in this repository is the current unstable release - version 8 alpha
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

For detailed release information, see the [Releases](https://github.com/RDFLib/rdflib/releases/) page.

## Documentation

Comprehensive documentation is available at [https://rdflib.readthedocs.io](https://rdflib.readthedocs.io), including latest, stable, and versioned builds.

## Installation

Install the stable release using `pip`:

```bash
pip install rdflib
```

Install optional dependencies (e.g., Berkeley DB, networkx, etc.) using extras:

```bash
pip install rdflib[berkeleydb,networkx,html,lxml,orjson]
```

Alternatively, download the package from PyPI at [https://pypi.python.org/pypi/rdflib](https://pypi.python.org/pypi/rdflib).

### Installation of the current main branch (for developers)

Install rdflib directly from the Git repository:

```bash
pip install git+https://github.com/rdflib/rdflib@main
```

or

```bash
pip install -e git+https://github.com/rdflib/rdflib@main#egg=rdflib
```

or from your locally cloned repository you can install it with one of the following options:

```bash
poetry install  # installs into a poetry-managed venv
```

or

```bash
pip install -e .
```

## Getting Started

RDFLib's core data structure is the `Graph`, a Python collection of RDF *Subject, Predicate, Object* Triples.

Example:

```python
from rdflib import Graph
g = Graph()
g.parse('http://dbpedia.org/resource/Semantic_Web')

for s, p, o in g:
    print(s, p, o)
```

URIs are grouped by *namespace*, common namespaces are included in RDFLib:

```python
from rdflib.namespace import DC, DCTERMS, DOAP, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD
```

Example:

```python
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, XSD

g = Graph()
semweb = URIRef('http://dbpedia.org/resource/Semantic_Web')
type = g.value(semweb, RDFS.label)
```

Adding triples to a graph `g`:

```python
g.add((
    URIRef("http://example.com/person/nick"),
    FOAF.givenName,
    Literal("Nick", datatype=XSD.string)
))
```

Bind namespaces to prefixes to shorten URIs:

```python
g.bind("foaf", FOAF)
g.bind("xsd", XSD)
```

Serializing the graph in Turtle format:

```python
print(g.serialize(format="turtle"))
```

New Namespaces can also be defined:

```python
dbpedia = Namespace('http://dbpedia.org/ontology/')

abstracts = list(x for x in g.objects(semweb, dbpedia['abstract']) if x.language=='en')
```

See also [./examples](./examples)

## Features

*   Parsers and serializers for RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, JSON-LD, RDFa and Microdata.
*   Graph interface with various Store implementations.
*   Store implementations for in-memory and persistent Berkeley DB storage.
*   SPARQL 1.1 implementation for queries and updates.

## Running Tests

Test with `pytest`:

```bash
poetry install
poetry run pytest
```

Generate HTML coverage reports:

```bash
poetry run pytest --cov --cov-report term --cov-report html
python -m http.server --directory=htmlcov
```

## Contributing

Contributions are welcome!  Read the [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and [developers guide](https://rdflib.readthedocs.io/en/latest/developers/) to get started.

Submit pull requests at: [https://github.com/RDFLib/rdflib/pulls](https://github.com/RDFLib/rdflib/pulls).

Use Gitpod or Google Cloud Shell for development.

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/RDFLib/rdflib)
[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FRDFLib%2Frdflib&cloudshell_git_branch=main&cloudshell_open_in_editor=README.md)

Report issues at: [https://github.com/RDFLib/rdflib/issues](https://github.com/RDFLib/rdflib/issues).

## Support & Contacts

For general questions, use Stack Overflow with the `rdflib` tag: [https://stackoverflow.com/questions/tagged/rdflib](https://stackoverflow.com/questions/tagged/rdflib).

Contact the maintainers via:

*   rdflib-dev mailing list: <https://groups.google.com/group/rdflib-dev>
*   Chat: [gitter](https://gitter.im/RDFLib/rdflib) or matrix [#RDFLib_rdflib:gitter.im](https://matrix.to/#/#RDFLib_rdflib:gitter.im)

[Back to Top](#rdflib-a-powerful-python-library-for-working-with-rdf-data)