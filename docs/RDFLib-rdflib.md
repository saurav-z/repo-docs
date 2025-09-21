# RDFLib: Your Python Toolkit for Semantic Web Data

**RDFLib is a powerful and versatile Python library for working with Resource Description Framework (RDF) data, enabling you to parse, serialize, and query semantic web information with ease.**  [Explore the original repository](https://github.com/RDFLib/rdflib).

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

*   **Comprehensive RDF Support:** Parsers and serializers for a wide range of RDF formats, including RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, Trig, JSON-LD, and HexTuples.
*   **Flexible Graph Interface:**  A Python `Graph` interface that can be backed by various store implementations for different storage needs.
*   **Diverse Store Implementations:** Includes in-memory stores, persistent disk-based stores (Berkeley DB), and support for remote SPARQL endpoints, with plugin support for custom stores.
*   **SPARQL 1.1 Compliance:**  Complete SPARQL 1.1 implementation, supporting both queries and update statements.
*   **SPARQL Extension Mechanisms:**  Provides mechanisms for extending SPARQL functionality.
*   **Extensive Documentation:** Comprehensive documentation is available to guide your work with rdflib.

## RDFlib Family of Packages

The RDFLib community provides numerous RDF-related Python projects. Here are a few examples:

*   [rdflib](https://github.com/RDFLib/rdflib) - The core RDFLib library.
*   [sparqlwrapper](https://github.com/RDFLib/sparqlwrapper) - A simplified Python wrapper for SPARQL services.
*   [pyLODE](https://github.com/RDFLib/pyLODE) - A documentation tool for OWL ontologies, based on LODE.
*   [pyrdfa3](https://github.com/RDFLib/pyrdfa3) - A library for extracting RDFa from HTML, SVG, and XML.
*   [pymicrodata](https://github.com/RDFLib/pymicrodata) - A module for extracting RDF from HTML5 Microdata.
*   [pySHACL](https://github.com/RDFLib/pySHACL) - A library for validating RDF graphs against SHACL graphs.
*   [OWL-RL](https://github.com/RDFLib/OWL-RL) - Implementation of the OWL2 RL Profile.

For a complete list of RDFLib family packages, visit: <https://github.com/RDFLib>

Your contributions to the RDFLib family are always appreciated!

## Versions & Releases

* `main` branch is the current unstable release - version 8 alpha
* `7.2.1` tiny clean up release, relaxes Python version requirement
* `7.2.0` general fixes and usability improvements, see changelog for details
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

Access detailed documentation at: <https://rdflib.readthedocs.io>

## Installation

Install the latest stable release using pip:

```bash
pip install rdflib
```

Install optional dependencies using pip extras:

```bash
pip install rdflib[berkeleydb,networkx,html,lxml,orjson]
```

Alternatively, download the package from PyPI at:  https://pypi.python.org/pypi/rdflib

### Installing the Current Main Branch (for Developers)

Install from the Git repository using pip:

```bash
pip install git+https://github.com/rdflib/rdflib@main
```

Or, install from your local clone:

```bash
poetry install  # installs into a poetry-managed venv
```

or

```bash
pip install -e .
```

## Getting Started

RDFLib provides a Pythonic API for working with RDF data, with the `Graph` object being the central data structure, representing a collection of *Subject, Predicate, Object* triples.

**Example: Creating a graph, loading data, and printing results:**

```python
from rdflib import Graph
g = Graph()
g.parse('http://dbpedia.org/resource/Semantic_Web')

for s, p, o in g:
    print(s, p, o)
```

**Namespaces:**

RDFLib includes common namespaces:

```python
from rdflib.namespace import DC, DCTERMS, DOAP, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD
```

**Example: Using Namespaces:**

```python
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, XSD

g = Graph()
semweb = URIRef('http://dbpedia.org/resource/Semantic_Web')
type = g.value(semweb, RDFS.label)
```

**Adding Triples:**

```python
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import FOAF, XSD

g = Graph()
g.add((
    URIRef("http://example.com/person/nick"),
    FOAF.givenName,
    Literal("Nick", datatype=XSD.string)
))
```

**Binding Namespaces:**

```python
from rdflib.namespace import FOAF, XSD

g.bind("foaf", FOAF)
g.bind("xsd", XSD)
print(g.serialize(format="turtle"))
```

**Defining New Namespaces:**

```python
from rdflib import Graph, URIRef, Literal, Namespace
dbpedia = Namespace('http://dbpedia.org/ontology/')

abstracts = list(x for x in g.objects(semweb, dbpedia['abstract']) if x.language=='en')
```

See more examples in the [`./examples`](./examples) directory.

## Features

*   Parsers and serializers for various RDF formats (RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, JSON-LD, RDFa, Microdata).
*   A `Graph` interface backed by various store implementations.
*   In-memory and Berkeley DB store implementations are included.
*   SPARQL 1.1 support (queries and updates).

## Running Tests

### Running tests on the host:

```bash
poetry install
poetry run pytest
```

### Test Coverage Report

```bash
poetry run pytest --cov
```

### Viewing Test Coverage

```bash
poetry run pytest --cov --cov-report term --cov-report html
python -m http.server --directory=htmlcov
```

## Contributing

Contribute to RDFLib by reading the [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and the [developers guide](https://rdflib.readthedocs.io/en/latest/developers/).

Submit Pull Requests here:

*   <https://github.com/RDFLib/rdflib/pulls>

You can use Gitpod or Google Cloud Shell for a development environment:

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/RDFLib/rdflib)
[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FRDFLib%2Frdflib&cloudshell_git_branch=main&cloudshell_open_in_editor=README.md)

Report issues here:

*   <https://github.com/RDFLib/rdflib/issues>

## Support & Contacts

For general questions, use [Stack Overflow](https://stackoverflow.com) with the `rdflib` tag: <https://stackoverflow.com/questions/tagged/rdflib>

Contact the maintainers via:

*   The rdflib-dev mailing list: <https://groups.google.com/group/rdflib-dev>
*   The chat on [Gitter](https://gitter.im/RDFLib/rdflib) or via matrix [#RDFLib_rdflib:gitter.im](https://matrix.to/#/#RDFLib_rdflib:gitter.im)