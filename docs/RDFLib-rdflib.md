# RDFLib: Your Go-To Python Library for Working with RDF

**RDFLib** is a powerful and versatile Python library that simplifies the process of working with Resource Description Framework (RDF) data, enabling developers to easily parse, manipulate, and serialize RDF graphs. Access the original repository [here](https://github.com/RDFLib/rdflib).

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

*   **Comprehensive RDF Support:** Parsers and serializers for popular RDF formats: RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, Trig, JSON-LD, and more.
*   **Flexible Graph Interface:** A core `Graph` interface that can be backed by various store implementations, including in-memory, persistent on-disk (Berkeley DB), and remote SPARQL endpoints.
*   **SPARQL 1.1 Implementation:** Includes support for SPARQL 1.1 queries and update statements, empowering advanced data retrieval and manipulation.
*   **Extensible Architecture:**  Supports plugin-based store implementations and SPARQL function extension mechanisms for customization.
*   **Pythonic API:** Designed to be intuitive and easy to use within the Python ecosystem.

## RDFLib Family of Packages

The RDFLib community maintains multiple related Python packages, including:

*   [rdflib](https://github.com/RDFLib/rdflib) - The core RDFLib library.
*   [sparqlwrapper](https://github.com/RDFLib/sparqlwrapper) - A SPARQL service wrapper.
*   [pyLODE](https://github.com/RDFLib/pyLODE) - OWL ontology documentation tool.
*   [pyrdfa3](https://github.com/RDFLib/pyrdfa3) - RDFa distiller/parser.
*   [pymicrodata](https://github.com/RDFLib/pymicrodata) - Extracts RDF from HTML5 Microdata.
*   [pySHACL](https://github.com/RDFLib/pySHACL) - SHACL validation for RDF graphs.
*   [OWL-RL](https://github.com/RDFLib/OWL-RL) - OWL2 RL Profile implementation.

Explore the full list of packages at: <https://github.com/RDFLib>

## Versions & Releases

*   `main` branch: Current unstable release (version 8 alpha)
*   `7.2.1`: Tiny cleanup release, relaxes Python version requirement
*   `7.2.0`: General fixes and usability improvements.
*   `7.1.4`: Tidy-up release.
*   `7.1.3`: Current stable release.
*   `7.1.1`: Previous stable release.
*   `7.0.0`: Previous stable release, supports Python 3.8.1+ only.
*   `6.x.y`: Supports Python 3.7+ only.
*   `5.x.y`: Supports Python 2.7 and 3.4+ (mostly backwards compatible with 4.2.2).

See <https://github.com/RDFLib/rdflib/releases/> for detailed release notes.

## Documentation

Comprehensive documentation is available at: <https://rdflib.readthedocs.io>

## Installation

Install the stable release of RDFLib using `pip`:

```bash
pip install rdflib
```

Install optional dependencies with extras:

```bash
pip install rdflib[berkeleydb,networkx,html,lxml,orjson]
```

Alternatively, download from PyPI:  <https://pypi.python.org/pypi/rdflib>

### Installation of the current main branch (for developers)
```bash
pip install git+https://github.com/rdflib/rdflib@main
```
or
```bash
pip install -e git+https://github.com/rdflib/rdflib@main#egg=rdflib
```
or from your locally cloned repository:
```bash
poetry install  # installs into a poetry-managed venv
```
or
```bash
pip install -e .
```

## Getting Started

RDFLib provides a Pythonic API for working with RDF data using the `Graph` object:

```python
from rdflib import Graph
g = Graph()
g.parse('http://dbpedia.org/resource/Semantic_Web')

for s, p, o in g:
    print(s, p, o)
```

## Namespaces

RDFLib includes common namespaces:

```python
from rdflib.namespace import DC, DCTERMS, DOAP, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD
```

Example usage:

```python
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, XSD

g = Graph()
semweb = URIRef('http://dbpedia.org/resource/Semantic_Web')
type = g.value(semweb, RDFS.label)
```

Adding triples:

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

```turtle
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

<http://example.com/person/nick> foaf:givenName "Nick"^^xsd:string .
```

## Running Tests

Run the test suite:
```bash
poetry install
poetry run pytest
```

Generate a coverage report:
```bash
poetry run pytest --cov
```

View test coverage:
```bash
poetry run pytest --cov --cov-report term --cov-report html
python -m http.server --directory=htmlcov
```

## Contributing

Contributions are welcome! Read the [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and [developers guide](https://rdflib.readthedocs.io/en/latest/developers/) and submit Pull Requests:

*   <https://github.com/RDFLib/rdflib/pulls>

Use Gitpod or Google Cloud Shell for development:

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/RDFLib/rdflib)
[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FRDFLib%2Frdflib&cloudshell_git_branch=main&cloudshell_open_in_editor=README.md)

Report issues:

*   <https://github.com/RDFLib/rdflib/issues>

## Support & Contacts

For "how do I..." questions, use Stack Overflow with the `rdflib` tag:

*   <https://stackoverflow.com/questions/tagged/rdflib>

Contact the maintainers:

*   rdflib-dev mailing list: <https://groups.google.com/group/rdflib-dev>
*   Chat: [gitter](https://gitter.im/RDFLib/rdflib) or via matrix [#RDFLib_rdflib:gitter.im](https://matrix.to/#/#RDFLib_rdflib:gitter.im)