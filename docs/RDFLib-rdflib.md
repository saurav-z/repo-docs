# RDFLib: Your Python Toolkit for Working with RDF Data

**RDFLib is a powerful and versatile Python library designed to work with Resource Description Framework (RDF) data, enabling you to parse, serialize, query, and manipulate semantic web information.**

[Explore the RDFLib Repository](https://github.com/RDFLib/rdflib)

**Key Features:**

*   **Comprehensive Parsing & Serialization:** Supports a wide range of RDF formats, including RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, Trig, JSON-LD, RDFa, and Microdata.
*   **Flexible Graph Interface:** Provides a `Graph` interface that can be backed by various store implementations for in-memory, persistent (Berkeley DB), and remote SPARQL endpoint storage.
*   **SPARQL 1.1 Implementation:** Includes a robust SPARQL 1.1 implementation for querying and updating RDF data.
*   **Extensible:** Offers mechanisms for extending SPARQL functionality.
*   **Pythonic API:** Designed with a user-friendly Python API for easy integration and use.
*   **Namespace Support:** Includes common namespaces (DC, DCTERMS, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD) to streamline RDF data handling.
*   **Extensive Documentation:** Comprehensive documentation available at [https://rdflib.readthedocs.io](https://rdflib.readthedocs.io).

**Use Cases:**

*   **Semantic Web Development:** Build applications that utilize semantic web technologies.
*   **Knowledge Graph Management:** Create, manage, and query knowledge graphs.
*   **Data Integration:** Integrate data from various sources using RDF as a common data model.
*   **Linked Data Applications:** Build applications that work with linked data.

**Installation:**

Install the stable release using pip:

```bash
pip install rdflib
```

Install optional dependencies with extras:

```bash
pip install rdflib[berkeleydb,networkx,html,lxml,orjson]
```

For developers, install from the git repository:

```bash
pip install git+https://github.com/rdflib/rdflib@main
```

**Getting Started:**

RDFLib's core data object is a `Graph`, representing a collection of RDF triples (Subject, Predicate, Object).

```python
from rdflib import Graph
g = Graph()
g.parse('http://dbpedia.org/resource/Semantic_Web')

for s, p, o in g:
    print(s, p, o)
```

**Example:**

```python
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, XSD

g = Graph()
semweb = URIRef('http://dbpedia.org/resource/Semantic_Web')
type = g.value(semweb, RDFS.label)
```

**Contributing:**

RDFLib welcomes contributions!  Read our [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and [developers guide](https://rdflib.readthedocs.io/en/latest/developers/) and submit pull requests:

*   [Pull Requests](https://github.com/RDFLib/rdflib/pulls)
*   [Issues](https://github.com/RDFLib/rdflib/issues)

**Support:**

*   **Stack Overflow:**  Ask questions and find answers using the `rdflib` tag: <https://stackoverflow.com/questions/tagged/rdflib>
*   **Mailing List:** rdflib-dev: <https://groups.google.com/group/rdflib-dev>
*   **Chat:** Gitter: [https://gitter.im/RDFLib/rdflib](https://gitter.im/RDFLib/rdflib) or via Matrix: [#RDFLib_rdflib:gitter.im](https://matrix.to/#/#RDFLib_rdflib:gitter.im)