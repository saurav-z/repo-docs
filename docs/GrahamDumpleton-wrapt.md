# Wrapt: Python Decorators, Wrappers, and Monkey Patching

**Wrapt is a powerful Python module that simplifies the creation of decorators, wrappers, and safe monkey patching for robust and introspectable code.**

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features of Wrapt

*   **Universal Decorators:** Works seamlessly with functions, methods (including instance, class, and static), and classes.
*   **Transparent Object Proxies:** Enables advanced wrapping techniques.
*   **Safe Monkey Patching Utilities:** Provides tools for making runtime modifications with confidence.
*   **Optimized Performance:** Includes a C extension for speed, with a Python fallback for broad compatibility.
*   **Comprehensive Introspection Preservation:** Preserves signatures, annotations, and other crucial metadata.
*   **Thread-Safe Implementations:** Built with thread safety in mind for concurrent environments.

## Why Use Wrapt?

Wrapt addresses limitations in standard decorator implementations, ensuring decorators work correctly across various scenarios, preserving essential information, and providing consistent behavior. Whether you're working with functions, methods, or classes, Wrapt simplifies and improves decorator creation, offering a robust and reliable solution for Python developers.

## Installation

Install Wrapt using pip:

```bash
pip install wrapt
```

## Quick Start

### Basic Decorator

```python
import wrapt

@wrapt.decorator
def pass_through(wrapped, instance, args, kwargs):
    return wrapped(*args, **kwargs)

@pass_through
def function():
    pass
```

### Decorator with Arguments

```python
import wrapt

def with_arguments(myarg1, myarg2):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        print(f"Arguments: {myarg1}, {myarg2}")
        return wrapped(*args, **kwargs)
    return wrapper

@with_arguments(1, 2)
def function():
    pass
```

### Universal Decorator

```python
import inspect
import wrapt

@wrapt.decorator
def universal(wrapped, instance, args, kwargs):
    if instance is None:
        if inspect.isclass(wrapped):
            # Decorator was applied to a class
            print("Decorating a class")
        else:
            # Decorator was applied to a function or staticmethod
            print("Decorating a function")
    else:
        if inspect.isclass(instance):
            # Decorator was applied to a classmethod
            print("Decorating a classmethod")
        else:
            # Decorator was applied to an instancemethod
            print("Decorating an instance method")
    
    return wrapped(*args, **kwargs)
```

## Documentation

Detailed documentation, examples, and advanced usage patterns are available at:

**[wrapt.readthedocs.io](https://wrapt.readthedocs.io/)**

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome! Please use the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to discuss potential changes, report bugs, or ask questions. Note that the project is mature, and focus is on maintaining compatibility.

See [TESTING.md](TESTING.md) for information about running tests.

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

## Links

*   **Documentation**: https://wrapt.readthedocs.io/
*   **PyPI**: https://pypi.python.org/pypi/wrapt
*   **Issues**: https://github.com/GrahamDumpleton/wrapt/issues/
*   **Changelog**: https://wrapt.readthedocs.io/en/latest/changes.html
*   **Source Code**: [GitHub Repository](https://github.com/GrahamDumpleton/wrapt)

## Related Blog Posts

This repository also contains a series of blog posts explaining the design and implementation of wrapt:

-   [How you implemented your Python decorator is wrong](blog/01-how-you-implemented-your-python-decorator-is-wrong.md)
-   [The interaction between decorators and descriptors](blog/02-the-interaction-between-decorators-and-descriptors.md)
-   [Implementing a factory for creating decorators](blog/03-implementing-a-factory-for-creating-decorators.md)
-   [Implementing a universal decorator](blog/04-implementing-a-universal-decorator.md)
-   [Decorators which accept arguments](blog/05-decorators-which-accept-arguments.md)
-   [Maintaining decorator state using a class](blog/06-maintaining-decorator-state-using-a-class.md)
-   [The missing synchronized decorator](blog/07-the-missing-synchronized-decorator.md)
-   [The synchronized decorator as context manager](blog/08-the-synchronized-decorator-as-context-manager.md)
-   [Performance overhead of using decorators](blog/09-performance-overhead-of-using-decorators.md)
-   [Performance overhead when applying decorators to methods](blog/10-performance-overhead-when-applying-decorators-to-methods.md)
-   [Safely applying monkey patches in Python](blog/11-safely-applying-monkey-patches-in-python.md)
-   [Using wrapt to support testing of software](blog/12-using-wrapt-to-support-testing-of-software.md)
-   [Ordering issues when monkey patching in Python](blog/13-ordering-issues-when-monkey-patching-in-python.md)
-   [Automatic patching of Python applications](blog/14-automatic-patching-of-python-applications.md)