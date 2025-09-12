# Wrapt: The Ultimate Python Module for Robust Decorators, Wrappers, and Monkey Patching

**Wrapt** empowers Python developers with a powerful and flexible module for creating decorators, wrappers, and monkey patching, designed for correctness and performance.

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features

*   **Universal Decorators:** Works seamlessly with functions, methods, classmethods, staticmethods, and classes.
*   **Transparent Object Proxies:** Enables advanced wrapping scenarios for maximum flexibility.
*   **Safe Monkey Patching Utilities:** Allows for safe runtime modifications to your code.
*   **Optimized Performance:** Includes a C extension for speed, with a pure Python fallback for compatibility.
*   **Introspection Preservation:** Preserves signatures, annotations, and other critical information for robust code.
*   **Thread-Safe Implementations:** Ensures your decorators function correctly in multithreaded environments.

## Why Use Wrapt?

Wrapt goes beyond the capabilities of `functools.wraps()` to provide decorators that are more reliable, predictable, and work consistently across a wider range of scenarios. It is built with correctness in mind, making it an excellent choice for building robust and maintainable Python applications.

## Installation

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

Explore comprehensive documentation, examples, and advanced usage patterns at:

**[wrapt.readthedocs.io](https://wrapt.readthedocs.io/)**

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

We welcome contributions! If you're interested in suggesting changes, improvements, or have found a bug, please reach out via the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/).

Please note that wrapt is a mature project. The primary focus is on ensuring that the package continues to work correctly with newer Python versions and maintaining compatibility as the Python ecosystem evolves.

### Testing

For information about running tests, including Python version-specific test conventions and available test commands, see [TESTING.md](TESTING.md).

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

## Links

*   **Documentation**: https://wrapt.readthedocs.io/
*   **PyPI**: https://pypi.python.org/pypi/wrapt
*   **Issues**: https://github.com/GrahamDumpleton/wrapt/issues/
*   **Changelog**: https://wrapt.readthedocs.io/en/latest/changes.html
*   **Source Code**: [https://github.com/GrahamDumpleton/wrapt](https://github.com/GrahamDumpleton/wrapt)

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