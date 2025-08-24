# Wrapt: Python Decorators, Wrappers, and Monkey Patching – Made Easy

**Wrapt** is a powerful Python module that simplifies the creation of decorators, wrappers, and supports monkey patching, ensuring your code is cleaner, more maintainable, and performs optimally. ([View on GitHub](https://github.com/GrahamDumpleton/wrapt))

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features of Wrapt

*   **Universal Decorators:** Works seamlessly with functions, methods (instance, class), classmethods, staticmethods, and even classes themselves.
*   **Transparent Object Proxies:** Provides advanced wrapping capabilities for complex scenarios.
*   **Safe Monkey Patching Utilities:**  Modify your code at runtime with confidence.
*   **Optimized Performance:** Leverages a C extension for performance-critical components, with a Python fallback for broad compatibility.
*   **Introspection Preservation:** Maintains function signatures, annotations, and other critical information for robust debugging and code understanding.
*   **Thread-Safe Decorator Implementations:** Ensures safe operation in multi-threaded environments.

## Why Use Wrapt?

Unlike other decorator implementations, Wrapt goes beyond `functools.wraps()` to guarantee that decorators preserve introspectability, signatures, and type checking. This leads to more predictable and consistent behavior, making it an ideal choice for production environments.

## Installation

Install Wrapt easily using pip:

```bash
pip install wrapt
```

## Examples

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

## Documentation & Resources

*   **Comprehensive Documentation:** Dive deeper with detailed explanations and examples at [wrapt.readthedocs.io](https://wrapt.readthedocs.io/)
*   **GitHub Repository:** Access the source code, report issues, and contribute at [https://github.com/GrahamDumpleton/wrapt](https://github.com/GrahamDumpleton/wrapt)
*   **PyPI:** Find the latest releases and download information at [https://pypi.python.org/pypi/wrapt](https://pypi.python.org/pypi/wrapt)
*   **Issue Tracking:**  Report bugs, suggest features, and get help via the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/)
*   **Changelog**: https://wrapt.readthedocs.io/en/latest/changes.html

## Supported Python Versions

Wrapt supports:

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

We welcome contributions!  Please use the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to discuss changes, report bugs, or ask questions.  See [TESTING.md](TESTING.md) for information about running tests.

## License

This project is licensed under the BSD License.  See the [LICENSE](LICENSE) file for details.

## Related Blog Posts

*   [How you implemented your Python decorator is wrong](blog/01-how-you-implemented-your-python-decorator-is-wrong.md)
*   [The interaction between decorators and descriptors](blog/02-the-interaction-between-decorators-and-descriptors.md)
*   [Implementing a factory for creating decorators](blog/03-implementing-a-factory-for-creating-decorators.md)
*   [Implementing a universal decorator](blog/04-implementing-a-universal-decorator.md)
*   [Decorators which accept arguments](blog/05-decorators-which-accept-arguments.md)
*   [Maintaining decorator state using a class](blog/06-maintaining-decorator-state-using-a-class.md)
*   [The missing synchronized decorator](blog/07-the-missing-synchronized-decorator.md)
*   [The synchronized decorator as context manager](blog/08-the-synchronized-decorator-as-context-manager.md)
*   [Performance overhead of using decorators](blog/09-performance-overhead-of-using-decorators.md)
*   [Performance overhead when applying decorators to methods](blog/10-performance-overhead-when-applying-decorators-to-methods.md)
*   [Safely applying monkey patches in Python](blog/11-safely-applying-monkey-patches-in-python.md)
*   [Using wrapt to support testing of software](blog/12-using-wrapt-to-support-testing-of-software.md)
*   [Ordering issues when monkey patching in Python](blog/13-ordering-issues-when-monkey-patching-in-python.md)
*   [Automatic patching of Python applications](blog/14-automatic-patching-of-python-applications.md)