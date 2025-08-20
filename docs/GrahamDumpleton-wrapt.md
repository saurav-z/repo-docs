# Wrapt: The Ultimate Python Module for Decorators, Wrappers, and Monkey Patching

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

**Wrapt is a robust Python library providing the tools you need to create decorators, wrappers, and safely perform monkey patching, all while maintaining full introspection and optimal performance.**

## Key Features

*   **Universal Decorators:** Apply decorators to functions, methods (including classmethods and staticmethods), and classes seamlessly.
*   **Transparent Object Proxies:** Build advanced wrapping solutions for complex scenarios.
*   **Safe Monkey Patching:** Modify Python code at runtime with confidence.
*   **Optimized Performance:**  Leverages a C extension for speed and includes a Python fallback for broad compatibility.
*   **Preserved Introspection:** Ensure decorators retain function signatures, annotations, and type checking capabilities.
*   **Thread-Safe Implementations:** Designed for safe use in multi-threaded applications.

## Why Use Wrapt?

Traditional Python decorators often fall short, particularly when dealing with complex scenarios or preserving the original function's attributes. Wrapt addresses these limitations by providing a comprehensive solution that focuses on correctness, consistency, and performance.  It ensures your decorators work reliably across a wider range of use cases, preserving key features like introspection and type hinting.

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

## Documentation and Examples

Explore comprehensive documentation, detailed examples, and advanced usage patterns at:

**[wrapt.readthedocs.io](https://wrapt.readthedocs.io/)**

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome! Please use the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to suggest changes, report bugs, or start discussions.

**Note:** Wrapt is a mature project. The focus is now on maintaining compatibility with newer Python versions.

## Testing

For details on running tests, see [TESTING.md](TESTING.md).

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

## Resources

*   **Documentation:** [wrapt.readthedocs.io](https://wrapt.readthedocs.io/)
*   **PyPI:** [https://pypi.python.org/pypi/wrapt](https://pypi.python.org/pypi/wrapt)
*   **GitHub Repository:** [https://github.com/GrahamDumpleton/wrapt](https://github.com/GrahamDumpleton/wrapt)
*   **Issues:** [https://github.com/GrahamDumpleton/wrapt/issues/](https://github.com/GrahamDumpleton/wrapt/issues/)
*   **Changelog:** [https://wrapt.readthedocs.io/en/latest/changes.html](https://wrapt.readthedocs.io/en/latest/changes.html)

## Related Blog Posts

A series of blog posts delve into the design and implementation of Wrapt, offering a deeper understanding of its capabilities:

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