# Wrapt: Python Decorator Library for Robust Wrapping and Monkey Patching

**Wrapt** empowers Python developers with a powerful and reliable solution for creating decorators, wrappers, and performing safe monkey patching.

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features of Wrapt:

*   **Universal Decorators:** Works seamlessly with functions, methods, classmethods, staticmethods, and classes.
*   **Transparent Object Proxies:** Enables advanced wrapping scenarios and flexible code modification.
*   **Monkey Patching Utilities:** Provides tools for safe and controlled runtime modifications.
*   **Optimized Performance:** Includes a C extension for speed, with a Python fallback for broad compatibility.
*   **Introspection Preservation:** Maintains function signatures, annotations, and other critical metadata.
*   **Thread-Safe Implementations:** Ensures decorators are safe to use in multi-threaded environments.

## Why Use Wrapt?

Wrapt goes beyond standard decorator implementations by providing enhanced correctness, consistency, and compatibility. It meticulously preserves crucial aspects of wrapped objects, ensuring your decorators behave predictably across a wider range of Python code.  This makes it ideal for building robust, maintainable, and testable Python applications.

## Installation

Get started with Wrapt quickly:

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

## Documentation and Resources

*   **Comprehensive Documentation:** Explore detailed examples and advanced usage at [wrapt.readthedocs.io](https://wrapt.readthedocs.io/).
*   **GitHub Repository:** Find the source code, contribute, and report issues at [https://github.com/GrahamDumpleton/wrapt](https://github.com/GrahamDumpleton/wrapt)

## Supported Python Versions

Wrapt is compatible with:

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Wrapt welcomes contributions!  Please use the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to discuss changes, report bugs, or ask questions.

## License

Wrapt is released under the BSD License. See the [LICENSE](LICENSE) file for details.

## Additional Resources

*   **PyPI:**  [https://pypi.python.org/pypi/wrapt](https://pypi.python.org/pypi/wrapt)
*   **Issues:** [https://github.com/GrahamDumpleton/wrapt/issues/](https://github.com/GrahamDumpleton/wrapt/issues/)
*   **Changelog:** [https://wrapt.readthedocs.io/en/latest/changes.html](https://wrapt.readthedocs.io/en/latest/changes.html)

## Related Blog Posts

This repository also contains a series of blog posts explaining the design and implementation of wrapt:

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