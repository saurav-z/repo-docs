# Wrapt: Powerful Python Decorators, Wrappers, and Monkey Patching

**Wrapt empowers Python developers with robust tools for creating decorators, wrappers, and implementing safe monkey patching, enhancing code reusability, and maintainability.**

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features of Wrapt

*   **Universal Decorators:** Works seamlessly with functions, methods (instance, class, and static), and classes.
*   **Transparent Object Proxies:** Provides advanced wrapping capabilities for complex scenarios.
*   **Safe Monkey Patching:** Enables secure runtime modifications to existing code.
*   **Optimized Performance:** Utilizes a C extension for speed, with a pure Python fallback.
*   **Introspection Preservation:** Maintains function signatures, annotations, and other critical metadata.
*   **Thread-Safe Implementations:** Ensures decorators function correctly in concurrent environments.

## What is Wrapt?

Wrapt is a Python module designed to provide a solid foundation for creating function wrappers and decorators. It addresses the limitations of standard decorator implementations to ensure correct behavior in a wide range of use cases. Unlike other solutions, Wrapt prioritizes introspection and preserves essential information like signatures, which is crucial for debugging and type checking.

## Installation

Install Wrapt using pip:

```bash
pip install wrapt
```

## Usage Examples

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

*   **Comprehensive Documentation:** Explore detailed explanations, examples, and advanced usage at [wrapt.readthedocs.io](https://wrapt.readthedocs.io/).
*   **Original Repository:** Access the source code, issues, and contribute on GitHub: [https://github.com/GrahamDumpleton/wrapt](https://github.com/GrahamDumpleton/wrapt)
*   **PyPI:** Find the package on PyPI: [https://pypi.python.org/pypi/wrapt](https://pypi.python.org/pypi/wrapt)
*   **Issues:** Report bugs, request features, or ask questions: [https://github.com/GrahamDumpleton/wrapt/issues/](https://github.com/GrahamDumpleton/wrapt/issues/)
*   **Changelog:** Review the change history: [https://wrapt.readthedocs.io/en/latest/changes.html](https://wrapt.readthedocs.io/en/latest/changes.html)

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome! If you have suggestions, bug reports, or questions, please open an issue on the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/).

**Note:** Wrapt is a mature project.  The focus is on maintaining compatibility and ensuring it functions well with current and future Python versions.

## License

This project is licensed under the BSD License. See the [LICENSE](LICENSE) file for details.

## Related Blog Posts

Explore a series of blog posts that dive deep into the design and implementation of wrapt:

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