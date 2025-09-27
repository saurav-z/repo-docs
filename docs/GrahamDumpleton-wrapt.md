# Wrapt: Powerful Python Decorators, Wrappers, and Monkey Patching

**Wrapt** is the ultimate Python module for crafting robust and reliable decorators, function wrappers, and safe monkey patching, enabling cleaner and more maintainable code.

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features of Wrapt

*   **Universal Decorators:** Works seamlessly with functions, methods (instance, class, static), and classes.
*   **Transparent Object Proxies:** Provides advanced wrapping capabilities for complex scenarios.
*   **Safe Monkey Patching:** Offers utilities for making runtime modifications to code safely.
*   **Optimized Performance:** Includes a C extension for performance-critical components with a Python fallback.
*   **Introspection Preservation:** Preserves function signatures, annotations, and other important introspection data.
*   **Thread-Safe Implementations:** Ensures decorators are safe to use in multithreaded environments.

## Why Use Wrapt?

Wrapt goes beyond the standard `functools.wraps()` to provide decorators that are more robust, work in a wider range of situations, and offer predictable behavior. Build decorators that preserve introspectability, signatures, and type checking abilities.

## Installation

Easily install Wrapt using pip:

```bash
pip install wrapt
```

## Quick Start Examples

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

*   **Comprehensive Documentation:** Explore detailed examples and usage patterns at [wrapt.readthedocs.io](https://wrapt.readthedocs.io/).
*   **Source Code:** [View the original repository on GitHub](https://github.com/GrahamDumpleton/wrapt).
*   **PyPI:** Find the package on [PyPI](https://pypi.python.org/pypi/wrapt)
*   **Issues:** Report issues or suggest improvements on [GitHub](https://github.com/GrahamDumpleton/wrapt/issues/).
*   **Changelog:** See the changes in each release on [wrapt.readthedocs.io/en/latest/changes.html](https://wrapt.readthedocs.io/en/latest/changes.html).

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome!  Please use the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to start a discussion about any changes, improvements, or bug fixes.

**Note:** Wrapt is considered a mature project. The focus is on maintaining compatibility with newer Python versions.

### Testing

For testing information, including Python version-specific test conventions and available test commands, see [TESTING.md](TESTING.md).

## License

Wrapt is licensed under the BSD License. See the [LICENSE](LICENSE) file for more details.