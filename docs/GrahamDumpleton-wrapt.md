# wrapt: The Python Module for Robust Decorators, Wrappers, and Monkey Patching

**[wrapt](https://github.com/GrahamDumpleton/wrapt) simplifies and enhances Python's decorator capabilities, making your code more reliable and maintainable.**

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features of wrapt

*   **Universal Decorators:** Works seamlessly with functions, methods (including classmethods and staticmethods), and classes.
*   **Transparent Object Proxies:** Provides advanced wrapping capabilities for complex scenarios.
*   **Safe Monkey Patching:** Includes utilities for safe and reliable runtime modifications.
*   **Optimized Performance:** Leverages a C extension module for performance-critical components, with a Python fallback.
*   **Introspection Preservation:** Preserves crucial information like signatures and annotations for robust code analysis.
*   **Thread-Safe Implementations:** Ensures reliable behavior in multi-threaded environments.

## Installation

Install `wrapt` easily using pip:

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

For detailed information, examples, and advanced usage, explore the comprehensive documentation:

*   **[wrapt.readthedocs.io](https://wrapt.readthedocs.io/)**

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome!  Please use the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to discuss any changes, improvements, or bug reports.

Note:  wrapt is considered a mature project.  The focus is on maintaining compatibility and ensuring correct functionality.

## License

This project is licensed under the BSD License.  See the [LICENSE](LICENSE) file for more details.

## Links

*   **Documentation**: [wrapt.readthedocs.io](https://wrapt.readthedocs.io/)
*   **PyPI**: [https://pypi.python.org/pypi/wrapt](https://pypi.python.org/pypi/wrapt)
*   **Issues**: [https://github.com/GrahamDumpleton/wrapt/issues/](https://github.com/GrahamDumpleton/wrapt/issues/)
*   **Changelog**: [https://wrapt.readthedocs.io/en/latest/changes.html](https://wrapt.readthedocs.io/en/latest/changes.html)