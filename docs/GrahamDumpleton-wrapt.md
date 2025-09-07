# Wrapt: Python Decorators, Wrappers, and Monkey Patching - Simplified

**Enhance your Python code with powerful and reliable decorators, wrappers, and monkey patching capabilities using the Wrapt library.**  [See the original repo on GitHub](https://github.com/GrahamDumpleton/wrapt).

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Overview

The **wrapt** Python module provides a robust and transparent object proxy, making it easy to create function wrappers, decorators, and apply safe monkey patching. Wrapt focuses on correctness and addresses limitations of standard decorator implementations, ensuring that decorators maintain introspection, signatures, and type checking.

## Key Features

*   **Universal Decorators:** Works seamlessly with functions, methods (including instance methods, classmethods, and staticmethods), and classes.
*   **Transparent Object Proxies:**  Provides advanced wrapping capabilities for complex scenarios.
*   **Safe Monkey Patching:** Offers utilities for modifying Python code at runtime without unexpected issues.
*   **Optimized Performance:** Includes a C extension for speed, with a Python fallback for systems without a compiler.
*   **Introspection Preservation:**  Maintains function signatures, annotations, and other essential metadata for accurate behavior.
*   **Thread-Safe Implementations:** Ensures that decorator operations are safe in multithreaded environments.

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

For comprehensive documentation, examples, and in-depth explanations, visit:

[**wrapt.readthedocs.io**](https://wrapt.readthedocs.io/)

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome. Please use the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to discuss changes, improvements, or report issues.

## License

This project is licensed under the BSD License.  See the [LICENSE](LICENSE) file for details.

## Links

*   **Documentation:** https://wrapt.readthedocs.io/
*   **PyPI:** https://pypi.python.org/pypi/wrapt
*   **Issues:** https://github.com/GrahamDumpleton/wrapt/issues/
*   **Changelog:** https://wrapt.readthedocs.io/en/latest/changes.html