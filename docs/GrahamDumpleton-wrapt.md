# wrapt: Powerful Python Decorators, Wrappers, and Monkey Patching

**wrapt** is a robust Python module providing advanced tools for creating decorators, wrappers, and monkey patching solutions. ([See the original repository](https://github.com/GrahamDumpleton/wrapt))

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features of wrapt

*   **Universal Decorators:** Work seamlessly with functions, methods (including classmethods and staticmethods), and classes.
*   **Transparent Object Proxies:** Enable advanced wrapping scenarios and flexible control.
*   **Safe Monkey Patching Utilities:** Modify code at runtime with confidence.
*   **Optimized Performance:** Includes a C extension for speed, with a Python fallback for broader compatibility.
*   **Introspection Preservation:** Maintains function signatures, annotations, and other critical metadata.
*   **Thread-Safe Implementations:** Ensures reliable operation in multi-threaded environments.

## Why Use wrapt?

wrapt goes beyond basic decorator implementations, offering greater correctness, introspection, and consistency.  It's designed to handle a wider range of scenarios than standard decorator techniques, providing more predictable and reliable behavior for your Python projects.

## Installation

Install `wrapt` using pip:

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

*   **Comprehensive Documentation:** [wrapt.readthedocs.io](https://wrapt.readthedocs.io/)
*   **PyPI:** [https://pypi.python.org/pypi/wrapt](https://pypi.python.org/pypi/wrapt)
*   **Issues:** [https://github.com/GrahamDumpleton/wrapt/issues/](https://github.com/GrahamDumpleton/wrapt/issues/)
*   **Changelog:** [https://wrapt.readthedocs.io/en/latest/changes.html](https://wrapt.readthedocs.io/en/latest/changes.html)
*   **Related Blog Posts**:  A series of blog posts explain the design and implementation of wrapt are available in the repository (see original README).

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome!  Please use the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to discuss ideas, report bugs, or ask questions.

*Note:*  wrapt is a mature project, and the focus is on maintenance and compatibility with newer Python versions.

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.