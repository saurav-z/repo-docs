# Arrow: Human-Friendly Dates and Times in Python

**Simplify your Python date and time manipulation with Arrow, a user-friendly library designed for clarity and ease of use.**

[View the Arrow Repository on GitHub](https://github.com/arrow-py/arrow)

Arrow provides a more intuitive and Pythonic way to work with dates, times, and timestamps, addressing the complexities of the built-in `datetime` module. It streamlines common tasks and enhances readability, making your code cleaner and more efficient.

## Key Features of Arrow:

*   **Simplified Date and Time Creation:** Easily create Arrow objects from various input formats.
*   **Intuitive Time Zone Handling:**  Timezone-aware by default and supports easy conversions.
*   **Human-Friendly Formatting:**  Format dates and times for readability, including support for multiple locales.
*   **Flexible Time Manipulation:**  Shift dates and times with relative offsets.
*   **Comprehensive ISO 8601 Support:** Parse and format dates using the ISO 8601 standard.
*   **Time Span and Range Generation:** Create time spans, ranges, and perform calculations.
*   **Integration with Existing Libraries:** Supports `dateutil`, `pytz`, and `ZoneInfo` objects.
*   **Type Hinting Support:**  Fully compatible with PEP 484-style type hints.

## Why Choose Arrow?

Arrow simplifies common date and time operations, reducing code complexity compared to using the built-in `datetime` module and other libraries. Arrow offers:

*   **Reduced Complexity:** Fewer modules and types to manage.
*   **Improved Readability:** More intuitive API for common operations.
*   **Enhanced Functionality:** Features like humanization and easy timezone conversion.

## Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

## Example Usage

```python
import arrow

# Create an Arrow object from a string
a = arrow.get('2013-05-11T21:23:58.970460+07:00')
print(a) # <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Convert to UTC
utc = a.to('UTC')
print(utc) # <Arrow [2013-05-11T14:23:58.970460+00:00]>

# Shift time
past = utc.shift(hours=-1)
print(past) # <Arrow [2013-05-11T13:23:58.970460+00:00]>

# Format the time
print(past.format('YYYY-MM-DD HH:mm:ss ZZ'))  # 2013-05-11 13:23:58 +00:00
print(past.humanize()) # 'an hour ago'
```

## Documentation

For detailed information and examples, please visit the [official documentation](https://arrow.readthedocs.io/).

## Contributing

Contributions to Arrow are welcome! See the [contributing guidelines](https://github.com/arrow-py/arrow/blob/master/CONTRIBUTING.md) for details on how to contribute code, documentation, or translations.

## Support Arrow

Support the development of Arrow by contributing financially via [Open Collective](https://opencollective.com/arrow).