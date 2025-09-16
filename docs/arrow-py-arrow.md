# Arrow: Human-Friendly Dates and Times for Python

**Simplify your Python date and time manipulations with Arrow, a powerful library designed for ease of use and readability.**  Check out the [Arrow repository on GitHub](https://github.com/arrow-py/arrow) for more details.

## Key Features

Arrow offers a streamlined approach to working with dates, times, and timestamps, making your code cleaner and more efficient:

*   **Intuitive API:**  Easily create, manipulate, and format dates and times.
*   **Simplified Timezone Handling:**  Work with timezones effortlessly, with UTC as the default.
*   **ISO 8601 Support:**  Seamlessly parse and format dates and times according to the ISO 8601 standard.
*   **Relative Offsets & Time Span Generation:**  Use the `shift` method and generate time spans for easy time calculations, supporting weeks and more.
*   **Human-Friendly Formatting:**  Convert timestamps into human-readable phrases with a growing list of locales.
*   **Full Datetime Replacement:** A fully-implemented, drop-in replacement for datetime.
*   **Python 3.8+ Compatibility:**  Works with current Python versions.
*   **Flexible:**  Supports `dateutil`, `pytz`, and `ZoneInfo` tzinfo objects.

## Why Use Arrow?

Arrow addresses the usability shortcomings of Python's built-in datetime modules, including:

*   **Reduced Complexity:**  Fewer modules and types to manage.
*   **Simplified Timezone Conversions:**  Easy handling of timezone conversions.
*   **Enhanced Functionality:**  Provides features like ISO 8601 parsing and humanization that are missing in the standard library.
*   **Improved Usability:**  Makes common date and time operations significantly more intuitive.

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Parse a datetime string
dt = arrow.get('2023-10-27T10:00:00+00:00')
print(dt)  # Output: <Arrow [2023-10-27T10:00:00+00:00]>

# Get the current UTC time
utc = arrow.utcnow()
print(utc)  # Output: <Arrow [2023-10-27T10:00:00+00:00]>

# Shift the time forward one hour
shifted = utc.shift(hours=1)
print(shifted)  # Output: <Arrow [2023-10-27T11:00:00+00:00]>

# Convert to a different timezone
local = shifted.to('US/Pacific')
print(local) # Output: <Arrow [2023-10-27T04:00:00-07:00]>

# Format the time
print(local.format('YYYY-MM-DD HH:mm:ss ZZ'))  # Output: 2023-10-27 04:00:00 -07:00

# Get a human-readable representation
print(local.humanize())  # Output: an hour ago
```

## Documentation

For detailed information and advanced usage, please visit [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Arrow welcomes contributions! To get involved:

1.  Find an issue or feature on the [issue tracker](https://github.com/arrow-py/arrow/issues).
2.  Fork the repository and create a branch for your changes.
3.  Add tests to verify your code.
4.  Run the test suite and linting checks using `tox && tox -e lint,docs` or `make build39 && make test && make lint` (replace `build39` with your Python version).
5.  Submit a pull request.

For questions, use the [discussions](https://github.com/arrow-py/arrow/discussions).

## Support Arrow

Support the project through [Open Collective](https://opencollective.com/arrow) to help ensure its continued development.