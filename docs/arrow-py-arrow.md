# Arrow: Effortless Date and Time Handling in Python

**Simplify your Python date and time operations with Arrow, a user-friendly library that makes working with dates, times, and timestamps a breeze.**  Explore the original repository [here](https://github.com/arrow-py/arrow).

## Key Features of Arrow

*   **Intuitive DateTime Replacement:** Provides a fully-implemented, drop-in replacement for Python's built-in `datetime` module.
*   **Python 3.8+ Compatibility:**  Works seamlessly with Python 3.8 and newer versions.
*   **Timezone Awareness by Default:**  Handles timezones intelligently and defaults to UTC for clarity.
*   **Simplified Creation:** Offers easy-to-use methods for creating date and time objects from various input formats.
*   **Flexible Time Shifting:**  Uses a convenient `shift` method for relative offsets, including weeks, for easy date manipulation.
*   **Automatic String Formatting and Parsing:**  Automatically formats and parses strings, making data handling more efficient.
*   **ISO 8601 Support:**  Provides comprehensive support for the widely used ISO 8601 standard.
*   **Timezone Conversion:**  Easily convert between timezones.
*   **Dateutil, Pytz, and ZoneInfo Support:**  Integrates with these popular time zone libraries.
*   **Time Span Generation:**  Creates time spans, ranges, floors, and ceilings for various time frames (microseconds to years).
*   **Human-Friendly Dates:**  Humanizes dates and times with support for multiple locales.
*   **Extensible:** Allows you to create your own Arrow-derived types.
*   **Type Hints:** Supports PEP 484-style type hints for improved code readability and maintainability.

## Why Choose Arrow?

Arrow simplifies date and time operations by addressing the usability issues found in Python's standard library:

*   Reduces the need to import numerous modules.
*   Simplifies working with multiple date and time types.
*   Provides a more intuitive and user-friendly approach to timezones and timestamp conversions.
*   Offers more straightforward ISO 8601 parsing, timespans, and humanization features.

## Quick Start

**Installation:**

Install Arrow using pip:

```bash
pip install -U arrow
```

**Example Usage:**

```python
import arrow

# Parse a datetime string
a = arrow.get('2013-05-11T21:23:58.970460+07:00')
print(a)  # Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
print(utc)  # Output: <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift the time
utc = utc.shift(hours=-1)
print(utc)  # Output: <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to a different timezone
local = utc.to('US/Pacific')
print(local)  # Output: <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get the timestamp
print(local.timestamp()) # Output: 1368303838.97046

# Format the date and time
print(local.format())  # Output: 2013-05-11 13:23:58 -07:00
print(local.format('YYYY-MM-DD HH:mm:ss ZZ')) # Output: 2013-05-11 13:23:58 -07:00

# Humanize the date and time
print(local.humanize())  # Output: an hour ago
print(local.humanize(locale='ko-kr')) # Output: 한시간 전
```

## Documentation

For detailed information, explore the official documentation at [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

We welcome contributions!  See the [issue tracker](https://github.com/arrow-py/arrow/issues) for tasks.  Start by forking the repository and following the contribution guidelines in the original README.

## Support Arrow

Consider supporting the project by donating via [Open Collective](https://opencollective.com/arrow) to help ensure the ongoing development and maintenance of Arrow.