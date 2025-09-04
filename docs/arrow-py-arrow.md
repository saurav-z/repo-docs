# Arrow: Human-Friendly Dates and Times for Python

**Simplify your Python date and time manipulations with Arrow, a powerful library designed for intuitive and efficient datetime handling.**  ([View the original repo](https://github.com/arrow-py/arrow))

Arrow provides a more sensible and user-friendly way to work with dates, times, and timestamps in Python. It addresses the complexities of the standard library with a clean and accessible API.

## Key Features:

*   **Simplified Creation:** Easily create `Arrow` objects from various input formats.
*   **Timezone Awareness:**  Works seamlessly with timezones and UTC by default.
*   **Flexible Manipulation:**  Shift dates and times with relative offsets (e.g., weeks, months).
*   **Automatic Formatting & Parsing:**  Intelligent string formatting and parsing, including ISO 8601 support.
*   **Timezone Conversion:** Effortlessly convert between timezones.
*   **Humanization:**  Generate human-readable representations of dates and times (e.g., "2 days ago").
*   **Extensible:**  Create custom Arrow-derived types.
*   **Type Hinting:**  Full support for PEP 484-style type hints.
*   **Drop-in Replacement:** Fully implements a drop-in replacement for the `datetime` module.

## Why Use Arrow?

Arrow addresses the common usability issues found in Python's built-in `datetime` and related modules:

*   **Reduces complexity:** Streamlines the use of modules and types.
*   **Simplifies timezone management:** Makes timezone conversions and handling more straightforward.
*   **Provides missing functionality:** Offers ISO 8601 parsing, timespans, and humanization.

## Quick Start

### Installation

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Parse a datetime string
a = arrow.get('2013-05-11T21:23:58.970460+07:00')
print(a)  # Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
print(utc)  # Output: <Arrow [2024-01-20T12:34:56.789000+00:00]> (example)

# Shift time
utc = utc.shift(hours=-1)
print(utc)  # Output: <Arrow [2024-01-20T11:34:56.789000+00:00]> (example)

# Convert to a different timezone
local = utc.to('US/Pacific')
print(local)  # Output: <Arrow [2024-01-20T03:34:56.789000-08:00]> (example)

# Get a timestamp
print(local.timestamp()) # Output: 1705750496.789000 (example)

# Format the time
print(local.format()) # Output: 2024-01-20 03:34:56 -08:00 (example)

print(local.format('YYYY-MM-DD HH:mm:ss ZZ')) # Output: 2024-01-20 03:34:56 -08:00 (example)

# Humanize
print(local.humanize()) # Output: 8 hours ago (example)
```

## Documentation

For comprehensive documentation, visit [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome! Please review the [contributing guidelines](https://github.com/arrow-py/arrow/blob/master/CONTRIBUTING.md).

## Support Arrow

Consider supporting Arrow through [Open Collective](https://opencollective.com/arrow).