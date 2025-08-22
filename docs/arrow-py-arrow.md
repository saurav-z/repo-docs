# Arrow: Dates and Times Made Easy for Python

**Simplify your Python date and time operations with Arrow, a user-friendly library designed for human readability and ease of use.**  For more details, visit the [original repository](https://github.com/arrow-py/arrow).

## Key Features

*   **Intuitive Date and Time Creation:** Easily create Arrow objects from various inputs.
*   **Timezone Awareness:**  Works seamlessly with timezones, UTC by default.
*   **Human-Friendly Formatting:**  Format dates and times with built-in options, and humanize them to be more readable.
*   **Flexible Time Manipulation:** Shift dates and times with relative offsets, including support for weeks.
*   **ISO 8601 Support:** Full support for parsing and formatting using the ISO 8601 standard.
*   **Time Span and Range Generation:** Create time spans and ranges for microsecond to year frames.
*   **Localization:** Supports humanizing dates and times with a growing list of locales.
*   **Drop-in Replacement:** Provides a fully implemented, drop-in replacement for the datetime module.

## Why Choose Arrow?

Arrow overcomes the complexities of Python's built-in datetime modules by:

*   Reducing the number of modules and types you need to import and understand.
*   Simplifying timezone conversions.
*   Offering intuitive methods for common tasks like ISO 8601 parsing, time spans, and humanization.

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Create an Arrow object from a string
arrow.get('2013-05-11T21:23:58.970460+07:00')
# <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
# <Arrow [2024-10-27T10:30:00.000000+00:00]>

# Shift the time by an hour
utc = utc.shift(hours=-1)
# <Arrow [2024-10-27T09:30:00.000000+00:00]>

# Convert to a different timezone
local = utc.to('US/Pacific')
# <Arrow [2024-10-27T02:30:00.000000-07:00]>

# Get a timestamp
local.timestamp()
# 1700000000.000000

# Format the date and time
local.format()
# '2024-10-27 02:30:00 -07:00'
local.format('YYYY-MM-DD HH:mm:ss ZZ')
# '2024-10-27 02:30:00 -07:00'

# Humanize the time
local.humanize()
# 'a few hours ago'
```

## Documentation

For in-depth information, refer to the official documentation at [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome!  Refer to the original README for details on how to contribute code, add locales, and more.