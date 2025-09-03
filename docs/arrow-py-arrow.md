# Arrow: Human-Friendly Dates and Times for Python

**Simplify date and time manipulation in Python with Arrow, a powerful and intuitive library.**  [View the Arrow Repository on GitHub](https://github.com/arrow-py/arrow)

Arrow provides a more sensible and user-friendly way to work with dates, times, and timestamps in Python, addressing the complexities of the built-in `datetime` module. It offers a clean API, timezone handling, and human-readable formatting.

## Key Features

*   **Simplified Creation:** Easily create Arrow objects from various input formats.
*   **Timezone Awareness:**  Timezone-aware by default, with easy conversion options.
*   **Intuitive Manipulation:**  Use the `shift` method for relative offsets (days, weeks, etc.).
*   **Automatic Formatting & Parsing:**  Format and parse strings with minimal code.
*   **ISO 8601 Support:**  Comprehensive support for the ISO 8601 standard.
*   **Time Span Operations:** Generate time spans, ranges, floors and ceilings for various time frames.
*   **Humanization:**  Human-readable date and time representations (e.g., "an hour ago").
*   **Type Hinting:** Full support for PEP 484-style type hints.

## Why Choose Arrow?

*   **Fewer Modules:**  Avoid juggling multiple Python modules (datetime, time, etc.).
*   **Reduced Complexity:**  Simplify working with dates, times, and timezones.
*   **Human-Readable Code:**  Write cleaner, more maintainable date/time code.

## Quick Start

**Installation:**

```bash
pip install -U arrow
```

**Example Usage:**

```python
import arrow

# Create an Arrow object from an ISO 8601 timestamp
a = arrow.get('2013-05-11T21:23:58.970460+07:00')
print(a)  # Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
print(utc)  # Output: <Arrow [2024-10-27T18:45:30.123456+00:00]> (example)

# Shift the time back one hour
utc = utc.shift(hours=-1)
print(utc)  # Output: <Arrow [2024-10-27T17:45:30.123456+00:00]> (example)

# Convert to a different timezone
local = utc.to('US/Pacific')
print(local) # Output: <Arrow [2024-10-27T10:45:30.123456-07:00]> (example)

# Get the timestamp (seconds since epoch)
print(local.timestamp()) # Output: 1700000000.123456 (example)

# Format the date and time
print(local.format())  # Output: 2024-10-27 10:45:30 -07:00 (example)
print(local.format('YYYY-MM-DD HH:mm:ss ZZ'))  # Output: 2024-10-27 10:45:30 -07:00 (example)

# Humanize the date
print(local.humanize())  # Output: an hour ago (example)
```

## Documentation

For comprehensive documentation, visit [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome!  Please see the original README for contribution guidelines.

## Support Arrow

Support the project via [Open Collective](https://opencollective.com/arrow).