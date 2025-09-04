<img src="header.jpg" alt="Header Image" width="400">

# Kindle Comic Converter (KCC): Optimize Comics & Manga for E-readers

**Tired of poorly formatted comics on your e-reader?** Kindle Comic Converter (KCC) is a powerful, open-source tool designed to transform your black & white comics and manga into stunning, optimized formats for e-ink devices like Kindle, Kobo, and reMarkable, delivering a superior reading experience. [**Get the latest release here!**](https://github.com/ciromattia/kcc/releases)

## Key Features:

*   **Wide Format Support:** Convert JPG, PNG, GIF, and PDF files into MOBI/AZW3, EPUB, KEPUB, CBZ, and PDF formats, ensuring compatibility with various e-readers.
*   **Optimized Image Processing:** Enhance image quality for e-ink screens with customizable options like gamma correction and black/white level adjustments, reducing eyestrain and improving contrast.
*   **Device-Specific Profiles:** Choose from pre-configured profiles for popular e-readers to automatically optimize for your device's resolution and screen size, maximizing the full screen display.
*   **Full-Screen Experience:** Eliminate margins and ensure proper fixed layout support for immersive comic and manga reading.
*   **Manga Support:** Correctly handles right-to-left reading and page splitting for manga.
*   **PDF Output for reMarkable:** Directly convert to PDF for optimal compatibility with reMarkable devices.
*   **Batch Conversion:** Quickly convert multiple files and folders with the drag-and-drop interface.
*   **File Size Reduction:** Downscale images to your device's resolution, significantly reducing file sizes without sacrificing visual quality, improving storage and page turn speeds.

## What's New?

*   **NEW:** PDF output is now supported for direct conversion to reMarkable devices! 

## Why Use KCC?

KCC addresses common formatting issues found even in official comic releases, such as:

*   Faded black levels and low contrast.
*   Unnecessary margins.
*   Incorrect page turn direction for manga.
*   Misaligned two-page spreads.

## Usage:

1.  **Download:** Obtain the latest version from the [releases page](https://github.com/ciromattia/kcc/releases).  Choose the appropriate executable for your operating system.
2.  **Drag and Drop:** Drag your comic or manga files/folders into the KCC window.
3.  **Customize Settings:** Adjust conversion options (hover for detailed tooltips) to fine-tune image processing and output format.
4.  **Convert:** Click the "Convert" button to create the optimized files. Hold `Shift` while clicking to change the output directory.
5.  **Transfer:** Drag and drop the generated output files to your e-reader's documents folder via USB.

**Watch the YouTube tutorial:** [https://www.youtube.com/watch?v=IR2Fhcm9658](https://www.youtube.com/watch?v=IR2Fhcm9658)

## Important Notes:

*   **KCC is not affiliated with Amazon's Kindle Comic Creator.** It's a user-friendly tool for readers.
*   For questions, feedback, and usage help, use the [MobileRead Forums](http://www.mobileread.com/forums/showthread.php?t=207461).
*   Report **technical** issues on the [GitHub issue tracker](https://github.com/ciromattia/kcc/issues/new).

## Donations & Contributors:

Support the development of KCC:

*   [Ciro Mattia Gonano (founder, active 2012-2014): PayPal](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=D8WNYNPBGDAS2)
*   [Paweł Jastrzębski (active 2013-2019): PayPal](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=YTTJ4LK2JDHPS) / [Bitcoin](https://jastrzeb.ski/donate/)
*   [Alex Xu (active 2023-Present): Ko-Fi](https://ko-fi.com/Q5Q41BW8HS)

## Commissions

Email (for commisions and inquiries): `kindle.comic.converter` gmail

## Sponsors

- Free code signing on Windows provided by [SignPath.io](https://about.signpath.io/), certificate by [SignPath Foundation](https://signpath.org/)

## Frequently Asked Questions (FAQ):

*   **Should I use Calibre?** No. Calibre doesn't properly support fixed layout EPUB/MOBI, so avoid modifying KCC output in Calibre.
*   **What output format should I use?** MOBI for Kindles, CBZ for Kindle DX and Koreader, KEPUB for Kobo.
*   **Colors inverted?** Disable Kindle dark mode.
*   **Cannot connect Kindle Scribe or 2024+ Kindle to macOS?** Use official MTP [Amazon USB File Transfer app](https://www.amazon.com/gp/help/customer/display.html/ref=hp_Connect_USB_MTP?nodeId=TCUBEdEkbIhK07ysFu) (no login required).

## Prerequisites & Installation (for developers):

*   Install dependencies: `pip install -r requirements.txt`
*   See [Installation Wiki](https://github.com/ciromattia/kcc/wiki/Installation) for optional dependencies like 7-Zip and KindleGen
*   Follow the instructions in the original README for building and running the application from source.

## Input & Output Formats:

**Input:** JPG, PNG, GIF, WebP, CBZ, ZIP, CBR, RAR, CB7, 7Z, PDF

**Output:** MOBI/AZW3, EPUB, KEPUB, CBZ, PDF

## Credits:

*   Ciro Mattia Gonano, Paweł Jastrzębski, Darodi and Alex Xu.
*   Based on KindleComicParser by Dc5e.
*   Includes scripts by K. Hendricks, Alex Yatskov, proDOOMman, Birua.
*   Icon by Nikolay Verin.

## [View the full source code and contribute to the project on GitHub!](https://github.com/ciromattia/kcc)

## Privacy
**KCC** is initiating internet connections in two cases:
* During startup - Version check.
* When error occurs - Automatic reporting on Windows and macOS.

## Copyright & License:

Copyright (c) 2012-2025 Ciro Mattia Gonano, Paweł Jastrzębski, Darodi and Alex Xu.
**KCC** is released under ISC LICENSE; see [LICENSE.txt](./LICENSE.txt) for further details.