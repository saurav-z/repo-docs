<img src="header.jpg" alt="KCC Header Image" width="400">

# Kindle Comic Converter (KCC): The Ultimate Tool for Reading Comics and Manga on Your E-reader

**KCC** is a free and open-source software that optimizes your favorite comics and manga for e-ink devices, ensuring a crisp, clear reading experience. **[Get the latest release here](https://github.com/ciromattia/kcc)**.

## Key Features

*   **Broad E-reader Support:** Converts comics and manga for Kindle, Kobo, reMarkable, and other e-readers.
*   **Versatile Input:** Supports JPG, PNG, GIF, WebP images, and archives (CBZ, CBR, ZIP, RAR, 7Z) and PDFs.
*   **Multiple Output Formats:** Creates MOBI/AZW3, EPUB, KEPUB, CBZ, and PDF files. PDF output is optimized for reMarkable devices.
*   **E-ink Optimization:**  Processes images to improve contrast and readability on e-ink screens, reduce file size, and enhance performance.
*   **Fixed Layout Support:** Ensures pages display in fullscreen without margins for a clean and immersive reading experience.
*   **Manga Support:**  Correctly handles right-to-left reading and page splitting for manga.
*   **User-Friendly GUI:**  Easy-to-use graphical interface with tooltips for all settings.
*   **Batch Conversion:**  Convert multiple files and folders with ease.
*   **Direct USB Transfer:** Simply drag and drop the converted files to your e-reader via USB.

## What's New

*   **PDF Support for reMarkable:**  Directly convert comics to PDF format for optimal compatibility with reMarkable devices (Rmk1, Rmk2, RmkPP).

## Why Use KCC?

KCC addresses common formatting issues often found in e-reader comic content, such as:

*   Faded black levels
*   Unnecessary margins
*   Incorrect page orientation
*   Unaligned two-page spreads

## Usage

1.  **Download:** Obtain the latest version from the [releases page](https://github.com/ciromattia/kcc/releases). Download the appropriate executable for your operating system.
2.  **Drag and Drop:** Drag and drop your comic files (images, archives, or PDFs) into the KCC window.
3.  **Configure Settings:** Adjust the settings (hover over each option for details) to customize your conversion.
4.  **Convert:** Click the "Convert" button. Use `Shift` while clicking to change the output directory.
5.  **Transfer:** Drag and drop the generated output files onto your e-reader via USB.

## Additional Information

*   **FAQ:** Find answers to common questions in the [FAQ section](https://github.com/ciromattia/kcc/wiki/FAQ).
*   **Wiki:** Explore the [KCC Wiki](https://github.com/ciromattia/kcc/wiki/) for more detailed information and installation instructions.
*   **CLI Version:** For power users, a command-line interface is available (see details in the original README).
*   **Donations:** Support the developers via PayPal or Bitcoin (see the original README).
*   **Issues & Feedback:**  Report technical issues or suggest new features on the [issue tracker](https://github.com/ciromattia/kcc/issues/new) or the [MobileRead forum](http://www.mobileread.com/forums/showthread.php?t=207461).

## Prerequisites

*   **KindleGen:** Kindle Previewer is required for MOBI conversion, especially if you use Linux AppImage or Flatpak.
*   **7-Zip:**  Optional, but recommended for faster and more reliable conversions, especially for archived files.  KCC will prompt to install it if needed.

## Input Formats

KCC supports the following input formats:

*   Folders containing: PNG, JPG, GIF or WebP files
*   CBZ, ZIP *(With `7z` executable)*
*   CBR, RAR *(With `7z` executable)*
*   CB7, 7Z *(With `7z` executable)*
*   PDF *(Only extracting JPG images)*

## License

KCC is released under the ISC LICENSE; see [LICENSE.txt](./LICENSE.txt) for further details.