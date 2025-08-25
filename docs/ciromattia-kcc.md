<img src="header.jpg" alt="Header Image" width="400">

# Kindle Comic Converter (KCC)

**Transform your black & white comics and manga into optimized, full-screen reading experiences for your e-reader with KCC!**  [Visit the KCC Repo](https://github.com/ciromattia/kcc)

[![GitHub release](https://img.shields.io/github/release/ciromattia/kcc.svg)](https://github.com/ciromattia/kcc/releases)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ciromattia/kcc/docker-publish.yml?label=docker%20build)](https://github.com/ciromattia/kcc/pkgs/container/kcc)
[![Github All Releases](https://img.shields.io/github/downloads/ciromattia/kcc/total.svg)](https://github.com/ciromattia/kcc/releases)

KCC is a powerful and user-friendly tool designed to optimize comics and manga for e-ink e-readers, including Kindle, Kobo, ReMarkable, and more. It addresses common formatting issues and enhances the reading experience on devices with limited resources.

**Key Features:**

*   **Optimized Output:** Generates files with fullscreen pages without margins.
*   **Wide Format Support:**  Supports JPG/PNG/GIF images, archives (CBZ, CBR, ZIP, RAR, 7Z), and PDFs as input. Outputs to MOBI/AZW3, EPUB, KEPUB, CBZ, and PDF.
*   **ReMarkable PDF Support:**  Direct PDF conversion optimized for reMarkable devices.
*   **Image Processing:** Includes image processing steps to improve appearance on e-ink screens.
*   **Device-Specific Downscaling:** Downscales to your device's screen resolution for optimal quality and filesize reduction.
*   **Formatting Fixes:** Corrects common Kindle formatting problems such as faded black levels, incorrect page turn directions and spread alignment.
*   **User-Friendly GUI:** An intuitive Qt6-built graphical user interface for easy conversion.

**New Features**

*   PDF output is now supported for direct conversion to reMarkable devices! 

**How to Use:**

1.  Drag and drop your comic files (images, archives, or PDFs) into the KCC window.
2.  Adjust settings using the tooltips for detailed explanations.
3.  Click "Convert" to create e-reader-optimized files.
4.  Hold `Shift` while clicking Convert to change output directory
5.  Drag and drop the output files to your device's documents folder via USB.

**Downloads:**

*   **[Download the latest release here](https://github.com/ciromattia/kcc/releases)**

    *   Choose the appropriate file for your operating system (Windows, macOS, or other).

**FAQ**

*   **Q: Should I use Calibre?**
    *   A: No. Calibre does not properly support fixed layout EPUB/MOBI, so modifying KCC output in Calibre will break the formatting.
*   **Q: What output format should I use?**
    *   A: MOBI for Kindles. CBZ for Kindle DX. CBZ for Koreader. KEPUB for Kobo.

**For more information:**
* [Our Wiki](https://github.com/ciromattia/kcc/wiki/)
*   YouTube tutorial: [https://www.youtube.com/watch?v=IR2Fhcm9658](https://www.youtube.com/watch?v=IR2Fhcm9658)

**Get Involved:**

*   **Report Issues:**  [File an issue here](https://github.com/ciromattia/kcc/issues/new).
*   **Contribute:** Fork the repository and submit pull requests for bug fixes and new features.
*   **General questions/feedback:** [Post it here](http://www.mobileread.com/forums/showthread.php?t=207461).

**Donate**
If you find KCC valuable you can consider donating to the authors:
* Ciro Mattia Gonano (founder, active 2012-2014):
  [![Donate PayPal](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=D8WNYNPBGDAS2)
* Paweł Jastrzębski (active 2013-2019):
  [![Donate PayPal](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=YTTJ4LK2JDHPS)
  [![Donate Bitcoin](https://img.shields.io/badge/Donate-Bitcoin-green.svg)](https://jastrzeb.ski/donate/)
* Alex Xu (active 2023-Present)
  [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Q5Q41BW8HS)
**Credits:**

*   **Ciro Mattia Gonano, Paweł Jastrzębski, Darodi, and Alex Xu.**

**License:**
Released under the ISC License; see [LICENSE.txt](./LICENSE.txt) for details.