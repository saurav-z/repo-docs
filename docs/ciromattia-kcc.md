<img src="header.jpg" alt="Header Image" width="400">

# Kindle Comic Converter (KCC)

**Transform your comic and manga collection into optimized, high-quality ebooks for your e-reader with Kindle Comic Converter (KCC) – the ultimate solution for e-ink devices!**  [Visit the original repository](https://github.com/ciromattia/kcc)

[![GitHub release](https://img.shields.io/github/release/ciromattia/kcc.svg)](https://github.com/ciromattia/kcc/releases)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ciromattia/kcc/docker-publish.yml?label=docker%20build)](https://github.com/ciromattia/kcc/pkgs/container/kcc)
[![Github All Releases](https://img.shields.io/github/downloads/ciromattia/kcc/total.svg)](https://github.com/ciromattia/kcc/releases)

KCC is a powerful, open-source tool designed to optimize comic books and manga for e-ink e-readers, delivering a superior reading experience. It supports a wide range of devices, including Kindle, Kobo, ReMarkable, and more.

**Key Features:**

*   **Broad Format Support:** Input formats include JPG/PNG/GIF image files in folders, archives (CBZ/CBR/RAR/7Z), and PDF. Outputs MOBI/AZW3, EPUB, KEPUB, CBZ, and PDF.
*   **Optimized for E-ink:** Processes images to improve contrast and readability on e-ink screens, reducing eye strain.
*   **Full-Screen Display:** Ensures pages display in fullscreen without margins, utilizing your device's full resolution.
*   **Device-Specific Profiles:** Includes profiles for popular e-readers, optimizing settings for each device's screen.
*   **File Size Reduction:** Downscales images to your device's resolution, significantly reducing file sizes without noticeable quality loss, saving space and improving performance.
*   **Manga Support:** Correctly handles right-to-left reading and two-page spreads.
*   **PDF Output for reMarkable:** Direct conversion to PDF optimized for reMarkable devices.
*   **User-Friendly GUI:** An intuitive graphical user interface built with Qt6 makes conversion easy.

**NEW**: PDF output is now supported for direct conversion to reMarkable devices! When using a reMarkable profile (Rmk1, Rmk2, RmkPP), the format automatically defaults to PDF for optimal compatibility with your device's native PDF reader.

**Download:**

*   **https://github.com/ciromattia/kcc/releases**

**How to Use:**

1.  Download the appropriate executable from the Releases page.
2.  Drag and drop your comic files or folders into the KCC window.
3.  Adjust the settings based on your device (hover over each option for tooltips).
4.  Click "Convert" to create your optimized ebook.
5.  Transfer the generated file to your e-reader via USB.

**FAQ**

*   **What output format should I use?** MOBI for Kindles. CBZ for Kindle DX. CBZ for Koreader. KEPUB for Kobo.
*   **Should I use Calibre?** No. Calibre doesn't properly support fixed layout EPUB/MOBI, so modifying KCC output in Calibre will break the formatting.
*   **Colors inverted?** Disable Kindle dark mode

**For more detailed information and troubleshooting, please see the wiki: https://github.com/ciromattia/kcc/wiki/Installation**

**Contributing & Support:**

*   For general questions and feedback, please visit the [mobileread forum](http://www.mobileread.com/forums/showthread.php?t=207461).
*   To report technical issues, please [file an issue here](https://github.com/ciromattia/kcc/issues/new).
*   If you can fix an open issue, fork & make a pull request.

**Donations:**

If you find **KCC** valuable you can consider donating to the authors:

*   Ciro Mattia Gonano (founder, active 2012-2014):  [![Donate PayPal](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=D8WNYNPBGDAS2)
*   Paweł Jastrzębski (active 2013-2019):  [![Donate PayPal](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=YTTJ4LK2JDHPS) [![Donate Bitcoin](https://img.shields.io/badge/Donate-Bitcoin-green.svg)](https://jastrzeb.ski/donate/)
*   Alex Xu (active 2023-Present)  [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Q5Q41BW8HS)

**Credits:**
*   [Ciro Mattia Gonano](http://github.com/ciromattia)
*   [Paweł Jastrzębski](http://github.com/AcidWeb)
*   [Darodi](http://github.com/darodi)
*   [Alex Xu](http://github.com/axu2)

**Copyright:**

Copyright (c) 2012-2025 Ciro Mattia Gonano, Paweł Jastrzębski, Darodi and Alex Xu. **KCC** is released under ISC LICENSE; see [LICENSE.txt](./LICENSE.txt) for further details.

Impact-Site-Verification: ffe48fc7-4f0c-40fd-bd2e-59f4d7205180
```
Key improvements and SEO considerations:

*   **Strong Hook:**  The one-sentence hook immediately introduces the tool and its core benefit, optimizing for e-readers.
*   **Clear Headings:**  Uses descriptive headings (Key Features, Download, How to Use, FAQ, Contributing, Credits, Copyright) for better readability and SEO.
*   **Bulleted Key Features:**  Highlights the most important aspects of KCC in an easy-to-scan format, which is good for users and SEO.
*   **Keyword Optimization:** Includes relevant keywords like "Kindle," "comic," "manga," "e-reader," "e-ink," "conversion," "optimize," "MOBI," "EPUB," "CBZ," "PDF," and the names of popular e-readers (Kobo, ReMarkable) throughout the text.
*   **Download Link Prominently Displayed:** Places the download link in a highly visible location.
*   **Actionable Instructions:** Provides clear steps on how to use KCC.
*   **FAQ Section:** Addresses common user questions, making the documentation more helpful and potentially increasing search traffic for those questions.
*   **Concise and Scannable:** The text is broken down into short paragraphs and bullet points to improve readability.
*   **Link to the Original Repo:**  This is added to help users easily find the original repository.
*   **Alt text and width added to image.**
*   **Includes a concise explanation of what `KCC` does in the one-sentence hook.**