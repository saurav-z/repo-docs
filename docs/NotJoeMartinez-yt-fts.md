# yt-fts: Supercharge Your YouTube Research with Full Text Search

Quickly search, analyze, and understand YouTube channels using powerful command-line tools.  [See the original repository here](https://github.com/NotJoeMartinez/yt-fts/).

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)

**Key Features:**

*   **Full-Text Search:** Search YouTube video transcripts using keywords, phrases, and advanced search syntax.
*   **Semantic Search:**  Leverage OpenAI or Gemini embeddings to find videos based on meaning and context.
*   **LLM-Powered Chat Bot:**  Engage in interactive conversations about video content.
*   **Video Summarization:** Get concise summaries of videos, with timestamped links to the relevant parts.
*   **Efficient Data Management:**  Download, store, and update subtitles for efficient access and analysis.
*   **Flexible Export Options:** Export search results and transcripts in various formats (CSV, TXT, VTT).
*   **Robust Installation:**  Easy installation via pip.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download`: Download Subtitles

Download subtitles for a YouTube channel or playlist.

*   `yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"`
*   `yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"`
*   `yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"`

**Options:**

*   `--playlist`: Download from a playlist
*   `--language`: Subtitle language (default: `en`)
*   `--jobs`: Parallel download jobs (default: 8)
*   `--cookies-from-browser`: Use browser cookies (chrome, firefox, etc.)

### `diagnose`: Troubleshoot Download Issues

Diagnose common download problems and get recommendations.

*   `yt-fts diagnose`
*   `yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox`

**Options:**

*   `--test-url`: Test URL (default: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
*   `--cookies-from-browser`: Browser to use for cookies
*   `--jobs`: Number of parallel jobs (default: 8)

### `list`: View Library Contents

List saved channels, videos, and transcripts.

*   `yt-fts list`  (Lists channels)
*   `yt-fts list --channel "3Blue1Brown"` (List videos for a channel)
*   `yt-fts list --transcript "dQw4w9WgXcQ"` (Show transcript for a video)

**Options:**

*   `--transcript`: Show transcript
*   `--channel`: Show videos for a channel
*   `--library`: Show list of channels

### `update`: Update Subtitles

Update subtitles for channels in your library.

*   `yt-fts update` (Updates all channels)
*   `yt-fts update --channel "3Blue1Brown" --jobs 5` (Updates a specific channel)

**Options:**

*   `--channel`: Channel to update
*   `--language`: Subtitle language (default: `en`)
*   `--jobs`: Parallel download jobs (default: 8)
*   `--cookies-from-browser`: Use browser cookies

### `delete`: Delete Channels

Delete a channel and all its associated data. Requires confirmation.

*   `yt-fts delete --channel "3Blue1Brown"`

**Options:**

*   `--channel`: Channel name or ID (required)

### `export`: Export Transcripts

Export transcripts for a channel.

*   `yt-fts export --channel "3Blue1Brown" --format txt` (Export to TXT)
*   `yt-fts export --channel "3Blue1Brown" --format vtt` (Export to VTT)

**Options:**

*   `--channel`: Channel name or ID (required)
*   `--format`: Export format (`txt`, `vtt`; default: `txt`)

### `search`: Full-Text Search

Search for keywords and phrases within saved transcripts.

*   `yt-fts search "[search query]"` (Search all channels)
*   `yt-fts search "[search query]" --channel "[channel name or id]"` (Search in a channel)
*   `yt-fts search "[search query]" --video-id "[video id]"` (Search in a video)

**Options:**

*   `--channel`: Channel to search
*   `--video-id`: Video ID to search
*   `--limit`: Max results (default: 10)
*   `--export`: Export results to CSV

**Advanced Search Syntax:**

Utilize SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for complex searches, including prefix and wildcard searches:

*   `yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"`
*   `yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"`

### `embeddings`:  Generate Semantic Embeddings

Enable semantic search by generating embeddings using OpenAI or Gemini API. Requires `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable or the `--api-key` flag.

*   `yt-fts embeddings --channel "3Blue1Brown"`
*   `yt-fts embeddings --interval 60 --channel "3Blue1Brown"`

**Options:**

*   `--channel`: Channel name or ID
*   `--api-key`: API key (if not provided, reads from env variables)
*   `--interval`: Chunking interval in seconds (default: 30)

### `vsearch`: Semantic Search

Perform semantic (vector) search on channels that have generated embeddings.

*   `yt-fts vsearch "[search query]" --channel "[channel name or id]"`
*   `yt-fts vsearch "[search query]" --video-id "[video id]"`

**Options:**

*   `--channel`: Channel name or ID
*   `--video-id`: Video ID
*   `--limit`: Max results (default: 10)
*   `--export`: Export results to CSV
*   `--api-key`: API key (if not provided, reads from env variables)

### `llm`: LLM-Powered Chat Bot

Start an interactive chat session with a model using semantic search context. Requires semantic search to be enabled for the channel.

*   `yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"`

**Options:**

*   `--channel`: Channel name or ID (required)
*   `--api-key`: API key (if not provided, reads from env variables)

### `summarize`: Video Summarization

Get a summarized overview of a YouTube video, including time-stamped links.

*   `yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"`
*   `yt-fts summarize "9-Jl0dxWQs8"`
*   `yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"`

**Options:**

*   `--model, -m`: Model to use (default: `gpt-4`)
*   `--api-key`: API key (if not provided, reads from env variables)

### `config`: Show Configuration

Displays your current configuration settings, including database and chroma paths.

*   `yt-fts config`

## How To

**Export Search Results:**
Export your search and vsearch results into CSV format using the `--export` flag:
```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**
Use the `delete` command to remove a channel and its data:
```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel's Subtitles:**
```bash
yt-fts update --channel "3Blue1Brown"