# yt-fts: Search YouTube Channels with Ease 🔎

Unleash the power of full-text search on YouTube channels by indexing subtitles, enabling you to find specific moments and information with ease.  [Check out the GitHub repo](https://github.com/NotJoeMartinez/yt-fts) for the latest updates and to contribute!

[![yt-fts in action](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

**Key Features:**

*   **Full-Text Search:**  Quickly find keywords and phrases within YouTube video transcripts.
*   **Semantic Search:**  Leverage OpenAI and Gemini embeddings (with ChromaDB support) for more nuanced and contextual searches.
*   **LLM-Powered Chatbot:** Interact with a chatbot trained on your chosen channel's content for in-depth answers.
*   **Video Summarization:**  Get concise summaries of YouTube videos with time-stamped links.
*   **Flexible Data Management:** Download, update, list, delete, and export video transcripts with ease.
*   **Advanced Search Syntax:** Utilize SQLite's Enhanced Query Syntax for powerful, refined searches (AND, OR, wildcards, etc.)
*   **User-Friendly Interface:** Command-line interface for easy use and automation.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download` - Download Subtitles
Downloads subtitles for a channel or playlist.

```bash
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download playlist
*   `-l, --language`: Subtitle language (default: `en`)
*   `-j, --jobs`: Parallel download jobs (default: `8`)
*   `--cookies-from-browser`: Browser to use cookies from (chrome, firefox, etc.)

### `diagnose` - Troubleshoot Download Issues
Diagnose and provide recommendations for fixing common download errors.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test (default: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
*   `--cookies-from-browser`: Browser to use cookies from
*   `-j, --jobs`: Parallel download jobs (default: 8)

### `list` - View Saved Data
Lists saved channels, videos, and transcripts.

```bash
yt-fts list
yt-fts list --channel "3Blue1Brown"
yt-fts list --transcript "dQw4w9WgXcQ"
```

**Options:**

*   `-t, --transcript`: Show video transcript
*   `-c, --channel`: Show videos for a channel
*   `-l, --library`: Show list of channels in library

### `update` - Update Subtitles
Updates subtitles for all or a specific channel.

```bash
yt-fts update
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Channel to update
*   `-l, --language`: Subtitle language (default: `en`)
*   `-j, --jobs`: Parallel download jobs (default: `8`)
*   `--cookies-from-browser`: Browser to use cookies from

### `delete` - Delete Channels
Deletes a channel and all its associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel to delete (required)

### `export` - Export Transcripts
Exports transcripts for a channel in various formats.

```bash
yt-fts export --channel "3Blue1Brown" --format txt
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Channel to export (required)
*   `-f, --format`: Export format (txt, vtt, default: txt)

### `search` - Full Text Search
Searches for a string within saved channels.  Utilize advanced search operators for precise results.

```bash
yt-fts search "[search query]"
yt-fts search "[search query]" --channel "[channel name or id]"
yt-fts search "[search query]" --video-id "[video id]"
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts search "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Channel to search in
*   `-v, --video-id`: Video ID to search in
*   `-l, --limit`: Result limit (default: 10)
*   `-e, --export`: Export results to CSV

**Advanced Search Syntax:**

Use SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries)

```bash
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### `embeddings` - Semantic Search Setup
Generates embeddings for semantic search (requires OpenAI or Gemini API key, set in `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variables).

```bash
yt-fts embeddings --channel "3Blue1Brown"
yt-fts embeddings --interval 60 --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel to generate embeddings for
*   `--api-key`: API key (if not provided, reads from environment variables)
*   `-i, --interval`: Transcript chunk interval (default: 30 seconds)

### `vsearch` - Semantic Search
Semantic search, which requires embeddings to be enabled for a channel.

```bash
yt-fts vsearch "[search query]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --video-id "[video id]"
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Channel to search in
*   `-v, --video-id`: Video ID to search in
*   `-l, --limit`: Result limit (default: 10)
*   `-e, --export`: Export results to CSV
*   `--api-key`: API key (if not provided, reads from environment variables)

### `llm` - Chatbot
Starts interactive chat session with a model using semantic search results as context.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Channel to use (required)
*   `--api-key`: API key (if not provided, reads from environment variables)

### `summarize` - Summarize Videos
Summarizes a YouTube video transcript.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
yt-fts summarize "9-Jl0dxWQs8"
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use
*   `--api-key`: API key (if not provided, reads from environment variables)

### `config` - View Configuration
Shows configuration settings.

```bash
yt-fts config
```

## How To

**Export Search Results:**

Export results to a CSV file.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**
```bash
yt-fts update --channel "3Blue1Brown"
```

**Export All Transcripts:**

```bash
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```