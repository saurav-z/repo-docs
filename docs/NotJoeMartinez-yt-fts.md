# yt-fts: Unlock YouTube's Knowledge - Powerful Full-Text & Semantic Search

**yt-fts** empowers you to deeply explore YouTube content with full-text search and semantic understanding of video transcripts. [Explore the yt-fts repository](https://github.com/NotJoeMartinez/yt-fts).

![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)

## Key Features:

*   **Full-Text Search:** Quickly find specific keywords or phrases within YouTube video transcripts.
*   **Semantic Search (Powered by AI):** Leverage OpenAI/Gemini embeddings to search by meaning and context for more relevant results.
*   **Advanced Search Syntax:** Use operators like AND, OR, and wildcards for highly refined searches.
*   **LLM Chat Bot:** Have conversational Q&A sessions with LLMs about channel content using semantic search results as context.
*   **Video Summarization:** Get concise summaries of YouTube videos with time-stamped links.
*   **Easy Installation:** Simple `pip install yt-fts` setup.
*   **Flexible Data Handling:** Download, update, list, delete, and export transcripts.
*   **Channel/Playlist Support:** Scrape entire channels or playlists.
*   **Error Diagnosis:** Easily troubleshoot download issues with the `diagnose` command.

## Installation

```bash
pip install yt-fts
```

## Commands:

### `download` - Download Subtitles
Download subtitles for a channel or playlist.

*   Use the channel or playlist URL as an argument.
*   Parallelize downloads with `--jobs`.
*   Use `--cookies-from-browser` for sign-in protected content.
*   Update command used to incrementally get more videos.

```bash
# Download channel
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"

# Download playlist
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download playlist
*   `-l, --language`: Subtitle language (default: `en`)
*   `-j, --jobs`: Parallel jobs (default: `8`, recommended: `4-16`)
*   `--cookies-from-browser`: Browser for cookie extraction

### `diagnose` - Diagnose Download Issues
Tests your connection to YouTube and provides recommendations to fix common problems.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test (default: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
*   `--cookies-from-browser`: Browser to use
*   `-j, --jobs`: Parallel jobs (default: `8`)

### `list` - List Saved Content
Lists saved channels, videos, and transcripts. The (ss) next to the channel name means semantic search is enabled.

```bash
# List all channels
yt-fts list

# List videos for a specific channel
yt-fts list --channel "3Blue1Brown"

# Show transcript for a specific video
yt-fts list --transcript "dQw4w9WgXcQ"

# Show library (same as default)
yt-fts list --library
```

**Options:**

*   `-t, --transcript`: Show transcript
*   `-c, --channel`: Show videos for a channel
*   `-l, --library`: Show channels in library

### `update` - Update Subtitles
Update subtitles for all channels or a specific channel. Some videos might not have subtitles enabled.

```bash
# Update all channels
yt-fts update

# Update specific channel
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Channel to update
*   `-l, --language`: Subtitle language (default: `en`)
*   `-j, --jobs`: Parallel jobs (default: `8`)
*   `--cookies-from-browser`: Browser for cookie extraction

### `delete` - Delete a Channel
Delete a channel and all its data. Requires confirmation.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel to delete (required)

### `export` - Export Transcripts
Exports transcripts in TXT or VTT format.

```bash
# Export to txt (default)
yt-fts export --channel "3Blue1Brown" --format txt

# Export to vtt
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Channel to export (required)
*   `-f, --format`: Export format (txt, vtt; default: txt)

### `search` - Full Text Search
Performs full-text search within saved transcripts. Search strings are limited to 40 characters.

```bash
# search in all channels
yt-fts search "[search query]" 

# search in channel 
yt-fts search "[search query]" --channel "[channel name or id]" 

# search in specific video
yt-fts search "[search query]" --video-id "[video id]"

# limit results 
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# export results to csv
yt-fts search "[search query]" --export --channel "[channel name or id]" 
```

**Options:**

*   `-c, --channel`: Channel to search within
*   `-v, --video-id`: Video ID to search within
*   `-l, --limit`: Results limit (default: 10)
*   `-e, --export`: Export results to CSV

**Advanced Search Syntax:**  Uses [SQLite's Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries).

```bash
# AND search
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show" 

# OR SEARCH 
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show" 

# wild cards
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show" 
```

### `embeddings` - Semantic Search Setup

Enables semantic search for a channel. Requires an OpenAI or Gemini API key set in the `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable or using the `--api-key` flag.

```bash
# make sure API key is set
# export OPENAI_API_KEY="[yourOpenAIKey]"
# or
# export GEMINI_API_KEY="[yourGeminiKey]"

yt-fts embeddings --channel "3Blue1Brown"

# specify time interval in seconds to split text by default is 30 
# the larger the interval the more accurate the llm response  
# but semantic search will have more text for you to read. 
yt-fts embeddings --interval 60 --channel "3Blue1Brown" 
```

**Options:**

*   `-c, --channel`: Channel to generate embeddings for
*   `--api-key`: API key (reads from env if not provided)
*   `-i, --interval`: Transcript chunk interval (default: 30 seconds)

After generating embeddings, you'll see `(ss)` next to the channel in `list`, and you can use `vsearch`.

### `vsearch` - Semantic Search
Performs semantic (vector) search. Requires that you enable semantic search for a channel with `embeddings`.

```bash
# search by channel name
yt-fts vsearch "[search query]" --channel "[channel name or id]"

# search in specific video
yt-fts vsearch "[search query]" --video-id "[video id]"

# limit results 
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# export results to csv
yt-fts vsearch "[search query]" --export --channel "[channel name or id]" 
```

**Options:**

*   `-c, --channel`: Channel to search within
*   `-v, --video-id`: Video ID to search within
*   `-l, --limit`: Results limit (default: 10)
*   `-e, --export`: Export results to CSV
*   `--api-key`: API key (reads from env if not provided)

### `llm` - LLM Chat Bot
Starts an interactive chat session using semantic search results as context. Requires semantic search to be enabled for a channel.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Channel to use (required)
*   `--api-key`: API key (reads from env if not provided)

### `summarize` - Video Summarization
Summarizes a YouTube video transcript, providing time stamped URLS. Requires a valid YouTube video URL or ID.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
# or
yt-fts summarize "9-Jl0dxWQs8"

# Use different model
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use
*   `--api-key`: API key (reads from env if not provided)

### `config` - Show Configuration
Displays current configuration settings (database, chroma paths, etc.).

```bash
yt-fts config
```

## How To

**Export Search Results:**

For both `search` and `vsearch`, use the `--export` flag to save results to a CSV file.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**

Use the `delete` command.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**

The `update` command currently only updates full text search and will not update the semantic search embeddings.

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export all of a channel's transcript:**

```bash
# Export to vtt
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```