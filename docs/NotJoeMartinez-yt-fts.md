# yt-fts: Your Command-Line YouTube Search Engine

Quickly search and analyze YouTube video transcripts with `yt-fts`, a powerful command-line tool that leverages [yt-dlp](https://github.com/yt-dlp/yt-dlp) to bring advanced search capabilities to your favorite YouTube channels.

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

**Key Features:**

*   **Full-Text Search:** Search through YouTube video transcripts using keywords and phrases.
*   **Semantic Search:** Utilize OpenAI/Gemini embeddings for more nuanced, context-aware searches.
*   **LLM Integration:** Interact with a chatbot that uses the search results as context.
*   **Video Summarization:** Generate concise summaries of YouTube video transcripts.
*   **Flexible Export Options:** Export search results and transcripts to CSV, VTT, or TXT formats.
*   **Easy Installation:** Simple `pip` installation for quick setup.

**Get Started:**

1.  **Installation:**

    ```bash
    pip install yt-fts
    ```

2.  **Download Subtitles:** Use the `download` command to populate your database.

    ```bash
    yt-fts download "https://www.youtube.com/@3blue1brown"
    ```

3.  **Search:** Use the `search` command to find specific keywords or phrases.

    ```bash
    yt-fts search "machine learning" --channel "3Blue1Brown"
    ```

**Detailed Command Reference:**

*   **`download`:** Download subtitles for channels or playlists.
    *   Options: `--playlist`, `--language`, `--jobs`, `--cookies-from-browser`.

    ```bash
    yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
    ```

*   **`diagnose`:** Diagnose and troubleshoot download issues.
    *   Options: `--test-url`, `--cookies-from-browser`, `-j`.

    ```bash
    yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
    ```

*   **`list`:** List saved channels, videos, and transcripts.
    *   Options: `-t`, `-c`, `-l`.

    ```bash
    yt-fts list --channel "3Blue1Brown"
    ```

*   **`update`:** Update subtitles for saved channels.
    *   Options: `-c`, `-l`, `-j`, `--cookies-from-browser`.

    ```bash
    yt-fts update --channel "3Blue1Brown" --jobs 5
    ```

*   **`delete`:** Delete a channel and its data.
    *   Options: `-c`.

    ```bash
    yt-fts delete --channel "3Blue1Brown"
    ```

*   **`export`:** Export transcripts for a channel.
    *   Options: `-c`, `-f`.

    ```bash
    yt-fts export --channel "3Blue1Brown" --format vtt
    ```

*   **`search`:** Perform full-text search.
    *   Options: `-c`, `-v`, `-l`, `-e`.

    ```bash
    yt-fts search "[search query]" --channel "[channel name or id]"
    ```

*   **`embeddings`:** Generate embeddings for semantic search (requires OpenAI or Gemini API key).
    *   Options: `-c`, `--api-key`, `-i`.

    ```bash
    yt-fts embeddings --channel "3Blue1Brown"
    ```

*   **`vsearch`:** Perform semantic search (requires enabled embeddings).
    *   Options: `-c`, `-v`, `-l`, `-e`.

    ```bash
    yt-fts vsearch "[search query]" --channel "[channel name or id]"
    ```

*   **`llm`:** Start an interactive chat session with the semantic search results.
    *   Options: `-c`, `--api-key`.

    ```bash
    yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
    ```

*   **`summarize`:** Summarize YouTube video transcripts.
    *   Options: `--model`, `--api-key`.

    ```bash
    yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
    ```

*   **`config`:** Show config settings.

    ```bash
    yt-fts config
    ```

**Advanced Search Syntax:**
The search string supports sqlite [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries).
which includes things like [prefix queries](https://www.sqlite.org/fts3.html#termprefix) which you can use to match parts of a word.

```bash
# AND search
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show" 

# OR SEARCH 
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show" 

# wild cards
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show" 
```

**How To:**

*   **Export search results:** Export search results with the `--export` flag.
    ```bash
    yt-fts search "life in the big city" --export
    yt-fts vsearch "existing in large metropolaten center" --export
    ```

*   **Delete a channel:** Delete a channel with the `delete` command.
    ```bash
    yt-fts delete --channel "3Blue1Brown"
    ```

*   **Update a channel:** The update command currently only works for full text search and will not update the semantic search embeddings.
    ```bash
    yt-fts update --channel "3Blue1Brown"
    ```

*   **Export all of a channel's transcript:**
    ```bash
    # Export to vtt
    yt-fts export --channel "[id/name]" --format "[vtt/txt]"
    ```

**Contribute:**

Feel free to contribute by opening issues or pull requests at the [yt-fts GitHub repository](https://github.com/NotJoeMartinez/yt-fts).