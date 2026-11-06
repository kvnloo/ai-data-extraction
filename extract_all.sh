#!/bin/bash
# Extract from ALL AI coding assistants at once

echo "================================================================================"
echo "AI CODING ASSISTANT DATA EXTRACTION - ALL TOOLS"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p extracted_data

# Track what we found
declare -a found_tools=()
declare -a not_found=()

# Extract from each tool
echo "🔍 Extracting from Claude Code..."
if python3 extract_claude_code.py 2>&1 | tee extracted_data/claude_extraction.log | grep -q "Total conversations: [1-9]"; then
    found_tools+=("Claude Code")
else
    not_found+=("Claude Code")
fi
echo ""

echo "🔍 Extracting from Cursor..."
if python3 extract_cursor.py 2>&1 | tee extracted_data/cursor_extraction.log | grep -q "Total conversations: [1-9]"; then
    found_tools+=("Cursor")
else
    not_found+=("Cursor")
fi
echo ""

echo "🔍 Extracting from Cursor CLI (cursor-agent)..."
if python3 extract_cursor_cli.py 2>&1 | tee extracted_data/cursor_cli_extraction.log | grep -q "Total conversations: [1-9]"; then
    found_tools+=("Cursor CLI")
else
    not_found+=("Cursor CLI")
fi
echo ""

echo "🔍 Extracting from Codex..."
if python3 extract_codex.py 2>&1 | tee extracted_data/codex_extraction.log | grep -q "Total conversations: [1-9]"; then
    found_tools+=("Codex")
else
    not_found+=("Codex")
fi
echo ""

echo "🔍 Extracting from Trae..."
if python3 extract_trae.py 2>&1 | tee extracted_data/trae_extraction.log | grep -q "Total conversations: [1-9]"; then
    found_tools+=("Trae")
else
    not_found+=("Trae")
fi
echo ""

echo "🔍 Extracting from Windsurf..."
if python3 extract_windsurf.py 2>&1 | tee extracted_data/windsurf_extraction.log | grep -q "Total conversations: [1-9]"; then
    found_tools+=("Windsurf")
else
    not_found+=("Windsurf")
fi
echo ""

echo "🔍 Extracting from Continue..."
if python3 extract_continue.py 2>&1 | tee extracted_data/continue_extraction.log | grep -q "Found [1-9]"; then
    found_tools+=("Continue")
else
    not_found+=("Continue")
fi
echo ""

echo "🔍 Extracting from Gemini CLI..."
if python3 extract_gemini.py 2>&1 | tee extracted_data/gemini_extraction.log | grep -q "Total conversations: [1-9]"; then
    found_tools+=("Gemini CLI")
else
    not_found+=("Gemini CLI")
fi
echo ""

echo "🔍 Extracting from OpenCode..."
if python3 extract_opencode.py 2>&1 | tee extracted_data/opencode_extraction.log | grep -q "Total conversations extracted: [1-9]"; then
    found_tools+=("OpenCode")
else
    not_found+=("OpenCode")
fi
echo ""

# Pi and OMP share an ancestry-aware parser; Droid has its own session format.
for tool in pi omp droid; do
    echo "🔍 Extracting from $tool..."
    if python3 "extract_$tool.py" 2>&1 | tee "extracted_data/${tool}_extraction.log" | grep -q "Total conversations: [1-9]"; then
        found_tools+=("$tool")
    else
        not_found+=("$tool")
    fi
    echo ""
done

# Hermes reads the default database and all profile databases without modifying them.
echo "🔍 Extracting from Hermes..."
if (set -o pipefail; python3 extract_hermes.py 2>&1 | tee extracted_data/hermes_extraction.log); then
    if grep -q "Total conversations: [1-9]" extracted_data/hermes_extraction.log; then
        found_tools+=("Hermes")
    else
        not_found+=("Hermes")
    fi
else
    echo "❌ Hermes extraction failed; see extracted_data/hermes_extraction.log" >&2
    exit 1
fi
echo ""

# Snapshot the input list before creating the combined file. Previous combined
# outputs must not be folded into the corpus again on subsequent runs.
shopt -s nullglob
conversation_files=()
for conversation_file in extracted_data/*.jsonl; do
    case "${conversation_file##*/}" in
        ALL_CONVERSATIONS_*) continue ;;
    esac
    conversation_files+=("$conversation_file")
done
combined_file="extracted_data/ALL_CONVERSATIONS_$(date +%Y%m%d_%H%M%S).jsonl"
if ! python3 combine_conversations.py --output "$combined_file" "${conversation_files[@]}"; then
    echo "❌ Conversation aggregation failed." >&2
    exit 1
fi

# Repository paths are explicit arguments; no repositories are auto-discovered.
# Git side tables live in a subdirectory and are not conversation records.
if [ "$#" -gt 0 ]; then
    echo "🔍 Extracting from Git..."
    if (set -o pipefail; python3 extract_git.py "$@" --sessions "$combined_file" 2>&1 | tee extracted_data/git_extraction.log); then
        if grep -q "Total commits: [1-9]" extracted_data/git_extraction.log; then
            found_tools+=("Git")
        else
            not_found+=("Git")
        fi
    else
        echo "❌ Git extraction failed; see extracted_data/git_extraction.log" >&2
        exit 1
    fi
else
    echo "Git extraction skipped: pass repository paths to extract_all.sh."
fi
echo ""

echo "================================================================================"
echo "EXTRACTION SUMMARY"
echo "================================================================================"
echo ""

if [ ${#found_tools[@]} -gt 0 ]; then
    echo "✅ Successfully extracted from:"
    for tool in "${found_tools[@]}"; do
        echo "   - $tool"
    done
    echo ""
fi

if [ ${#not_found[@]} -gt 0 ]; then
    echo "⚠️  No data found for:"
    for tool in "${not_found[@]}"; do
        echo "   - $tool"
    done
    echo ""
fi

# Count total conversations
total_lines=$(wc -l < "$combined_file" | tr -d ' ')

echo "📊 Total conversations extracted: $total_lines"
echo ""

# Show file sizes
echo "📁 Output files:"
if [ ${#conversation_files[@]} -gt 0 ]; then
    ls -lh "${conversation_files[@]}" | awk '{print "   " $9 " (" $5 ")"}'
fi
echo ""

if [ -f "$combined_file" ]; then
    combined_size=$(ls -lh "$combined_file" | awk '{print $5}')
    echo "✅ Combined file created:"
    echo "   $combined_file ($combined_size)"
    echo ""
fi

echo "================================================================================"
echo "COMPLETE!"
echo "================================================================================"
