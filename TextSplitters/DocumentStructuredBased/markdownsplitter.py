from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

markdown_text = """
# Space Exploration

Space exploration represents humanity’s effort to understand the universe beyond Earth.
It has evolved from simple observations to advanced missions.

## Early Missions

The launch of the first artificial satellite marked a turning point.
Early missions focused on orbiting Earth and studying space conditions.

## Modern Space Programs

Modern missions explore the Moon, Mars, and beyond.
Private companies now play a major role in space exploration.

# Benefits of Space Technology

Satellites support communication, navigation, and weather forecasting.
These technologies have a direct impact on everyday life.
"""

# Markdown-aware splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=15,
    chunk_overlap=10
)

splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=15,
    chunk_overlap=10
)

# Split plain text
chunks = splitter.split_text(markdown_text)
new_chunks = splitter2.split_text(markdown_text)

print("1: ", chunks[0])
print("2: ", chunks[1])
print("3: ", new_chunks[0])
print("4: ", new_chunks[1])

