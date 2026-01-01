from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text = """
Space exploration represents humanity’s effort to understand the universe beyond Earth. From early observations of the night sky to modern space missions, humans have always been curious about what lies beyond our planet. Advances in science and engineering have allowed us to send spacecraft, satellites, and humans into space, expanding our knowledge of the cosmos.

The launch of the first artificial satellite marked a turning point in space exploration. Since then, missions to the Moon, Mars, and beyond have provided valuable insights into planetary formation, climate, and the possibility of life outside Earth. Space telescopes have enabled scientists to observe distant galaxies and study phenomena that cannot be seen from the ground.

Space exploration also plays a practical role in everyday life. Satellites support communication, navigation, weather forecasting, and environmental monitoring. These technologies help improve disaster response, global connectivity, and scientific research, demonstrating that space exploration has direct benefits on Earth.

As space agencies and private companies continue to collaborate, the future of space exploration looks promising. Plans for lunar bases, crewed missions to Mars, and deeper exploration of the solar system reflect humanity’s ongoing desire to push boundaries and explore the unknown. With continued innovation and responsible exploration, space will remain a frontier of discovery and inspiration.
"""
loader = PyPDFLoader("../DocumentLoaders/Pothole_2023.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=5,
                                 chunk_overlap=0,
                                 separator="")

result1 = splitter.split_text(text)

result2 = splitter.split_documents(docs)

print("text: ", result1, end="\n\n")
print("documents: ", result2[0].page_content)
