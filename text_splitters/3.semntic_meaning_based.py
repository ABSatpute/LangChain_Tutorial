from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

sample = """

🚗 Traffic in Lagos is one of the most pressing urban challenges. The city experiences frequent gridlocks that waste hours of productivity daily.

📊 Researchers have suggested several solutions. These include improving public transport, using AI-powered traffic management systems, and promoting carpooling.

💡 Among these, AI-powered systems show the most promise. By analyzing real-time data from cameras, sensors, and GPS, such systems can optimize traffic signals and reduce congestion.

🌍 Other cities like Singapore and London have already adopted smart traffic systems. Their success stories can provide valuable lessons for Lagos.

👥 However, implementation in Nigeria requires overcoming challenges. These include infrastructure limitations, high costs, and the need for strong government support."""

chunks = text_splitter.split_text(sample)
print(len(chunks))
print(chunks)