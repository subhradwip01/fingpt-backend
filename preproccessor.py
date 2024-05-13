from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize Ollama and other necessary components
embeddings_open = OllamaEmbeddings(model="mistral")
llm_open = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
loader = DirectoryLoader('./data/langchain_doc_small', glob="./*.txt")
doc = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(doc)

# Create Chroma vector database
persist_directory = 'vdb_langchain_doc_small'
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings_open, persist_directory=persist_directory)
vectordb.persist()
vectordb = None
print("Documents loaded and processed successfully.")

