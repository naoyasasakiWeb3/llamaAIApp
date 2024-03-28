from pathlib import Path
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.readers.file import PDFReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from transformers import AutoTokenizer, AutoModel

Settings.llm = Ollama(model="llama2", request_timeout=60.0)

print("llama2 is ready to use!")

#response = Settings.llm.complete("what is tokyo vision for the 2040s")
#print("Response:", response)

# Create an instance of PDFReader
path = Path("data/versionup2023_March.pdf")
pdf_reader = PDFReader()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Use the instance to call load_data
documents = pdf_reader.load_data(file=path)

# documentsをstrに変換
documents = ' '.join([doc.text for doc in documents])
# Assuming `documents` is a single string
inputs = tokenizer([documents], return_tensors="pt", truncation=True, max_length=512)

# Or, if `documents` is already a list of strings
# inputs = tokenizer(documents, return_tensors="pt")

outputs = model(**inputs)
# outputsにはテキストのベクトル表現が含まれる

query_engine = outputs.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)