from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader  # Cambiado a PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Especifica la URL del servidor remoto
ollama_url = "http://157.230.220.153:11434"

# Inicializa Ollama para que se conecte al servidor remoto
llm = Ollama(model="phi3:mini", base_url=ollama_url)
#llm = Ollama(model="tinyllama", base_url=ollama_url)

# Cargar el archivo PDF para el sistema de recuperación
loader = PyPDFLoader('/datos/Descargas/prueba/Ficha-Tecnica-NuevoNiro.pdf')  # Cambia a la ruta de tu PDF
documents = loader.load()

# Verificar si los documentos se han cargado correctamente
print(f"Documentos cargados: {documents}")

# Dividir los documentos en trozos más pequeños
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Verificar los fragmentos generados
print(f"Fragmentos generados: {texts}")

# Usar embeddings locales de Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Generar embeddings para los fragmentos
embeddings_result = embeddings.embed_documents([text.page_content for text in texts])

# Verificar los embeddings generados
print(f"Embeddings generados: {embeddings_result}")

# Crear el almacén de vectores utilizando FAISS
if len(embeddings_result) > 0:
    vectorstore = FAISS.from_documents(texts, embeddings)
    print("Almacén de vectores creado correctamente.")
else:
    print("Error: No se pudieron generar embeddings. Verifica los textos y los embeddings.")

# Configurar el sistema de QA con recuperación
qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Chat interactivo con RAG
print("------------------------------------------------------------------")
print("--CHAT USANDO RAG-- Escribe 'salir' para terminar la conversación.")
print("------------------------------------------------------------------")

while True:
    # Entrada del usuario
    input_text = input("Tú: ")

    # Salir del chat si el usuario escribe "salir"
    if input_text.lower() in ["salir", "exit"]:
        print("Chat finalizado.")
        break

    # Usar el sistema de QA con recuperación para responder
    response = qa_chain.run(input_text)

    # Imprimir la respuesta
    print(f"Ollama con RAG (local): {response}")
