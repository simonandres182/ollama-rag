from langchain.llms import Ollama

# Especifica la URL del servidor remoto
ollama_url = "http://157.230.220.153:11434"

# Inicializa Ollama para que se conecte al servidor remoto
llm = Ollama(model="phi3:mini", base_url=ollama_url)
#llm = Ollama(model="tinyllama", base_url=ollama_url)

print("---------------------------------------------------------------")
print("--CHAT SIN RAG-- Escribe 'salir' para terminar la conversación.")
print("---------------------------------------------------------------")

# Bucle de chat
while True:
    # Entrada del usuario
    input_text = input("Tú: ")

    # Salir del chat si el usuario escribe "salir"
    if input_text.lower() in ["salir", "exit"]:
        print("Chat finalizado.")
        break

    # Genera la respuesta usando el modelo remoto
    res = llm.predict(input_text)

    # Imprime la respuesta
    print(f"Ollama: {res}")
