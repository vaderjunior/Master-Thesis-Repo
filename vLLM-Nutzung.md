# 🤖 API access for research projects (Python)

To conserve resources and make optimal use of GPU memory, the AI models are provided centrally by the admin team on dedicated graphics cards. You don't need to start your own Docker containers or vLLM instances!

Our **Open-WebUI** functions not only as a web interface but also as a **central API gateway** for all Python scripts and data analysis pipelines. The interface behaves exactly like the official OpenAI API.

---

### 1. Generate a personal API key
Since authentication runs via your LDAP account, every user needs their own personal API key.

1. Log in to **Open-WebUI** (`https://openwebui.srv.peasec.de`).
2. Click your profile picture at the bottom left -> **Settings**.
3. Go to the **Account** tab.
4. Under *API Keys*, click the **gear icon** to generate a new key (starts with `sk-...`).
5. Copy this key. **Warning:** treat the key like a password and never commit it to public Git repositories!

---

### 2. Available models & infrastructure

Our infrastructure is designed for maximum performance and is split across two high-end graphics cards. Use the exact model ID depending on your research task:

| Model ID | API endpoint / routing link | Authentication | Optimised for... |
| :--- | :--- | :--- | :--- |
| **`Qwen/Qwen3-VL-32B-Instruct`** | `https://llm-reasoning.srv.peasec.de` | Bearer token / OpenWebUI key | **Complex reasoning & multimodality:** complex logic tasks, analysis of images/documents, and long contexts (up to 64K). |
| **`Qwen/Qwen3-VL-8B-Instruct`** | `https://llm-extractor.srv.peasec.de` | Bearer token / OpenWebUI key | **Information extraction & batching:** fast, structured data extraction from large volumes of text, and parallel workloads. |
| **`BAAI/bge-m3`** | `https://llm-embedder.srv.peasec.de` | Bearer token | **Text embeddings:** mathematical vectorisation of texts and graph links (via the `/embeddings` endpoint). |
| **`BAAI/bge-reranker-v2-m3`** | `https://llm-reranker.srv.peasec.de` | Bearer token | **Reranking:** mathematical filtering and sorting of relevant answers (via the `/score` endpoint). |

🔒 System API key for internal scripts: > For direct communication with the vLLM subdomains (outside the Open WebUI gateway), the shared infrastructure key tjsd8z9f78e8vc9fb8efxdv79rg7tju8o9p9 can be used as a bearer token if needed.


> 💡 **Note on the reranker:** the model `BAAI/bge-reranker-v2-m3` is permanently integrated into the WebUI's document pipeline (RAG). When you upload documents via the workspace, this model filters and sorts your search results fully automatically in the background.

---

### 3. Python integration (single-turn example)
You don't need to learn any new frameworks. Just use the official `openai` package for Python.

* **Important:** the correct base URL for API calls to our system is `https://openwebui.srv.peasec.de/api/v1`.

```python
# Installation: pip install openai
from openai import OpenAI

# 1. Initialise the client and redirect it to the PEASEC gateway
client = OpenAI(
    base_url="[https://openwebui.srv.peasec.de/api/v1](https://openwebui.srv.peasec.de/api/v1)", 
    api_key="sk-your-personal-key"
)

# 2. Send a request to the 32B reasoning model
try:
    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-32B-Instruct", 
        messages=[
            {"role": "system", "content": "You are a precise research assistant."},
            {"role": "user", "content": "Summarise the theory of cognitive dissonance in 2 sentences."}
        ],
        temperature=0.3
    )
    
    print("Model response:")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"An error occurred: {e}")

```
### 3.1. Contextual chats (multi-turn history)

The API is **stateless** by design. The model forgets the context immediately after each request and, on a new API call, no longer knows what was discussed in the previous step.

If your script is meant to simulate a continuous chat (e.g. a discussion or a series of follow-up questions about data), you have to manage the history yourself in code, store the AI's responses in it, and send the entire array along at every step:

```python
# Example of a multi-step chat history
chat_history = [
    {"role": "system", "content": "You are a precise data analyst."}
]

# Step 1: append the first question and send it to the gateway
chat_history.append({"role": "user", "content": "Which programming language is best for data science?"})

res1 = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=chat_history
)
ans1 = res1.choices[0].message.content
print(f"Answer 1: {ans1}\n")

# CRITICAL STEP: add the model's response to the history!
chat_history.append({"role": "assistant", "content": ans1})

# Step 2: ask a follow-up question (the model now knows what the previous step was about)
chat_history.append({"role": "user", "content": "Name the 3 most important libraries for it."})

res2 = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=chat_history
)
print(f"Answer 2: {res2.choices[0].message.content}")
```

### Benefits of this workflow for your research

* **100% OpenAI compatibility:** your scripts use the official, standardised OpenAI library. Code you write here will also work without changes against other OpenAI-conformant backends.
* **Intelligent load balancing:** the vLLM engine handles *continuous batching* and multiprocessing in the background fully automatically via Nvidia MPS (Multi-Process Service). Several researchers can send massive workloads to the same GPU simultaneously without blocking each other.
* **No loading times (zero delay):** the minutes-long loading of model weights at script start is eliminated entirely, since the models remain permanently in the graphics cards' VRAM. You get token responses within milliseconds.
* **Transparency & data security:** Open-WebUI handles secure API key management via your university LDAP account as a gateway. All processed research data remains local on the PEASEC infrastructure and never travels to external servers at any point.



# 🤖 API-Zugriff für Forschungsprojekte (Python)

Um Ressourcen zu schonen und den GPU-Speicher optimal zu nutzen, werden die KI-Modelle zentral vom Admin-Team auf dedizierten Grafikkarten bereitgestellt. Ihr müsst keine eigenen Docker-Container oder vLLM-Instanzen starten! 

Unsere **Open-WebUI** fungiert nicht nur als Weboberfläche, sondern auch als **zentrales API-Gateway** für alle Python-Skripte und Datenanalyse-Pipelines. Die Schnittstelle verhält sich exakt so wie die offizielle OpenAI-API.

---

### 1. Persönlichen API-Key generieren
Da die Authentifizierung über euren LDAP-Account läuft, benötigt jeder Nutzer seinen eigenen, persönlichen API-Key.

1. Loggt euch in **Open-WebUI** ein (`https://openwebui.srv.peasec.de`).
2. Klickt unten links auf euer Profilbild -> **Einstellungen (Settings)**.
3. Geht auf den Reiter **Konto (Account)**.
4. Klickt unter *API-Keys* auf das **Zahnrad-Symbol**, um einen neuen Schlüssel zu generieren (beginnt mit `sk-...`).
5. Kopiert diesen Schlüssel. **Achtung:** Behandelt den Key wie ein Passwort und checkt ihn niemals in öffentliche Git-Repositories ein!

---

### 2. Verfügbare Modelle & Infrastruktur

Unsere Infrastruktur ist auf maximale Performance ausgelegt und auf zwei High-End-Grafikkarten aufgeteilt. Nutzt je nach eurer Forschungsaufgabe die exakte Modell-ID:

| Modell-ID | API-Endpunkt / Routing-Link | Authentifizierung | Optimiert für... |
| :--- | :--- | :--- | :--- |
| **`Qwen/Qwen3-VL-32B-Instruct`** | `https://llm-reasoning.srv.peasec.de` | Bearer Token / OpenWebUI-Key | **Complex Reasoning & Multimodalität:** Komplexe Logikaufgaben, Analyse von Bildern/Dokumenten und lange Kontexte (bis 64K). |
| **`Qwen/Qwen3-VL-8B-Instruct`** | `https://llm-extractor.srv.peasec.de` | Bearer Token / OpenWebUI-Key | **Information Extraction & Batching:** Schnelle, strukturierte Datenextraktion aus großen Textmengen und parallele Workloads. |
| **`BAAI/bge-m3`** | `https://llm-embedder.srv.peasec.de` | Bearer Token | **Text-Embeddings:** Mathematische Vektorisierung von Texten und Graph-Links (über den `/embeddings`-Endpunkt). |
| **`BAAI/bge-reranker-v2-m3`** | `https://llm-reranker.srv.peasec.de` | Bearer Token | **Reranking:** Mathematisches Filtern und Sortieren von relevanten Antworten (über den `/score`-Endpunkt). |

🔒 System-API-Key für interne Skripte: > Für die direkte Kommunikation mit den vLLM-Subdomains (außerhalb des Open WebUI Gateways) kann bei Bedarf der gemeinsame Infrastruktur-Schlüssel tjsd8z9f78e8vc9fb8efxdv79rg7tju8o9p9 als Bearer-Token genutzt werden.


> 💡 **Hinweis zum Reranker:** Das Modell `BAAI/bge-reranker-v2-m3` ist fest in die Dokumenten-Pipeline (RAG) der WebUI integriert. Wenn ihr Dokumente über den Arbeitsbereich hochladet, filtert und sortiert dieses Modell eure Suchergebnisse vollautomatisch im Hintergrund.

---

### 3. Python Integration (Single-Turn-Beispiel)
Ihr müsst keine neuen Frameworks lernen. Nutzt einfach das offizielle `openai` Paket für Python. 

* **Wichtig:** Die korrekte Base-URL für API-Calls an unser System lautet `https://openwebui.srv.peasec.de/api/v1`.

```python
# Installation: pip install openai
from openai import OpenAI

# 1. Client initialisieren und auf das PEASEC-Gateway umleiten
client = OpenAI(
    base_url="[https://openwebui.srv.peasec.de/api/v1](https://openwebui.srv.peasec.de/api/v1)", 
    api_key="sk-euer-persoenlicher-schluessel"
)

# 2. Anfrage an das 32B-Reasoning-Modell stellen
try:
    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-32B-Instruct", 
        messages=[
            {"role": "system", "content": "Du bist ein präzise antwortender Forschungsassistent."},
            {"role": "user", "content": "Fasse die Theorie der kognitiven Dissonanz in 2 Sätzen zusammen."}
        ],
        temperature=0.3
    )
    
    print("Modell-Antwort:")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")

```
### 3.1. Kontextbezogene Chats (Multi-Turn Historie)

Die API ist von Grund auf **zustandslos (stateless)**. Das Modell vergisst nach jeder Anfrage sofort den Kontext und weiß bei einem neuen API-Aufruf nicht mehr, was im vorherigen Schritt besprochen wurde. 

Wenn euer Skript einen fortlaufenden Chat simulieren soll (z. B. eine Diskussion oder eine Reihe von Folgefragen zu Daten), müsst ihr die Historie im Code selbst verwalten, die Antworten der KI darin abspeichern und das gesamte Array bei jedem Schritt mitsenden:

```python
# Beispiel für einen mehrstufigen Chat-Verlauf
chat_history = [
    {"role": "system", "content": "Du bist ein präzise antwortender Datenanalyst."}
]

# Schritt 1: Erste Frage anhängen und an das Gateway senden
chat_history.append({"role": "user", "content": "Welche Programmiersprache ist für Data Science am besten?"})

res1 = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=chat_history
)
ans1 = res1.choices[0].message.content
print(f"Antwort 1: {ans1}\n")

# CRITICAL STEP: Antwort des Modells in die Historie aufnehmen!
chat_history.append({"role": "assistant", "content": ans1})

# Schritt 2: Rückfrage stellen (Das Modell weiß jetzt, worum es im Schritt zuvor ging)
chat_history.append({"role": "user", "content": "Nenne mir dafür die 3 wichtigsten Bibliotheken."})

res2 = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=chat_history
)
print(f"Antwort 2: {res2.choices[0].message.content}")
```

### Vorteile dieses Workflows für eure Forschung

* **100% OpenAI-Kompatibilität:** Eure Skripte nutzen die offizielle und standardisierte OpenAI-Bibliothek. Code, den ihr hier schreibt, funktioniert ohne Änderungen auch mit anderen OpenAI-konformen Backends.
* **Intelligentes Load-Balancing:** Die vLLM-Engine regelt das *Continuous Batching* und Multi-Processing im Hintergrund über Nvidia MPS (Multi-Process Service) vollautomatisch. Mehrere Forscher können gleichzeitig massive Workloads an dieselbe GPU schicken, ohne sich gegenseitig zu blockieren.
* **Keine Ladezeiten (Zero-Delay):** Das minutenlange Laden von Modellgewichten beim Skriptstart entfällt komplett, da die Modelle dauerhaft im VRAM der Grafikkarten verbleiben. Ihr erhaltet Token-Antworten direkt in Millisekunden.
* **Transparenz & Datensicherheit:** Open-WebUI übernimmt als Gateway das sichere API-Key-Management über euren universitären LDAP-Zugang. Alle verarbeiteten Forschungsdaten verbleiben lokal auf der PEASEC-Infrastruktur und wandern zu keinem Zeitpunkt auf externe Server.