### Setup and Usage Guide

#### 1. Configure the API Key

Add your **Gemini API key** to the `.env` file in the project root directory.

---

#### 2. Create a Virtual Environment and Install Dependencies

Open **Command Prompt (CMD)** or a terminal in the project directory and run the following commands.

**Create a virtual environment:**

* **Linux**

```bash
python3 -m venv venv
```

* **Windows**

```bash
python -m venv venv
```

**Activate the virtual environment:**

* **Linux**

```bash
source venv/bin/activate
```

* **Windows**

```bash
venv\Scripts\activate
```

**Install required packages:**

```bash
pip install -r requirements.txt
```

---

#### 3. Run the AI Server

Start the Python server using the following command:

* **Linux**

```bash
python3 main.py
```

* **Windows**

```bash
python main.py
```

Once the server is running, open the following URL in your browser:

```
http://127.0.0.1:8000/static/index.html
```

---

#### 4. Adding Custom Data to the RAG Vector Database

To add your own data for the **RAG (Retrieval-Augmented Generation) vector database**:

1. Open the `ingest.py` file.
2. Edit the file and add the document(s) you want to include.
3. Run the ingestion script:

* **Linux**

```bash
python3 ingest.py
```

* **Windows**

```bash
python ingest.py
```

This will process the documents and store them in the vector database for retrieval during AI queries.

Presentation video url: 
https://youtu.be/iCNbXiDpQdk?si=_ugeGowHPHqXgWj0

Enjoy chating with the Ai.


...................................
Developer: 
Hrb Ranjan 
(Contact : WhatsApp +880 1979189844, E-mail: hranjanbappy@gmail.com)
