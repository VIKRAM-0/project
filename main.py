from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from pymilvus import Index
import logging
from transformers import AutoTokenizer, pipeline

# Initialize the app
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve templates
templates = Jinja2Templates(directory="templates")

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the GPT-2 model for text generation
gpt2_generator = pipeline("text-generation", model="gpt2-medium", device=0)

# Tokenizer initialization (used instead of NLTK)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define the collection schema for storing vectors
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=model.get_sentence_embedding_dimension()),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
]

# Create the collection schema
schema = CollectionSchema(fields, description="Text embedding collection")

# Collection name
collection_name = "text_embedding_collection"

# Create or load the collection
try:
    collection = Collection(name=collection_name)
    logger.info(f"Collection '{collection_name}' already exists.")
except Exception:
    collection = Collection(name=collection_name, schema=schema)
    logger.info(f"Collection '{collection_name}' created.")

# Create an index on the embedding field
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
    "metric_type": "L2"
}
index = Index(collection, "embedding", index_params)
logger.info(f"Index created on collection '{collection_name}'.")

# Load the collection for querying and inserting data
collection.load()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/load")
async def load_data(request: Request):
    try:
        body = await request.json()
        url = body.get("url")

        if not url:
            raise HTTPException(status_code=400, detail="URL must be provided")

        # Fetch the content from the provided URL
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch the URL")

        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()

        # Tokenize the text using AutoTokenizer (alternative to NLTK)
        sentences = tokenizer.tokenize(text)
        logger.info(f"Sentences being inserted: {sentences[:5]}...")  # Log first few sentences

        # Convert the text into embeddings
        embeddings = model.encode(sentences)
        logger.info(f"Shape of sentence embeddings: {embeddings.shape}")  # Log embeddings shape

        # Batch the embeddings and sentences to avoid exceeding the message size limit
        batch_size = 1000  # Insert in batches of 1000 embeddings at a time
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i + batch_size].tolist()
            batch_sentences = sentences[i:i + batch_size]
            entities = [batch_embeddings, batch_sentences]

            collection.insert(entities)
            logger.info(f"Inserted batch {i // batch_size + 1} containing {len(batch_sentences)} sentences.")

        logger.info(f"Loaded {len(sentences)} sentences from the URL.")
        return {"message": "Data loaded successfully"}
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/query")
async def query_data(request: Request):
    try:
        body = await request.json()
        query = body.get("query")

        if not query:
            raise HTTPException(status_code=400, detail="Query must be provided")

        # Encode the query and search in Milvus
        query_embedding = model.encode([query])
        logger.info(f"Query embedding shape: {query_embedding.shape}")

        # Search for the top 10 closest embeddings
        search_params = {
            "metric_type": "L2",  # Euclidean distance
            "params": {"nprobe": 50}  # Increase nprobe for more clusters to be searched
        }
        results = collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=10,
            output_fields=["text"]
        )

        # If no results found
        if not results:
            logger.info("No search results found in Milvus.")
            raise HTTPException(status_code=404, detail="No relevant texts found")

        # Extract the most relevant sentences
        relevant_texts = [hit.entity.get("text") for hit in results[0]]
        if not relevant_texts:
            raise HTTPException(status_code=404, detail="No relevant texts found")

        # Prepare the context for text generation
        context = " ".join(relevant_texts[-5:])

        # Refine the prompt
        prompt = (
            f"Based on the context provided below, explain clearly and concisely the concept of NLP. "
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        # Generate the answer using GPT-2
        answer = gpt2_generator(prompt, max_new_tokens=350, num_return_sequences=1, do_sample=False)
        generated_answer = answer[0]["generated_text"].split("Answer:")[-1].strip()

        logger.info(f"Generated answer: {generated_answer}")
        return {"answer": generated_answer}
    
    except Exception as e:
        logger.error(f"Error querying data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_data():
    try:
        # Drop the collection to clear all data
        collection.drop()
        logger.info("Data cleared successfully.")
        return {"message": "Data cleared successfully."}
    except Exception as e:
        logger.error(f"Error clearing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
