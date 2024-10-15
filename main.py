from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import logging

nltk.download('punkt')  # Download the punkt tokenizer if not already done

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve templates
templates = Jinja2Templates(directory="templates")

# Load pre-trained models
model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
stored_texts = []  # Initialize the list to store loaded texts

# Load the Llama2 model for text generation
llama2_generator = pipeline("text-generation", model="gpt2-medium", device=0)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/load")
async def load_data(request: Request):
    try:
        body = await request.json()  # Get the JSON body from the request
        url = body.get("url")  # Extract the URL safely using .get()

        if url is None:
            raise HTTPException(status_code=400, detail="URL must be provided")

        # Make the request to the URL
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch the URL")

        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()

        # Use NLTK's sentence tokenizer for better splitting
        sentences = sent_tokenize(text)
        embeddings = model.encode(sentences)

        # Add embeddings to the FAISS index
        faiss_index.add(embeddings)

        # Store the original sentences in the stored_texts list
        stored_texts.extend(sentences)

        logger.info(f"Loaded {len(sentences)} sentences from the URL.")
        return {"message": "Data loaded successfully"}
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_data(request: Request):
    try:
        body = await request.json()  # Get the JSON body from the request
        query = body.get("query")  # Extract the query safely using .get()

        if query is None:
            raise HTTPException(status_code=400, detail="Query must be provided")

        # Encode the query to get its embedding
        query_embedding = model.encode([query])
        
        # Perform the search in the FAISS index
        distances, indices = faiss_index.search(query_embedding, k=10)  # Retrieve up to 10 results

        # Ensure indices are valid
        valid_indices = [i for i in indices[0] if i < len(stored_texts)]
        if not valid_indices:
            raise HTTPException(status_code=404, detail="No relevant texts found.")

        # Retrieve relevant texts based on valid indices
        relevant_texts = [stored_texts[i] for i in valid_indices]

        # Increase context length for better answers
        context = " ".join(relevant_texts[-5:])  # Use the last 5 relevant texts for context

        # Refine the prompt for clarity and to avoid repetition
        prompt = (
            f"Based on the context provided below, please explain clearly and concisely what natural language processing is. "
            f"Avoid repetition and ensure the answer is informative and coherent.\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        # Generate an answer based on the context
        answer = llama2_generator(prompt, max_new_tokens=350, num_return_sequences=1, do_sample=False)

        # Process the generated answer to only return the relevant part
        generated_answer = answer[0]["generated_text"].split("Answer:")[-1].strip()

        # Remove any repetitive phrases from the generated answer
        # This simple check is just an example; you may want to implement more complex logic.
        unique_words = set()
        cleaned_answer = []
        for word in generated_answer.split():
            if word not in unique_words:
                unique_words.add(word)
                cleaned_answer.append(word)

        cleaned_answer_text = ' '.join(cleaned_answer).strip()

        # Log the cleaned generated answer
        logger.info(f"Generated answer: {cleaned_answer_text}")

        # Return the cleaned and complete answer
        return {"answer": cleaned_answer_text}
        
    except Exception as e:
        logger.error(f"Error querying data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_data():
    global stored_texts
    stored_texts = []  # Reset the stored texts
    faiss_index.reset()  # Clear the FAISS index
    logger.info("Data cleared successfully.")
    return {"message": "Data cleared successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
