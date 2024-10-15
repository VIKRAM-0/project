# Project Description
The main objective of this program is to develop a system which would fetch data from wikipedia and store the data into a vector database and when a query is given to it the data from the vector database is retrived and it is passed into a generative model for answering the query.

# Technologies used
>Fast API

>BeautifulSoup

>Transformer

>Gpt2-medium

>Uvicorn

## Fast API
Fast API is used for automating the process of information feteching and query processing. This allows use to develop an better and a faster application.
## BeautifulSoup 
BeautifulSoup is used for precessing the content which is feteched from the wikipedia website which would be in a raw `HTML` foramt. By using this we would be able to extract only the content which is present in the webpage and not the tags.
## Transformer
Transformer are used to convert the extracted text and convert it into embeddings which would be easier for storing the data. This also provides us with the creation of pipeline which would be help full for text generation. The pretrained model which is used for converting text into embeddings is `all-MiniLM-L6-v2`.
## GPT2-Medium
This is the generative model which is used for generating text based on the query and the context which would be extracted from the vector database. This process is done by passing the query and the related context as a prompt into the model which would allow the model to understand about the task and give correct answers.
## Uvicorn
Uvicorn is the server that allows your FastAPI application to run, receive HTTP requests, and deliver responses, making it a crucial component in deploying and running the app.

## Running Application
1. As the first step clone the respository.
2. Change the working directory to the cloned respository.
3. Now run `pip install requirements.txt` in the terminal.
4. After installation open the terminal of the editor and run the command `uvicorn main:app --reload`.
5. Now follow the link present in the terminal. It would be like this `Uvicorn running on http://127.0.0.1:8000`.
6. To exit the application go to the terminal and press `ctrl + c`.

## Interface of the website

![image](https://github.com/user-attachments/assets/fdb23074-d537-4c67-a610-a5652ff742a5)
