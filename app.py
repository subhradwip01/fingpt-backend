from flask import Flask
from flask_cors import CORS
from flask import Blueprint, jsonify, request
from flask_mongoengine import MongoEngine
from flask_bcrypt import Bcrypt
import jwt
from functools import wraps
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os 

# app configuration
app = Flask(__name__)
bcrypt = Bcrypt(app)
CORS(app)
app.config['MONGODB_SETTINGS'] = {
    'host': 'mongodb+srv://kulavisubhradwip15:fUxVMhdOUBnza0OZ@cluster0.vv69eor.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
}
db = MongoEngine(app)
app.config['SECRET_KEY'] = 'secret'


# -----USER-----
class User(db.Document):
    name = db.StringField(required=True)
    email = db.EmailField(required=True, unique=True)
    password = db.StringField(required=True)

    meta = {'collection': 'users'} 
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    print(name)
    print(email)
    if not name or not email or not password:
        return jsonify({'error': 'Please provide name, email, and password'}), 400

    # Check if the user already exists
    if User.objects(email=email):
        return jsonify({'error': 'User already exists'}), 409

    # Hash the password
    hashed_password = bcrypt.generate_password_hash(password)
    # Create a new user
    user = User(name=name, email=email, password=hashed_password)
    user.save()
    token = jwt.encode({'email': email}, app.config['SECRET_KEY'], algorithm='HS256')
    return jsonify({'message': 'User signed up successfully', 'token': token,'user':user}), 200

@app.route('/signin', methods=['POST'])
def signin():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Please provide email and password'}), 400

    # Find the user by email
    user = User.objects(email=email).first()

    if not user:
        return jsonify({'error': 'User not found'}), 404

    # Check if password matches
    if not bcrypt.check_password_hash(user.password,password):
        return jsonify({'error': 'Invalid password'}), 401

    # Generate JWT token
    token = jwt.encode({'email': email}, app.config['SECRET_KEY'], algorithm='HS256')

    # User authenticated
    return jsonify({'message': 'User signed in successfully', 'token': token,'user':user}), 200

def token_required(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'error': 'Token is missing'}), 401

        try:
            # Extract the token from the Authorization header
            token = token.split()[1]  # Assuming format is "Bearer <token>"
            # Verify and decode the token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        # Attach the decoded token data to the request object
        request.user = data
        return func(*args, **kwargs)

    return decorated

# -----------Docs----------------------

@app.route('/loadDocs', methods=['POST'])
@token_required
def loadDocs():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('file')

    # If user does not select any files
    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400

    # Ensure the 'data' directory exists
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save each file to the 'data' directory
    for file in files:
        if file.filename == '':
            return jsonify({'error': 'File has no filename'}), 400
        file.save(os.path.join(data_dir, file.filename))
    
    embeddings_open = OllamaEmbeddings(model="mistral")
    llm_open = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))
    doc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(doc)

    # Create Chroma vector database
    persist_directory = 'vdb_langchain_doc_small'
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings_open, persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None
    print("Documents loaded and processed successfully.")
    
    return jsonify({'message': 'file(s) uploaded successfully'}), 200

@app.route('/query', methods=['POST'])
@token_required
def resolveQuery():
    data = request.json
    query = data.get('query')
    embeddings_open = OllamaEmbeddings(model="mistral")
    llm_open = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    persist_directory = 'vdb_langchain_doc_small'
    vectordb = Chroma(persist_directory=persist_directory,embedding_function=embeddings_open)
    retriever = vectordb.as_retriever()

    def build_prompt(template_num="template_1"):
        template = """ You are a helpful chatbot, named RSLT. You answer the questions of the customers giving a lot of details based on what you find in the context.
        Do not say anything that is not in the website
        You are to act as though you're having a conversation with a human.
        You are only able to answer questions, guide and assist, and provide recommendations to users. You cannot perform any other tasks outside of this.
        Your tone should be professional and friendly.
        Your purpose is to answer questions people might have, however if the question is unethical you can choose not to answer it.
        Your responses should always be one paragraph long or less.
        Context: {context}
        Question: {question}
        Helpful Answer:"""

        template2 = """You are a helpful chatbot, named RSLT. You answer the questions of the customers giving a lot of details based on what you find in the context. 
        Your responses should always be one paragraph long or less.
        Question: {question}
        Helpful Answer:"""

        if template_num == "template_1":
            prompt = PromptTemplate(input_variables=["context", "question"], template=template)
            return prompt

        elif template_num == "template_2":
            prompt = PromptTemplate(input_variables=["question"], template=template2)
            return prompt

        else:
            print("Please choose a valid template")


    qa_chain = RetrievalQA.from_chain_type(llm=llm_open,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    return_source_documents=True,
                                    verbose=True,
                                    chain_type_kwargs={"prompt": build_prompt("template_1")})
    llm_response = qa_chain(query)
    def get_sources(llm_response):
        sources = [{'source': source.metadata['source']} for source in llm_response["source_documents"]]
        return sources;
    return jsonify({'message':llm_response['result'],'sources':get_sources(llm_response)}), 200



if __name__ == "__main__":
    app.run(debug=True, port=5050)
