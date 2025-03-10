import os
from flask import Flask, request, jsonify, send_from_directory
import asyncio
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.utils.workflow import draw_all_possible_flows
from llama_parse import LlamaParse
from workflow import RAGWorkflow

# Initialize the Flask application
app = Flask(__name__, static_folder='/home/ahmed/learn/agenticpdf/apps/static')
load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')
LLAMA_CLOUD_BASE_URL = os.getenv('LLAMA_CLOUD_BASE_URL')

# Global variable to store the index
global_index = None

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    file = request.files['file']
    filename = os.path.join('uploads', file.filename)
    file.save(filename)
    return jsonify({'message': 'File uploaded successfully', 'path': filename})

@app.route('/parse', methods=['POST'])
def parse_index():
    global global_index
    filepath = request.json['filepath']
    documents = LlamaParse(api_key=LLAMA_CLOUD_API_KEY).load_data(filepath)
    global_index = VectorStoreIndex.from_documents(
        documents,
        embed_model=OpenAIEmbedding(model_name="text-embedding-3-small", api_key=OPENAI_API_KEY)
    )
    return jsonify({'message': 'Resume parsed and indexed'})

@app.route('/chat', methods=['POST'])
def chat_with_resume():
    global global_index
    query = request.json['query']
    if global_index is None:
        return jsonify({'error': 'Resume not indexed. Please upload and parse a resume first.'}), 400

    # Create a query engine from the index
    llm = OpenAI(model="text-davinci-003", api_key=OPENAI_API_KEY)
    query_engine = global_index.as_query_engine(llm=llm, similarity_top_k=5)

    # Perform the query
    response = query_engine.query(query)
    return jsonify({'answer': response.response})

@app.route('/visualize_workflow')
def visualize_workflow():
    workflow = RAGWorkflow()
    workflow.visualize(os.path.join(app.static_folder, 'workflow_visual.html'))
    return send_from_directory(app.static_folder, 'workflow_visual.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)