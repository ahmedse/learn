import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import asyncio
from dotenv import load_dotenv
from workflow import RAGWorkflow, QueryEvent
import nest_asyncio

nest_asyncio.apply()

# Initialize the Flask application
app = Flask(__name__, static_folder='/home/ahmed/learn/agenticpdf/apps/static/', template_folder='../templates')
load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the workflow
workflow = RAGWorkflow()
chat_session_active = False
workflow_context = {}


@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handle file upload and trigger the workflow setup."""
    global workflow_context, chat_session_active

    print("UPLOAD: Upload endpoint triggered.")  # Log when the endpoint is triggered

    # Check if a file is in the request
    if 'file' not in request.files:
        print("UPLOAD: No file part in the request.")  # Log missing file part
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        print("UPLOAD: No selected file.")  # Log empty filename
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    print(f"UPLOAD: Saving file to {filepath}")  # Log file save path

    try:
        file.save(filepath)
        print(f"UPLOAD: File saved successfully to {filepath}")  # Log successful save
    except Exception as e:
        print(f"UPLOAD: Failed to save file: {str(e)}")  # Log file save error
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

    # Trigger the workflow setup step
    try:
        print("UPLOAD: Starting workflow setup.")  # Log workflow setup start
        asyncio.run(setup_workflow(filepath))
        chat_session_active = True
        print("UPLOAD: Workflow setup completed successfully.")  # Log workflow setup success
        return jsonify({
            'message': 'Workflow setup completed successfully. You can now chat with the document.',
            'filepath': filepath
        })
    except Exception as e:
        print(f"UPLOAD: Workflow setup failed: {str(e)}")  # Log workflow setup failure
        return jsonify({'error': f'Workflow setup failed: {str(e)}'}), 500


async def setup_workflow(filepath):
    """Set up the workflow with the uploaded file."""
    global workflow_context

    print(f"WORKFLOW: Starting setup for file {filepath}")  # Log workflow setup start

    try:
        # Create the StartEvent for the setup step
        print(f"WORKFLOW: Creating StartEvent for file {filepath}")  # Log StartEvent creation
        start_event = workflow.create_start_event(resume_file=filepath)

        # Trigger the workflow's set_up step
        print(f"WORKFLOW: Triggering set_up step with file {filepath}")  # Log set_up step
        await workflow.set_up(ctx={}, ev=start_event)
        print("WORKFLOW: set_up step completed successfully.")  # Log set_up success

        # Initialize the workflow context
        workflow_context = {"current_iteration": 0}
        print("WORKFLOW: Workflow context initialized.")  # Log context initialization
    except Exception as e:
        print(f"WORKFLOW: Workflow set_up step failed: {str(e)}")  # Log set_up failure
        raise


@app.route('/chat', methods=['POST'])
def chat():
    """Handle user chat queries."""
    global workflow_context, chat_session_active

    if not chat_session_active:
        print("CHAT: No active chat session.")  # Log inactive session
        return jsonify({'error': 'No active chat session. Please upload a file to start.'}), 400

    user_query = request.json.get('query')
    if not user_query:
        print("CHAT: No query provided.")  # Log missing query
        return jsonify({'error': 'No query provided'}), 400

    try:
        print(f"CHAT: Received user query: {user_query}")  # Log the user query

        # Trigger the ask_question step
        response = asyncio.run(handle_query(user_query))
        print(f"CHAT: Query response: {response}")  # Log the response

        return jsonify({'response': response})
    except Exception as e:
        print(f"CHAT: Error occurred: {str(e)}")  # Log query failure
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500


async def handle_query(user_query):
    """Handle a user query using the RAG workflow."""
    global workflow_context

    try:
        # Create a QueryEvent for the query
        print(f"WORKFLOW: Creating QueryEvent for query: {user_query}")  # Log QueryEvent creation
        query_event = QueryEvent(query=user_query)

        # Trigger the ask_question step
        print("WORKFLOW: Triggering ask_question step...")  # Log ask_question step
        response_event = await workflow.ask_question(ctx=workflow_context, ev=query_event)

        # Return the response
        print(f"WORKFLOW: Successfully handled query. Response: {response_event.result}")  # Log response
        return response_event.result
    except Exception as e:
        print(f"WORKFLOW: Error while handling query: {str(e)}")  # Log query error
        raise


@app.route('/end_chat', methods=['POST'])
def end_chat():
    """Manually stop the chat session."""
    global chat_session_active
    chat_session_active = False
    print("CHAT: Chat session ended by user.")  # Log session end
    return jsonify({'message': 'Chat session ended by user.'})


@app.route('/visualize_workflow')
def visualize_workflow():
    """Visualize the workflow."""
    workflow.visualize(os.path.join(app.static_folder, 'workflow_visual.html'))
    print("WORKFLOW: Workflow visualization generated.")  # Log visualization
    return send_from_directory(app.static_folder, 'workflow_visual.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)