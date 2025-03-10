import os
from flask import Flask, jsonify
import asyncio
from dotenv import load_dotenv
from workflow import MyWorkflow
from flask import send_from_directory

# Define the base path for static files
BASE_STATIC_PATH = '/home/ahmed/learn/agentic/apps/static'

app = Flask(__name__)
load_dotenv()

async def run_workflow():
    workflow = MyWorkflow(timeout=10, verbose=False)
    result = await workflow.run()
    return result

@app.route('/')
def index():
    return send_from_directory(BASE_STATIC_PATH, 'index.html')

@app.route('/run_workflow', methods=['GET'])
def workflow_api():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(run_workflow())
    return jsonify({'result': result})

@app.route('/visualize_workflow')
def visualize_workflow():
    workflow = MyWorkflow()
    workflow.visualize(os.path.join(BASE_STATIC_PATH, 'workflow_visual.html'))
    return send_from_directory(BASE_STATIC_PATH, 'workflow_visual.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)