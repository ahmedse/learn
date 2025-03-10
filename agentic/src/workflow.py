import os
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step
from llama_index.utils.workflow import draw_all_possible_flows

class MyWorkflow(Workflow):
    @step
    async def start_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="Hello, World!")

    def visualize(self, filename=""):
        base_static_path = '/home/ahmed/learn/agentic/apps/static'
        full_path = os.path.join(base_static_path, filename)
        draw_all_possible_flows(self, filename=full_path)