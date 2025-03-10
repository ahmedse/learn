import os
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Event, step, Context
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_parse import LlamaParse
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgent

class QueryEvent(Event):
    query: str

class ChatEvent(Event):
    continue_chat: bool

class RAGWorkflow(Workflow):
    storage_dir = "./storage"
    llm: OpenAI
    query_engine: VectorStoreIndex
    agent: FunctionCallingAgent

    @step
    async def set_up(self, ctx: Context, ev: StartEvent) -> ChatEvent:
        if not os.path.exists(ev.resume_file):
            raise ValueError("No resume file provided")
        
        self.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        documents = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY")).load_data(ev.resume_file)

        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=OpenAIEmbedding(model_name="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        )
        self.query_engine = index.as_query_engine(llm=self.llm, similarity_top_k=5)

        def query_resume(q: str) -> str:
            response = self.query_engine.query(q)
            return response.response
        
        resume_tool = FunctionTool.from_defaults(fn=query_resume)
        self.agent = FunctionCallingAgent.from_tools(tools=[resume_tool], llm=self.llm, verbose=True)

        return ChatEvent(continue_chat=True)

    @step
    async def chat_with_user(self, ctx: Context, ev: ChatEvent) -> StopEvent:
        if not ev.continue_chat:
            return StopEvent(result="Chat ended by user.")

        # Get user query
        user_query = ctx.get("user_query", "Could you please ask your question?")
        response = self.agent.chat(user_query)
        print(f"Agent Response: {response}")

        # Check if user wants to continue the chat
        continue_chat = ctx.get("continue_chat", True)
        if not continue_chat:
            return StopEvent(result="User ended the chat session.")

        # Add a condition to stop after a certain number of iterations or based on user input
        max_iterations = ctx.get("max_iterations", 5)
        current_iteration = ctx.get("current_iteration", 0) + 1

        if current_iteration >= max_iterations:
            return StopEvent(result="Maximum chat iterations reached.")
        
        # Update the context with the new iteration count
        ctx["current_iteration"] = current_iteration
        return ChatEvent(continue_chat=continue_chat)

    def visualize(self, filename=""):
        base_static_path = '/home/ahmed/learn/agentic/apps/static'
        full_path = os.path.join(base_static_path, filename)
        draw_all_possible_flows(self, filename=full_path)