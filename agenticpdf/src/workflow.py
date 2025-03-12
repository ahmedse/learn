import os
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Event, step, Context
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_parse import LlamaParse
from llama_index.utils.workflow import draw_all_possible_flows


class QueryEvent(Event):
    query: str

class SetupCompleteEvent(Event):
    """Event emitted when setup is complete."""
    pass

class RAGWorkflow(Workflow):
    storage_dir = "./storage"
    llm: OpenAI
    query_engine: VectorStoreIndex

    def create_start_event(self, resume_file: str) -> StartEvent:
        """Helper method to create a StartEvent."""
        if not os.path.exists(resume_file):
            raise ValueError(f"The resume file does not exist: {resume_file}")
        print(f"WORKFLOW: Creating StartEvent for file {resume_file}")
        return StartEvent(resume_file=resume_file)

    @step
    async def set_up(self, ctx: Context, ev: StartEvent) -> SetupCompleteEvent:
        """First step: Set up the RAG workflow."""
        print(f"WORKFLOW: Setting up workflow for file {ev.resume_file}")

        # Initialize LLM
        self.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

        # Check if the index is already stored
        if os.path.exists(self.storage_dir):
            print("WORKFLOW: Index found on disk. Loading from storage...")
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            index = load_index_from_storage(storage_context)
        else:
            print("WORKFLOW: Index not found. Parsing and creating a new index...")
            try:
                # Parse the resume document
                documents = LlamaParse(
                    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                    result_type="markdown",
                    content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers"
                ).load_data(ev.resume_file)
                print(f"WORKFLOW: Parsed {len(documents)} documents.")

                # Create a new vector store index
                index = VectorStoreIndex.from_documents(
                    documents,
                    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
                )
                # Persist the index to disk
                print("WORKFLOW: Persisting index to disk...")
                index.storage_context.persist(persist_dir=self.storage_dir)
            except Exception as e:
                raise ValueError(f"Failed to parse and index the resume file: {str(e)}")

        # Create a query engine
        self.query_engine = index.as_query_engine(llm=self.llm, similarity_top_k=5)
        print("WORKFLOW: Query engine created successfully.")

        # Return the setup completion event
        return SetupCompleteEvent()

    @step
    async def handle_setup_complete(self, ctx: Context, ev: SetupCompleteEvent) -> StopEvent:
        """Handler for setup complete event - in a real app this would wait for queries."""
        print("WORKFLOW: Setup complete. Ready to handle queries.")
        # In a real application, you would wait for queries here
        # For visualization purposes, we'll just return a stop event
        return StopEvent(result="Setup complete. The system is ready to answer queries about the resume.")

    @step  
    async def ask_question(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        """Answer a query using the initialized RAG pipeline."""
        print(f"WORKFLOW: Received query: {ev.query}")

        try:
            # Query the engine
            response = self.query_engine.query(f"This is a question about the resume: {ev.query}")
            print(f"WORKFLOW: Query response: {response.response}")
        except Exception as e:
            print(f"WORKFLOW: Error during query execution: {str(e)}")
            raise ValueError(f"Query execution failed: {str(e)}")

        # Return the response as a StopEvent
        return StopEvent(result=response.response)

    def visualize(self, filename=""):
        """Visualize the workflow."""
        print("WORKFLOW: Visualizing the workflow...")
        base_static_path = "/home/ahmed/learn/agentic/apps/static"
        full_path = os.path.join(base_static_path, filename)

        # Generate the workflow visualization
        try:
            draw_all_possible_flows(self, filename=full_path)  # Use the workflow's structure
            print(f"WORKFLOW: Workflow visualization saved to {full_path}")
        except Exception as e:
            print(f"WORKFLOW: Error during visualization: {str(e)}")
            raise  # Re-raise the exception only if caught