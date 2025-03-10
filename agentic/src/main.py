import asyncio
from dotenv import load_dotenv
from workflow import MyWorkflow

load_dotenv()

async def main():
    workflow = MyWorkflow(timeout=10, verbose=False)
    result = await workflow.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())