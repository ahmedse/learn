import pytest
import asyncio
from src.workflow import MyWorkflow

@pytest.mark.asyncio
async def test_my_workflow():
    workflow = MyWorkflow(timeout=10, verbose=False)
    result = await workflow.run()
    assert result == "Hello, World!"