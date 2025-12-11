import asyncio
from acp_sdk.client import Client

async def run_workflow() -> None:
    async with Client(base_url="http://127.0.0.1:6666") as drafter:

        topic = "Impact of climate change on agriculture in 2025."
        
        response1 = await drafter.run_sync(
            agent="research_drafter",
            input=topic
        )
        draft = response1.output[0].parts[0].content
        print(f"\nDraft Summary:\n{draft}")

        
        

if __name__ == "__main__":
    asyncio.run(run_workflow())