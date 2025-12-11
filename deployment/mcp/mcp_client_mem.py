import asyncio
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession   
import time
from functions import get_session


session_id = 'test_session'
qa_pairs = [
    {
        "question": "Gói giải mã gen GeneMap-Adult có thể giúp con người hiểu rõ hơn về khả năng thích ứng và phát triển tính cách như thế nào?",
        "answer": "GeneMap-Adult phân tích tính cách theo mô hình Big5 và RIASEC, từ đó cho thấy mức độ cởi mở, hướng ngoại hay tận tâm của một người, giúp họ hiểu rõ cách mình có thể thích ứng và phát triển trong các hoàn cảnh thay đổi."
    },
    {
        "question": "Theo kết quả của GeneMap-Adult, đặc điểm nào trong mô hình Big5 phản ánh khả năng cởi mở và dễ thích nghi?",
        "answer": "Trong Big5, yếu tố 'Cởi mở với trải nghiệm' phản ánh trực tiếp khả năng chấp nhận cái mới và dễ dàng thích nghi với sự thay đổi."
    },
    {
        "question": "Mô hình RIASEC trong GeneMap-Adult có thể cho biết xu hướng phát triển nghề nghiệp của một người khi hoàn cảnh thay đổi không?",
        "answer": "Có. RIASEC chỉ ra nhóm nghề nghiệp phù hợp như Nhà sáng tạo, Người tổ chức hay Người thương thuyết, giúp cá nhân định hướng công việc linh hoạt hơn khi môi trường biến đổi."
    },
    {
        "question": "GeneMap-Adult có đề cập đến 8 nhóm thông minh Gardner – liệu sự đa dạng trí tuệ này có thể coi là minh chứng cho khả năng phát triển đặc điểm mới của con người?",
        "answer": "Đúng. Sự tồn tại của nhiều dạng trí thông minh (ngôn ngữ, logic, vận động, âm nhạc, nội tâm...) cho thấy con người có thể phát triển hoặc tăng cường các đặc điểm mới khi điều kiện thay đổi."
    },
    {
        "question": "Việc dự báo nguy cơ mắc các bệnh (Ung thư, Cao huyết áp, Gút...) có thể hỗ trợ con người điều chỉnh lối sống để thích nghi với môi trường sống như thế nào?",
        "answer": "Thông tin dự báo bệnh giúp cá nhân chủ động thay đổi chế độ ăn uống, tập luyện và tầm soát định kỳ để giảm thiểu rủi ro, từ đó thích nghi tốt hơn với môi trường sống hiện tại và tương lai."
    },
    {
        "question": "Gói GeneMap-Adult cung cấp khuyến nghị về tầm soát định kỳ – điều này có giúp con người chủ động thích nghi với thay đổi trong sức khỏe không?",
        "answer": "Có. Khuyến nghị tầm soát định kỳ giúp phát hiện sớm các vấn đề sức khỏe tiềm ẩn, nhờ đó con người kịp thời điều chỉnh lối sống và phòng ngừa bệnh tật."
    },
    {
        "question": "Giá trị 4.800.000 cho một gói GeneMap-Adult có thể được coi là đầu tư cho khả năng thích ứng dài hạn của con người hay không?",
        "answer": "Có thể coi đây là một khoản đầu tư hợp lý, vì thông tin di truyền cung cấp giúp cá nhân hiểu rõ nguy cơ, tiềm năng và định hướng phát triển dài hạn, từ đó nâng cao khả năng thích ứng."
    }
]

test_queries = [
        "GeneMap-Adult có thể giúp gì cho khả năng thích ứng?",
        "Đặc điểm Big5 nào liên quan đến khả năng cởi mở?",
        "RIASEC có thể dự báo nghề nghiệp không?",
    ]


async def main():


    async with get_session() as session:
        print(f"Preparing to use MCP session...")
        print(session)
        list_tools = await session.list_tools()
        print("Available Tools:", list_tools)

        # Create a list of awaitable tasks, one for each query
        tasks = [
        session.call_tool(
            "memory_saver",
            arguments={"input": qa["question"], "output": qa["answer"], "session_id": session_id}
        ) for qa in qa_pairs
    ]
        print("\n--- Starting Memory Saving ---")
        start_time = time.time()
        await asyncio.gather(*tasks)
        end_time = time.time()
        print(f"--- Finished Memory Saving in {end_time - start_time:.2f} seconds ---")


        # load memory
        tasks = [
            session.call_tool(
                "memory_loader",
                arguments={"query": q, "session_id": session_id}
            ) for q in test_queries
        ]

        print("\n--- Starting Memory Loading ---")
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        print(f"--- Finished Memory Loading in {end_time - start_time:.2f} seconds ---")

        for query, result in zip(test_queries, results):
            print("Query:", query)
            print("Memory Loaded:", result.content[0].text if result.content else "No content")
            



if __name__ == "__main__":
    asyncio.run(main())