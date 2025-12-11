import time
import uuid
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from mcp.server.fastmcp import FastMCP
import os
import asyncio
from dotenv import load_dotenv
from os.path import join, dirname
from sqlalchemy.ext.asyncio import create_async_engine,AsyncSession
from sqlalchemy.orm import sessionmaker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.memory import VectorStoreRetrieverMemory
from functions import cosine_relevance_score
from sqlalchemy import text
# --- 1. SETUP ---

env_path = join(dirname(__file__), '.env')
load_dotenv(env_path)
DEVICE = os.getenv("DEVICE", "cpu")
mcp = FastMCP("PostgreSQL-VDB", port=os.getenv("MCP_PORT", 50051), host=os.getenv("MCP_HOST", "192.168.1.60"))

print("--- Initializing Embeddings Model (this may take a moment) ---")
print("Database URL: ", os.getenv("DB_URL"))
db_engine = create_async_engine(os.getenv("DB_URL"), echo=False, future=True)
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding_v2"),
    model_kwargs={'device': DEVICE}, # or 'cpu'
    encode_kwargs={'normalize_embeddings': True}
)

reranker_device = DEVICE  # Change to "cuda:0" if you have GPU
reranker_model_kwargs = {'device': reranker_device}
reranker_model_path = os.getenv("RERANKING_MODEL", "AITeamVN/Vietnamese_Reranker")
reranker_model = HuggingFaceCrossEncoder(
    model_name=reranker_model_path, 
    model_kwargs=reranker_model_kwargs
)
collection = "test_memory_vdb" 
vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection,
            connection=db_engine,
            use_jsonb=True,
            relevance_score_fn=cosine_relevance_score,
        )
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
memory = VectorStoreRetrieverMemory(retriever=retriever)
# memory.save_context({"input": "Who is Jay?"}, {"output": "A data scientist"})

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

async def save_memory():
    for qa in qa_pairs:
        await memory.asave_context(
            {"input": qa["question"]},
            {"output": qa["answer"]}
        )

# asyncio.run(save_memory())
# Query long-term memory
import time
async def load_memory():
    test_queries = [
        "GeneMap-Adult có thể giúp gì cho khả năng thích ứng?",
        "Đặc điểm Big5 nào liên quan đến khả năng cởi mở?",
        "RIASEC có thể dự báo nghề nghiệp không?",
    ]
    for query in test_queries:
        start_time = time.time()
        docs = await memory.aload_memory_variables({"input": query})
        end_time = time.time()
        print(f"Query: {query}")
        print("Docs:", docs['history'])
        print("Docs Type:", type(docs))

        print(f"Time taken: {end_time - start_time:.4f} seconds")
        print("-" * 40)

    
# asyncio.run(load_memory())




AsyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=db_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


#query collections in db 
session = AsyncSessionLocal()
#  public | langchain_pg_collection | table | datpd

query_collections = text("SELECT lc.name FROM langchain_pg_collection AS lc")
async def get_collections():
    async with AsyncSessionLocal() as session:
        collections = await session.execute(query_collections)
        print("Collections in DB:")
        collections = collections.scalars().all()
        type_collections = type(collections)
        print("Type of collections:", type_collections)
        for collection in collections:
            print("-", collection)
    await session.close()

            
            
asyncio.run(get_collections())