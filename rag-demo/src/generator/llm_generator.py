import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class LLMGenerator:
    """
    Nhận context (từ Retriever) + query (từ User) → sinh câu trả lời bằng LLM.

    Luồng:
        (context, query) → Prompt Template → LLM → Response string
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        """
        Args:
            model_name: Tên mô hình LLM (mặc định là "gpt-4o-mini")
            temperature: Tham số điều chỉnh độ sáng tạo của câu trả lời (mặc định 0.3)
        """
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        
        # Khởi tạo LLM
        self.llm = ChatOpenAI(
            model = self.model_name,
            temperature = temperature,
            api_key = self.api_key,
        )

        # Prompt template
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("user", self._get_user_prompt()),
        ])

        # Output parser (chỉ lấy phần text trả về, loại bỏ metadata nếu có)
        self.output_parser = StrOutputParser()

        self.chain = self.rag_prompt | self.llm | self.output_parser

    def generate(self, context: str, question: str) -> str:
        """
        Sinh câu trả lời từ context + question.
        
        Args:
            context: Chuỗi context từ Retriever.retrieve_with_context()
            question: Câu hỏi của user
            
        Returns:
            str: Câu trả lời từ LLM
        """
        try:
            response = self.chain.invoke({
                "context": context,
                "question": question,
            })
            return response
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            raise

    def _get_system_prompt(self) -> str:
        """
        Trả về System prompt cho LLM.
        """
        return (
            "Bạn là trợ lý pháp luật AI chuyên về luật Việt Nam. "
            "Hãy trả lời câu hỏi dựa trên context được cung cấp. "
            "Nếu context không chứa đủ thông tin, hãy nói rõ rằng bạn không có đủ dữ liệu.\n\n"
            "Quy tắc:\n"
            "1. Chỉ trả lời dựa trên context, KHÔNG bịa thông tin.\n"
            "2. Trích dẫn điều luật cụ thể nếu có.\n"
            "3. Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu.\n"
            "4. Nếu không tìm thấy thông tin liên quan, trả lời: "
            "'Tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu.'"
        )
    
    def _get_user_prompt(self) -> str:
        """
        Trả về User prompt template cho LLM.
        """
        return (
            "Dưới đây là các tài liệu liên quan:\n\n"
            "{context}\n\n"
            "---\n"
            "Câu hỏi: {question}\n\n"
            "Hãy trả lời câu hỏi trên dựa vào context:"
        )
    
    def generate_with_custom_prompt(
            self, context: str, question: str, system_prompt: str
    ) -> str:
        """
        Sinh câu trả lời với system prompt tùy chỉnh.
        
        Args:
            context: Context string
            question: Câu hỏi
            system_prompt: System prompt tùy chỉnh
            
        Returns:
            str: Câu trả lời từ LLM
        """
        custom_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", self._get_user_prompt()),
        ])

        custom_chain = custom_prompt | self.llm | self.output_parser

        try:
            response = custom_chain.invoke({
                "context": context,
                "question": question,
            })
            return response
        except Exception as e:
            print(f"Error during LLM generation with custom prompt: {e}")
            raise
        