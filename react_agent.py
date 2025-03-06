import os
from typing import Dict, Any
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.azure_openai import AzureOpenAI

class EdisonReActAgent:
    """Edison ReAct Agent that uses dummy tools for testing."""

    def __init__(self):
        # Configure Azure OpenAI
        self.llm = AzureOpenAI(
            engine=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_KEY"),
            azure_endpoint=os.getenv("LLM_ENDPOINT"),
            api_version=os.getenv("OPENAI_VERSION", "2024-02-01"),
        )

        # Create and initialize the agent
        self.agent = self._create_agent()

    def _create_agent(self) -> ReActAgent:
        """Create and configure the ReAct agent with dummy tools."""
        # Define Edison's dummy tools
        tools = [
            FunctionTool.from_defaults(
                fn=self._ocr_retrieval_tool,
                name="ocr_retrieval",
                description="For processing queries with attached homework images, diagrams or screenshots"
            ),
            FunctionTool.from_defaults(
                fn=self._qa_retrieval_tool,
                name="qa_retrieval",
                description="For retrieving similar past questions and answers from the course database"
            ),
            FunctionTool.from_defaults(
                fn=self._textbook_retrieval_tool,
                name="textbook_retrieval",
                description="For retrieving information from lecture content, textbooks or course materials"
            ),
            FunctionTool.from_defaults(
                fn=self._hw_retrieval_tool,
                name="hw_retrieval",
                description="For retrieving specific homework, lab, project or worksheet materials"
            ),
            FunctionTool.from_defaults(
                fn=self._logistics_retrieval_tool,
                name="logistics_retrieval",
                description="For retrieving course logistics information like deadlines, policies, and schedules"
            )
        ]

        # Create the agent
        return ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            verbose=True
        )

    def _ocr_retrieval_tool(self, content: str) -> str:
        """Dummy OCR tool that just prints and returns a message."""
        print(f"[DUMMY OCR] Processing image content: {content[:50]}...")
        return "Retrieved OCR content: This is dummy OCR text extracted from an image."

    def _qa_retrieval_tool(self, query: str, top_k: int = 3) -> str:
        """Dummy QA retrieval tool that just prints and returns a message."""
        print(f"[DUMMY QA] Retrieving QA pairs for: {query[:50]}...")
        return "Retrieved historical QA: This is a dummy response with historical question-answer pairs."

    def _textbook_retrieval_tool(self, query: str) -> str:
        """Dummy textbook content retrieval tool that just prints and returns a message."""
        print(f"[DUMMY TEXTBOOK] Retrieving textbook content for: {query[:50]}...")
        return "Retrieved course documents: This is dummy textbook content from course materials."

    def _hw_retrieval_tool(self, query: str, question_category: str = None) -> str:
        """Dummy homework retrieval tool that just prints and returns a message."""
        print(f"[DUMMY HW] Retrieving homework materials for: {query[:50]}...")
        return "Retrieved homework-related content: This is dummy homework problem content."

    def _logistics_retrieval_tool(self, query: str) -> str:
        """Dummy logistics retrieval tool that just prints and returns a message."""
        print(f"[DUMMY LOGISTICS] Retrieving logistics info for: {query[:50]}...")
        return "Retrieved logistics information: This is dummy course logistics information."

    def process_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Process a query using the ReAct agent with optional context."""
        # Create a system prompt to guide tool selection
        # system_prompt = """You are Edison, an educational assistant. Based on the category and content of the question, choose the appropriate tools:
        # - Use ocr_retrieval for processing images and diagrams
        # - Use qa_retrieval for finding similar past questions and answers
        # - Use textbook_retrieval for lecture and textbook material lookups
        # - Use hw_retrieval for homework, project, assignment, or worksheet specific questions
        # - Use logistics_retrieval for course logistics, deadlines, or policy questions

        # You can use multiple tools if needed to give a complete response."""

        # Format the query with context
        formatted_query = query
        if context:
            category = context.get("category", "")
            subcategory = context.get("subcategory", "")
            thread_title = context.get("thread_title", "")
            formatted_query = f"[Category: {category}] [Subcategory: {subcategory}] [Thread: {thread_title}]\n{query}"

        # Get response from the agent with the system prompt
        print(f"[AGENT] Processing query: {formatted_query[:50]}...")
        response = self.agent.chat(formatted_query)
        return response.response