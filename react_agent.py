import os
from typing import List, Dict, Any
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.azure_openai import AzureOpenAI
import ast

from utils import (
    question_ocr,
    retrieve_qa,
    retrieve_docs_hybrid,
    retrieve_docs_manual
)

class EdisonReActAgent:
    """Edison ReAct Agent that dynamically selects retrieval tools based on context."""

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
        """Create and configure the ReAct agent with tools."""
        # Define Edison's tools
        tools = [
            FunctionTool.from_defaults(
                fn=self._ocr_tool,
                name="ocr_retrieval",
                description="For processing queries with attached homework images, diagrams or screenshots"
            ),
            FunctionTool.from_defaults(
                fn=self._qa_retrieval_tool,
                name="qa_retrieval",
                description="For retrieving similar past questions and answers from the course database"
            ),
            FunctionTool.from_defaults(
                fn=self._content_retrieval_tool,
                name="content_retrieval",
                description="For retrieving information from lecture content, textbooks or course materials"
            ),
            FunctionTool.from_defaults(
                fn=self._homework_retrieval_tool,
                name="hw_retrieval",
                description="For retrieving specific homework, lab, project or worksheet materials"
            )
        ]

        # Create the agent
        return ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            verbose=True
        )

    def _ocr_tool(self, content: str) -> str:
        """Process images using OCR."""
        try:
            return question_ocr(content)
        except Exception as e:
            return f"Error processing image: {str(e)}"

    def _qa_retrieval_tool(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant QA pairs."""
        try:
            return retrieve_qa(conversation=query, top_k=top_k)
        except Exception as e:
            return f"Error retrieving QA pairs: {str(e)}"

    def _content_retrieval_tool(self, query: str) -> str:
        """Retrieve course content."""
        try:
            return retrieve_docs_hybrid(
                text=query,
                index_name=os.getenv('CONTENT_INDEX_NAME'),
                top_k=int(os.getenv('CONTENT_INDEX_TOP_K', '1')),
                semantic_reranking=True
            )
        except Exception as e:
            return f"Error retrieving content: {str(e)}"

    def _homework_retrieval_tool(self, query: str, question_category: str = None) -> str:
        """Retrieve homework-specific content."""
        try:
            # Parse category mappings from environment variables
            category_mapping = ast.literal_eval(os.getenv('CATEGORY_MAPPING', '{}'))
            subcategory_mapping = ast.literal_eval(os.getenv('SUBCATEGORY_MAPPING', '{}'))

            # Extract subcategory from query if possible
            # This is a simple approach - in production you might want to use the LLM to extract this
            subcategory = ""
            if ":" in query:
                parts = query.split(":", 1)
                if len(parts) > 1 and len(parts[0].strip().split()) <= 3:  # Simple heuristic for detecting categories
                    subcategory = parts[0].strip()

            # Create a simple prompt function for document path selection
            def get_prompt(paths, question_info):
                return [
                    {"role": "system", "content": "You are an assistant that helps select the most relevant document path for a student question. Choose the path that best matches the question context."},
                    {"role": "user", "content": f"Here are the available paths:\n{paths}\n\nQuestion: {question_info}\n\nReturn a Python dictionary with the key 'selected_path' and the most appropriate path as the value."}
                ]

            # Call retrieve_docs_manual with all necessary parameters
            problem_list_manual, selected_doc_manual, retrieved_docs = retrieve_docs_manual(
                question_category=question_category or "Homework",
                category_mapping=category_mapping,
                question_subcategory=subcategory,
                subcategory_mapping=subcategory_mapping,
                question_info=query,
                get_prompt=get_prompt
            )

            # Format the response to provide context
            if retrieved_docs != 'none' and retrieved_docs != 'none (error)':
                return f"Retrieved homework-related content from {selected_doc_manual}:\n\n{retrieved_docs}"
            else:
                return f"No specific homework materials found. Available documents include: {problem_list_manual}"

        except Exception as e:
            return f"Error retrieving homework materials: {str(e)}"

    def process_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Process a query using the ReAct agent with optional context."""
        # Create a system prompt to guide tool selection
        system_prompt = """You are Edison, an educational assistant. Based on the category and content of the question, choose the appropriate tools:
        - Use ocr_retrieval for processing images and diagrams
        - Use qa_retrieval for finding similar past questions and answers
        - Use content_retrieval for lecture and textbook material lookups
        - Use hw_retrieval for homework, project, assignment, or worksheet specific questions

        You can use multiple tools if needed to give a complete response."""

        # Format the query with context
        formatted_query = query
        if context:
            category = context.get("category", "")
            subcategory = context.get("subcategory", "")
            thread_title = context.get("thread_title", "")
            formatted_query = f"[Category: {category}] [Subcategory: {subcategory}] [Thread: {thread_title}]\n{query}"

        # Get response from the agent with the system prompt
        response = self.agent.chat(formatted_query, system_prompt=system_prompt)
        return response.response