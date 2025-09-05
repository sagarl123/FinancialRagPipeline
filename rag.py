from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain_openai import ChatOpenAI
from langchain.storage import LocalFileStore
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
import os
import uuid
import logging
from typing import List, Optional, Tuple, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialRAGPipeline:
    """
    A comprehensive RAG pipeline for processing financial documents with tables, text, and images.
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize the RAG pipeline.
        
        Args:
            pdf_path (str): Path to the PDF file to process
        """
        self.pdf_path = pdf_path
        self.load_environment_variables()
        self.setup_models()
        self.setup_storage()
        
    def load_environment_variables(self) -> None:
        """Load environment variables from .env file."""
        try:
            load_dotenv()
            self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            
            if not self.openai_api_key:
                logger.warning("OPENAI_API_KEY not found in environment variables")
                
        except Exception as e:
            logger.error(f"Error loading environment variables: {e}")
            raise
    
    def setup_models(self) -> None:
        """Initialize the language models."""
        try:
            self.local_model = ChatOllama(temperature=0.5, model="llama3.2:3b")
            self.embeddings = OllamaEmbeddings(model="llama3.2:3b")
            self.openai_model = ChatOpenAI(
                model="gpt-4o-mini",  # Fixed model name
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def setup_storage(self) -> None:
        """Initialize storage components."""
        try:
            self.id_key = "doc_id"
            self.store = LocalFileStore("./local_docstore")
            
            self.vector_store = Chroma(
                collection_name='rag_collection',
                embedding_function=self.embeddings,
                persist_directory="./chroma_langchain_db"
            )
            
            self.retriever = MultiVectorRetriever(
                vectorstore=self.vector_store,
                docstore=self.store,
                id_key=self.id_key
            )
            logger.info("Storage components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
            raise
    
    def extract_pdf_content(self) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        Extract text, tables, and images from PDF.
        
        Returns:
            Tuple containing (texts, tables, images)
        """
        try:
            logger.info(f"Processing PDF: {self.pdf_path}")
            
            # Extract chunks with images
            chunks = partition_pdf(
                filename=self.pdf_path,
                infer_table_structure=True,
                strategy="hi_res",
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=10000,
                combine_text_under_n_chars=2000,
                new_after_n_chars=6000
            )
            
            # Extract table chunks
            table_chunks = partition_pdf(
                filename=self.pdf_path,
                strategy='hi_res'
            )
            
            texts = self._get_texts(chunks)
            tables = self._get_tables(table_chunks)
            images = self._get_images(chunks)
            
            logger.info(f"Extracted {len(texts)} texts, {len(tables)} tables, {len(images)} images")
            return texts, tables, images
            
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            raise
    
    def _get_texts(self, chunks: List[Any]) -> List[Any]:
        """Extract text chunks from PDF chunks."""
        try:
            texts = []
            for chunk in chunks:
                if 'CompositeElement' in str(type(chunk)):
                    texts.append(chunk)
            return texts
        except Exception as e:
            logger.error(f"Error extracting texts: {e}")
            return []
    
    def _get_tables(self, chunks: List[Any]) -> List[Any]:
        """Extract table chunks from PDF chunks."""
        try:
            tables = []
            for chunk in chunks:
                if chunk.category == "Table":
                    tables.append(chunk)
            return tables
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
            return []
    
    def _get_images(self, chunks: List[Any]) -> List[str]:
        """Extract base64 encoded images from PDF chunks."""
        try:
            images_base64 = []
            for chunk in chunks:
                if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                    for element in chunk.metadata.orig_elements:
                        if 'Image' in str(type(element)):
                            if hasattr(element.metadata, 'image_base64'):
                                images_base64.append(element.metadata.image_base64)
            return images_base64
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
            return []
    
    def generate_summaries(self, texts: List[Any], tables: List[Any]) -> Tuple[List[str], List[str]]:
        """
        Generate summaries for texts and tables.
        
        Returns:
            Tuple containing (text_summaries, table_summaries)
        """
        try:
            # Text and table summarization prompt
            prompt_text = """
            You are an assistant tasked with summarizing tables and text.
            Give a concise summary of the table or text.

            Respond only with the summary, no additional comment.
            Do not start your message by saying "Here is a summary" or anything like that.
            Just give the summary as it is.

            Table or text chunk: {element}
            """
            
            prompt = ChatPromptTemplate.from_template(prompt_text)
            summarize_chain = {"element": lambda x: x} | prompt | self.local_model | StrOutputParser()
            
            # Generate text summaries
            text_summaries = []
            if texts:
                try:
                    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
                    logger.info(f"Generated {len(text_summaries)} text summaries")
                except Exception as e:
                    logger.error(f"Error generating text summaries: {e}")
            
            # Generate table summaries
            table_summaries = []
            if tables:
                try:
                    tables_html = [table.text for table in tables]
                    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})
                    logger.info(f"Generated {len(table_summaries)} table summaries")
                except Exception as e:
                    logger.error(f"Error generating table summaries: {e}")
            
            return text_summaries, table_summaries
            
        except Exception as e:
            logger.error(f"Error in summary generation: {e}")
            return [], []
    
    def generate_image_summaries(self, images: List[str]) -> List[str]:
        """Generate summaries for images using vision model."""
        try:
            if not images:
                return []
            
            image_prompt_template = """Describe the image in detail. For context,
            the image is a logo of APPLE company and it is the part of quarterly report of financial transaction."""
            
            messages = [
                (
                    "user",
                    [
                        {"type": "text", "text": image_prompt_template},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,{image}"},
                        },
                    ],
                )
            ]
            
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = {"image": lambda x: x} | prompt | ChatOllama(model="llama3.2-vision:11b") | StrOutputParser()
            
            try:
                image_summaries = chain.batch(images)
                logger.info(f"Generated {len(image_summaries)} image summaries")
                return image_summaries
            except Exception as e:
                logger.error(f"Error generating image summaries: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Error in image summary generation: {e}")
            return []
    
    def store_documents(self, texts: List[Any], text_summaries: List[str], 
                       tables: List[Any], table_summaries: List[str],
                       images: List[str], image_summaries: List[str]) -> None:
        """Store all documents and summaries in the vector store and docstore."""
        try:
            # Store text documents
            if texts and text_summaries:
                self._store_text_documents(texts, text_summaries)
            
            # Store table documents
            if tables and table_summaries:
                self._store_table_documents(tables, table_summaries)
            
            # Store image documents
            if images and image_summaries:
                self._store_image_documents(images, image_summaries)
                
            logger.info("All documents stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            raise
    
    def _store_text_documents(self, texts: List[Any], text_summaries: List[str]) -> None:
        """Store text documents and their summaries."""
        try:
            text_ids = [str(uuid.uuid4()) for _ in texts]
            summary_texts = [
                Document(page_content=summary, metadata={self.id_key: text_ids[i]}) 
                for i, summary in enumerate(text_summaries)
            ]
            
            # Encode texts for LocalFileStore
            encoded_texts = [str(text).encode('utf-8') for text in texts]
            
            self.retriever.vectorstore.add_documents(summary_texts)
            self.retriever.docstore.mset(list(zip(text_ids, encoded_texts)))
            
            logger.info(f"Stored {len(texts)} text documents")
            
        except Exception as e:
            logger.error(f"Error storing text documents: {e}")
            raise
    
    def _store_table_documents(self, tables: List[Any], table_summaries: List[str]) -> None:
        """Store table documents and their summaries."""
        try:
            table_ids = [str(uuid.uuid4()) for _ in tables]
            summary_tables = [
                Document(page_content=summary, metadata={self.id_key: table_ids[i]}) 
                for i, summary in enumerate(table_summaries)
            ]
            
            # Encode table text for LocalFileStore
            encoded_tables = [str(table.text).encode('utf-8') for table in tables]
            
            self.retriever.vectorstore.add_documents(summary_tables)
            self.retriever.docstore.mset(list(zip(table_ids, encoded_tables)))
            
            logger.info(f"Stored {len(tables)} table documents")
            
        except Exception as e:
            logger.error(f"Error storing table documents: {e}")
            raise
    
    def _store_image_documents(self, images: List[str], image_summaries: List[str]) -> None:
        """Store image documents and their summaries."""
        try:
            image_ids = [str(uuid.uuid4()) for _ in images]
            summary_images = [
                Document(page_content=summary, metadata={self.id_key: image_ids[i]}) 
                for i, summary in enumerate(image_summaries)
            ]
            
            # Encode images for LocalFileStore
            encoded_images = [image.encode('utf-8') for image in images]
            
            self.retriever.vectorstore.add_documents(summary_images)
            self.retriever.docstore.mset(list(zip(image_ids, encoded_images)))
            
            logger.info(f"Stored {len(images)} image documents")
            
        except Exception as e:
            logger.error(f"Error storing image documents: {e}")
            raise
    
    def query(self, question: str) -> str:
        """
        Query the RAG system with a question.
        
        Args:
            question (str): The question to ask
            
        Returns:
            str: The answer from the RAG system
        """
        try:
            logger.info(f"Processing query: {question}")
            
            # Retrieve relevant documents
            retrieved_data = self.retriever.invoke(question)
            
            if not retrieved_data:
                return "No relevant information found for your question."
            
            # Generate response using OpenAI
            chat_prompt = """
            You are an auditor/expert in reading financial statements.
            I will provide you report information. Based on information provide concise report.
            The information is: {element}
            """
            
            prompt_chat = ChatPromptTemplate.from_template(chat_prompt)
            result_chain = prompt_chat | self.openai_model | StrOutputParser()
            
            response = result_chain.invoke({"element": retrieved_data})
            logger.info("Query processed successfully")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing your question: {str(e)}"
    
    def run_pipeline(self) -> None:
        """Run the complete RAG pipeline."""
        try:
            logger.info("Starting RAG pipeline...")
            
            # Extract content from PDF
            texts, tables, images = self.extract_pdf_content()
            
            # Generate summaries
            text_summaries, table_summaries = self.generate_summaries(texts, tables)
            image_summaries = self.generate_image_summaries(images)
            
            # Store documents
            self.store_documents(texts, text_summaries, tables, table_summaries, 
                               images, image_summaries)
            
            logger.info("RAG pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            raise


def main():
    """Example usage of the FinancialRAGPipeline."""
    try:
        # Initialize pipeline
        pdf_path = 'pdf/2022 Q3 AAPL.pdf'
        pipeline = FinancialRAGPipeline(pdf_path)
        
        # Run the complete pipeline
        pipeline.run_pipeline()
        
        # Example queries
        questions = [
            "What is net sales?",
            "What are the main financial highlights?",
            "What tables are available in the report?"
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            answer = pipeline.query(question)
            print(f"Answer: {answer}")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()