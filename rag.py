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

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

class FinancialRAGPipeline:

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path 
        self.load_environment_variables()
        self.setup_models()
        self.setup_storage()
    
    def load_environment_variables(self):
        try:
            load_dotenv()
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

            if not self.openai_api_key:
                logger.warning("OPEN AI API not found in environment variables")
        except Exception as e:
            logging.error(f"Error loading environment variables {e}")
            raise 
    
    def setup_models(self):
        try:
            self.local_model = ChatOllama(temperature = 0.5, model = "llama3.2:3b")
            self.embeddings = OllamaEmbeddings(model = "llama3.2:3b")
            self.openai_model = ChatOpenAI(
                model = "gpt-5-mini-2025-08-07",
                temperature = 0,
                max_tokens  = None,
                timeout = None,
                max_retries = 2
            )
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Error while initializing models: {e}")
            raise 
    
    def setup_storage(self):
        try:
            self.id_key = "doc_id"
            self.store = LocalFileStore("./local_docstore")

            self.vector_store = Chroma(
                collection_name = "rag_collection",
                embedding_function = self.embeddings,
                persist_directory = "./chroma_langchain_db"
            )

            self.retriever = MultiVectorRetriever(
                vectorstore = self.vector_store,
                docstore = self.store,
                id_key = self.id_key
            )
            logger.info("Storage components initiaziled successfully")
        except Exception as e:
            logger.error(f"error initializing storage: {e}")
            raise 
    
    def extract_pdf_content(self):
        try:
            logger.info(f"Processing PDF: {self.pdf_path}")

            chunks = partition_pdf(
                filename = self.pdf_path,
                infer_table_structure= True,
                strategy = "hi_res",
                extract_image_block_types=['Image'],
                extract_image_block_to_payload= True, 
                chunking_strategy="by_title",
                max_characters = 7000,
                combine_text_under_n_chars = 2000,
                new_after_n_chars = 6000
            )

            table_chunks = partition_pdf(
                filename = self.pdf_path,
                strategy = "hi_res"
            )

            texts = self._get_texts(chunks)
            tables = self._get_tables(table_chunks)
            images = self._get_images(chunks)

            logger.info(f"Extracted {len(texts)} texts, {len(images)} images and {len(tables)} tables")
            return texts, tables, images
        except Exception as e:
            logger.error(f"Errpr while extracting PDF content: {e}")
            raise 

    
    def _get_texts(self, chunks):
        try:
            texts = []
            for chunk in chunks:
                if "CompositeElement" in str(type(chunk)):
                    texts.append(chunk)
            return texts
        except Exception as e:
            logger.error(f"Error occured while extracting texts: {e}")
            return []
    
    def _get_tables(self, chunks):
        try:
            tables = []
            for chunk in chunks:
                if chunk.category == 'Table':
                    tables.append(chunk)
            return tables 
        except Exception as e:
            logger.error(f"Error occured while extracting tables: {e}")
            return []

    def _get_images(self, chunks):
        try:
            images_base64 = []
            for chunk in chunks:
                if hasattr(chunk, 'metadata') and hasattr(chunk.metadata,'orig_elements'):
                    for element in chunk.metadata.orig_elements:
                        if 'Image' in str(type(element)):
                            if hasattr(element.metadata, 'image_base64'):
                                images_base64.append(element.metadata.image_base64)
            return images_base64
        except Exception as e:
            logger.error(f"Error occured while extracting images: {e}")
            return []
    
    def generate_summaries(self, texts, tables):
        try:
            prompt_text = """
            You are an assistant tasked with summarizing tables and text.
            Give a concise summary of the table or text.

            Respond only with the summary, no additional comment.
            Do not start your message by saying "Here is a summary" or anything like that.
            Just give the summary as it is.

            Table or text chunk: {element}
            """
            prompt = ChatPromptTemplate.from_template(prompt_text)
            summarize_chain = {"element": lambda x:x} | prompt| self.local_model| StrOutputParser()

            text_summaries = []

            if texts:
                try:
                    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
                    logger.info(f"Generated {len(text_summaries)} text summaries")
                except Exception as e:
                    logger.error(f"Error generating text summaries: {e}")

            table_summaries = []
            if tables:
                try:
                    tables_text = [table.text for table in tables]
                    table_summaries = summarize_chain.batch(tables_text, {"max_concurrency":3})
                    logger.info(f"Generated {len(table_summaries)} table sumaries")
                except Exception as e:
                    logger.error(f"Error generating table summaries: {e}")

            return text_summaries, table_summaries 
        
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return [], []
    
    def generate_image_summaries(self, images):
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
            chain = {"image": lambda x:x} | prompt | ChatOllama(model="gemma3:4b") | StrOutputParser()
            image_summaries = []
            try:
                image_summaries= chain.batch(images)
                logger.info(f"Generated {len(image_summaries)} iamge summaries")
                return image_summaries 
            except Exception as e:
                logger.error(f"error while generating image summaries: {e}")
                return []
        except Exception as e:
            logger.error(f"Error generating image summaries: {e}")
            return []

    def store_documents(self, texts, text_summaries, tables, tables_summaries, images, images_summaries):
        try:
            if texts and text_summaries:
                self._store_text_documents(texts, text_summaries)
            
            if tables and tables_summaries:
                self._store_table_documents(tables, tables_summaries)
            
            if images and images_summaries:
                self._store_image_documents(images, images_summaries)

            logger.info("all documents stored successfully")

        except Exception as e:
            logger.error(f"Error occur while storing documents: {e}")
            raise 
    
    def _store_text_documents(self, texts, texts_summaries):
        try:
            text_ids = [str(uuid.uuid4()) for _ in texts]
            summary_texts = [
                Document(page_content = summary, metadata = {self.id_key: text_ids[i], "type":"text summary"}) for i, summary in enumerate(texts_summaries)
            ]
            full_texts = [
                Document(page_content = text.text, metadata = {self.id_key:text_ids[i],"type":"full summary"}) for i,text in enumerate(texts)
            ]
            encoded_texts = [str(text).encode('utf-8') for text in texts]
            self.retriever.vectorstore.add_documents(summary_texts+full_texts)
            self.retriever.docstore.mset(list(zip(text_ids, encoded_texts)))

            logger.info(f"Stored {len(texts)} text documents")

        except Exception as e:
            logger.error(f"Error storing text documents: {e}")
            raise 
    
    def _store_table_documents(self, tables, tables_summaries):
        try:
            table_ids = [str(uuid.uuid4()) for _ in tables]
            summary_tables = [
                Document(page_content = summary, metadata = {self.id_key:table_ids[i], "type":"table summary"})
                for i, summary in enumerate(tables_summaries)
                ]
            # tables = [table.text for table in tables]
            full_tables = [
                Document(page_content=table.text, metadata ={self.id_key:table_ids[i], "type":"full table text"}) 
                for i,table in enumerate(tables)
            ]
            encoded_tables = [str(table.text).encode('utf-8') for table in tables]
            self.retriever.vectorstore.add_documents(summary_tables+full_tables)
            self.retriever.docstore.mset(list(zip(table_ids, encoded_tables)))
            logger.info(f"Successfully stored {len(tables)} table documents")

        except Exception as e:
            logger.error(f"Error occured storing table documents: {e}")
            raise 
    
    def _store_image_documents(self, images, images_summaries):
        try:
            image_ids = [str(uuid.uuid4()) for _ in images]
            summary_images = [
                Document(page_content = summary, metadata = {self.id_key:image_ids[i]})
                for i, summary in enumerate(images_summaries)
            ]
            encoded_images = [image.encode('utf-8') for image in images]
            self.retriever.vectorstore.add_documents(summary_images)
            self.retriever.docstore.mset(list(zip(image_ids, encoded_images)))
            logger.info(f"Stored {len(images)} image documetns")        

        except Exception as e:
            logger.error(f"Error occured while storing image documents: {e}")
            raise 
    
    def query(self, question):
        try:
            logger.info(f"Processing query: {question}")
            retrieved_data = self.vector_store.similarity_search(question, k = 3)
            # retrieved_data = self.retriever.invoke(question)

            if not retrieved_data:
                return "relavant information regarding your question not found"
            context = "\n\n".join([doc.page_content for doc in retrieved_data])
            chat_prompt = """
            You are an auditor/expert in reading financial statements.
            I will provide you report information. Based on information provide concise report.
            Context: {context}
            Question: {question}
            """
            prompt_chat = ChatPromptTemplate.from_template(chat_prompt)
            result_chain = prompt_chat| self.local_model | StrOutputParser()

            response = result_chain.invoke({"context": context,"question":question})
            logger.info("Qeury processed successfully")

            return response 
        
        except Exception as e:
            logger.error(f"Error while processing query: {e}")
            return f"Error processing your question: {str(e)}"
    

    def run_pipelines(self):
        try:
            logger.info("Runnig Rag Pipeline")

            # extract content from pdf
            texts, tables, images = self.extract_pdf_content()

            # generate summaries 
            texts_summaries, tables_summaries = self.generate_summaries(texts, tables)
            images_summaries = self.generate_image_summaries(images)

            self.store_documents(texts, texts_summaries, tables, tables_summaries, images, images_summaries)
            logger.info(f"Rag pipeline completed successfully")
        except Exception as e:
            logger.error(f"error in rag pipeline: {e}")
            raise 


def main():
    try:
        pdf_path = 'pdf/2022 Q3 AAPL.pdf'
        pipeline = FinancialRAGPipeline(pdf_path)
        pipeline.run_pipelines()

        questions = [
            "What is net sales?",
            "What are the main financial highlights?",
            "What tables are available in the report?"
        ]

        for question in questions:
            print(f"\n question: {question}")
            answer = pipeline.query(question)
            print(f"answer: {answer}")
    except Exception as e:
        logger.error(f"error occured in main: {e}")

if __name__ == "__main__":
    main()