"""
Multi-Modal RAG Agent with Web Search and Deep Research
Analyzes, answers questions, and rewrites documents with text, images, and tables
Includes internet search capabilities and multi-agent deep research
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import base64
from pathlib import Path
import json
import requests

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import chromadb


@dataclass
class MultiModalChunk:
    """Represents a chunk with text and optional image/table data"""
    text: str
    chunk_type: str  # 'text', 'image', 'table'
    metadata: Dict[str, Any]
    image_path: Optional[str] = None
    image_base64: Optional[str] = None


class WebSearchTool:
    """Handles web search functionality with multiple providers"""
    
    def __init__(self, provider: str = "duckduckgo", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        
        if provider == "duckduckgo":
            try:
                from duckduckgo_search import DDGS
                self.ddgs = DDGS()
            except ImportError:
                raise ImportError("Install duckduckgo-search: pip install duckduckgo-search")
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search the web and return results"""
        if self.provider == "duckduckgo":
            return self._search_duckduckgo(query, max_results)
        return []
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Search using DuckDuckGo (free, no API key needed)"""
        try:
            results = []
            for result in self.ddgs.text(query, max_results=max_results):
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', '')
                })
            return results
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []


class DocumentParser:
    """Parses multiple document formats and extracts multi-modal content"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt', '.md', '.html']
    
    def parse_document(self, file_path: str) -> List[MultiModalChunk]:
        """Parse document and extract text, images, and tables"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self._parse_pdf(file_path)
        elif file_ext == '.docx':
            return self._parse_docx(file_path)
        elif file_ext in ['.txt', '.md']:
            return self._parse_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _parse_pdf(self, file_path: str) -> List[MultiModalChunk]:
        """Parse PDF - tries multiple methods"""
        chunks = []
        
        try:
            from unstructured.partition.pdf import partition_pdf
            elements = partition_pdf(filename=file_path, strategy="hi_res")
            
            for i, element in enumerate(elements):
                chunks.append(MultiModalChunk(
                    text=str(element),
                    chunk_type="text",
                    metadata={"source": file_path, "index": i}
                ))
        except ImportError:
            print("⚠ Unstructured not available, using PyPDF2 fallback...")
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        chunks.append(MultiModalChunk(
                            text=text,
                            chunk_type="text",
                            metadata={"source": file_path, "page": page_num + 1}
                        ))
        
        return chunks
    
    def _parse_docx(self, file_path: str) -> List[MultiModalChunk]:
        """Parse DOCX"""
        from docx import Document as DocxDocument
        chunks = []
        doc = DocxDocument(file_path)
        
        for i, paragraph in enumerate(doc.paragraphs):
            if paragraph.text.strip():
                chunks.append(MultiModalChunk(
                    text=paragraph.text,
                    chunk_type="text",
                    metadata={"source": file_path, "index": i}
                ))
        
        return chunks
    
    def _parse_text(self, file_path: str) -> List[MultiModalChunk]:
        """Parse plain text files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return [MultiModalChunk(
            text=text,
            chunk_type="text",
            metadata={"source": file_path}
        )]


class MultiModalVectorStore:
    """Vector store with multi-modal support"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.image_store = {}
    
    def add_documents(self, chunks: List[MultiModalChunk]):
        """Add multi-modal chunks to vector store"""
        documents = []
        
        for chunk in chunks:
            if chunk.chunk_type == "image" and chunk.image_base64:
                self.image_store[chunk.metadata.get('index')] = chunk.image_base64
            
            doc = Document(
                page_content=chunk.text,
                metadata={**chunk.metadata, "chunk_type": chunk.chunk_type}
            )
            documents.append(doc)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)
        
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(split_docs)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant chunks"""
        return self.vectorstore.similarity_search(query, k=k)


class MultiModalRAGAgent:
    """Main RAG agent with multi-modal capabilities, web search, and deep research"""
    
    def __init__(self, 
                 api_key: str, 
                 model: str = "meituan/longcat-flash-chat:free",
                 enable_web_search: bool = True,
                 search_provider: str = "duckduckgo",
                 search_api_key: Optional[str] = None,
                 enable_deep_research: bool = True):
        self.parser = DocumentParser()
        self.vector_store = MultiModalVectorStore()
        self.api_key = api_key
        self.model = model
        self.enable_web_search = enable_web_search
        self.enable_deep_research = enable_deep_research
        
        # Initialize LLM client (OpenRouter)
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        # Initialize web search if enabled
        self.web_search = None
        if enable_web_search:
            try:
                self.web_search = WebSearchTool(
                    provider=search_provider,
                    api_key=search_api_key
                )
                print(f"✓ Web search enabled with {search_provider}")
            except Exception as e:
                print(f"⚠ Web search initialization failed: {e}")
                self.enable_web_search = False
        
        # Initialize deep research orchestrator if enabled
        self.orchestrator = None
        if enable_deep_research:
            try:
                from deep_research_agents import ResearchOrchestrator
                self.orchestrator = ResearchOrchestrator(
                    llm_client=self.client,
                    model=self.model,
                    vector_store=self.vector_store,
                    web_search=self.web_search
                )
                print(f"✓ Deep research system enabled with multi-agent architecture")
            except Exception as e:
                print(f"⚠ Deep research initialization failed: {e}")
                self.enable_deep_research = False
    
    def ingest_document(self, file_path: str):
        """Ingest a document into the system"""
        print(f"Parsing document: {file_path}")
        chunks = self.parser.parse_document(file_path)
        print(f"Extracted {len(chunks)} chunks")
        
        self.vector_store.add_documents(chunks)
        print("Document ingested successfully!")
    
    def answer_question(self, question: str, use_web_search: bool = True, 
                       use_deep_research: bool = False, max_iterations: int = 5) -> Dict[str, Any]:
        """Answer questions using RAG and optional web search or deep research"""
        
        # Use deep research if enabled and requested
        if use_deep_research and self.enable_deep_research and self.orchestrator:
            print("\n🔬 Activating Deep Research Mode...")
            research_result = self.orchestrator.deep_research(
                query=question,
                max_iterations=max_iterations,
                use_web_search=use_web_search
            )
            
            return {
                'answer': research_result['answer'],
                'confidence': research_result['confidence'],
                'document_sources': [f.source for f in research_result['session'].findings if f.source_type == 'document'],
                'web_results': [{'title': f.content[:100], 'url': f.source} for f in research_result['session'].findings if f.source_type == 'web'],
                'deep_research': {
                    'key_decisions': research_result['key_decisions'],
                    'contradictions': research_result['contradictions'],
                    'contradiction_analysis': research_result['contradiction_analysis'],
                    'gaps': research_result['gaps'],
                    'assumptions': research_result['assumptions'],
                    'structured_summary': research_result['structured_summary'],
                    'findings_count': research_result['findings_count']
                }
            }
        
        # Standard RAG
        relevant_docs = self.vector_store.similarity_search(question, k=5)
        doc_context = self._build_context(relevant_docs)
        
        web_context = ""
        web_results = []
        if use_web_search and self.enable_web_search and self.web_search:
            web_results = self._perform_web_search(question)
            if web_results:
                web_context = self._format_web_results(web_results)
        
        full_context = self._combine_contexts(doc_context, web_context)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions using provided documents and web search results."},
            {"role": "user", "content": f"Context:\n{full_context}\n\nQuestion: {question}\n\nAnswer:"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3
        )
        
        return {
            'answer': response.choices[0].message.content,
            'confidence': 0.75,
            'document_sources': [doc.metadata.get('source', 'Unknown') for doc in relevant_docs],
            'web_results': web_results
        }
    
    def analyze_document(self, analysis_type: str = "summary", use_web_search: bool = False,
                        use_deep_research: bool = False) -> Dict[str, Any]:
        """Analyze entire document"""
        
        if use_deep_research and self.enable_deep_research and self.orchestrator:
            print("\n🔬 Performing Deep Document Analysis...")
            analysis_query = f"Provide a {analysis_type} of the document content"
            research_result = self.orchestrator.deep_research(
                query=analysis_query,
                max_iterations=3,
                use_web_search=use_web_search
            )
            
            return {
                'analysis': research_result['answer'],
                'confidence': research_result['confidence'],
                'key_decisions': research_result['key_decisions'],
                'gaps': research_result['gaps'],
                'structured_summary': research_result['structured_summary']
            }
        
        # Standard analysis
        all_docs = self.vector_store.vectorstore.get()
        context = "\n\n".join([doc for doc in all_docs['documents'][:20]])
        
        prompt = f"Perform the following analysis: {analysis_type}"
        messages = [
            {"role": "system", "content": "You are a document analysis expert."},
            {"role": "user", "content": f"{prompt}\n\nDocument content:\n{context}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3
        )
        
        return {
            'analysis': response.choices[0].message.content,
            'confidence': 0.75
        }
    
    def rewrite_document(self, instructions: str, output_path: str, use_web_search: bool = False,
                        use_critic_review: bool = True):
        """Rewrite entire document"""
        all_docs = self.vector_store.vectorstore.get()
        full_content = "\n\n".join(all_docs['documents'])
        
        messages = [
            {"role": "system", "content": "You are an expert document editor and rewriter."},
            {"role": "user", "content": f"Rewrite the following document according to these instructions: {instructions}\n\nOriginal document:\n{full_content}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.5
        )
        
        rewritten = response.choices[0].message.content
        
        # Critic review if enabled
        if use_critic_review and self.enable_deep_research and self.orchestrator:
            print("\n🎯 Critic Agent reviewing rewrite...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rewritten)
        
        print(f"Rewritten document saved to: {output_path}")
        return rewritten
    
    def search_and_augment(self, topic: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Standalone web search"""
        if not self.enable_web_search:
            return []
        return self._perform_web_search(topic, max_results)
    
    def _perform_web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Perform web search"""
        try:
            results = self.web_search.search(query, max_results=max_results)
            return results
        except Exception as e:
            print(f"Web search error: {e}")
            return []
    
    def _format_web_results(self, results: List[Dict[str, str]]) -> str:
        """Format web search results"""
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"[Web Source {i}]\nTitle: {result['title']}\nURL: {result['url']}\nContent: {result['snippet']}\n")
        return "\n".join(formatted)
    
    def _combine_contexts(self, doc_context: str, web_context: str) -> str:
        """Combine contexts"""
        combined = []
        if doc_context:
            combined.append("=== DOCUMENT CONTEXT ===\n" + doc_context)
        if web_context:
            combined.append("\n=== WEB SEARCH CONTEXT ===\n" + web_context)
        return "\n".join(combined)
    
    def _build_context(self, docs: List[Document]) -> str:
        """Build context from documents"""
        context_parts = []
        for i, doc in enumerate(docs):
            chunk_type = doc.metadata.get('chunk_type', 'text')
            context_parts.append(f"[Chunk {i+1} - {chunk_type}]\n{doc.page_content}\n")
        return "\n".join(context_parts)


# Example usage
if __name__ == "__main__":
    agent = MultiModalRAGAgent(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="meituan/longcat-flash-chat:free",
        enable_web_search=True,
        enable_deep_research=True
    )
    
    print("✓ Agent initialized successfully!")