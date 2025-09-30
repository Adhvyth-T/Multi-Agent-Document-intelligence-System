"""
Multi-Modal RAG Agent - Usage Examples
Complete examples showing all features including web search
"""

import os
from dotenv import load_dotenv
from multimodal_rag import MultiModalRAGAgent

# Load environment variables
load_dotenv()

# ============================================================================
# Example 1: Basic Setup with Free Web Search (DuckDuckGo)
# ============================================================================

def example_basic_setup():
    """Initialize agent with free web search"""
    print("=" * 60)
    print("Example 1: Basic Setup with Free Web Search")
    print("=" * 60)
    
    agent = MultiModalRAGAgent(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="meituan/longcat-flash-chat:free",
        enable_web_search=True,
        search_provider="duckduckgo"  # FREE - No API key needed!
    )
    
    print("✓ Agent initialized with free web search!")
    return agent


# ============================================================================
# Example 2: Document Ingestion and Basic Q&A
# ============================================================================

def example_document_qa(agent):
    """Ingest documents and ask questions"""
    print("\n" + "=" * 60)
    print("Example 2: Document Q&A")
    print("=" * 60)
    
    # Ingest a document
    print("\n📄 Ingesting document...")
    agent.ingest_document("sample_research_paper.pdf")
    
    # Ask a question without web search
    print("\n💬 Question (Document only):")
    result = agent.answer_question(
        "What are the main contributions of this paper?",
        use_web_search=False
    )
    print(f"Answer: {result['answer']}")
    print(f"\nSources: {result['document_sources']}")


# ============================================================================
# Example 3: Q&A with Web Search Enhancement
# ============================================================================

def example_web_enhanced_qa(agent):
    """Ask questions with web search for additional context"""
    print("\n" + "=" * 60)
    print("Example 3: Web-Enhanced Q&A")
    print("=" * 60)
    
    # Ask a question with web search enabled
    print("\n💬 Question (Document + Web):")
    result = agent.answer_question(
        "How does this approach compare to current industry standards?",
        use_web_search=True
    )
    
    print(f"\nAnswer: {result['answer']}")
    
    print(f"\n📚 Document Sources:")
    for src in result['document_sources']:
        print(f"  - {src}")
    
    print(f"\n🌐 Web Sources:")
    for i, web_result in enumerate(result['web_results'], 1):
        print(f"  {i}. {web_result['title']}")
        print(f"     {web_result['url']}")
        print(f"     {web_result['snippet'][:100]}...\n")


# ============================================================================
# Example 4: Standalone Web Search
# ============================================================================

def example_standalone_search(agent):
    """Use web search without documents"""
    print("\n" + "=" * 60)
    print("Example 4: Standalone Web Search")
    print("=" * 60)
    
    # Search for recent information
    topics = [
        "latest AI research 2024",
        "multimodal AI applications",
        "RAG systems best practices"
    ]
    
    for topic in topics:
        print(f"\n🔍 Searching: {topic}")
        results = agent.search_and_augment(topic, max_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   {result['snippet'][:150]}...")


# ============================================================================
# Example 5: Document Analysis with Web Context
# ============================================================================

def example_web_enhanced_analysis(agent):
    """Analyze documents with web-enhanced context"""
    print("\n" + "=" * 60)
    print("Example 5: Web-Enhanced Document Analysis")
    print("=" * 60)
    
    # Analyze without web
    print("\n📊 Analysis (Document only):")
    summary = agent.analyze_document(
        analysis_type="summary",
        use_web_search=False
    )
    print(summary[:300] + "...\n")
    
    # Analyze with web enhancement
    print("\n📊 Analysis (Document + Web context):")
    enhanced_summary = agent.analyze_document(
        analysis_type="summary",
        use_web_search=True
    )
    print(enhanced_summary[:300] + "...\n")


# ============================================================================
# Example 6: Document Rewriting with Web Enhancement
# ============================================================================

def example_web_enhanced_rewriting(agent):
    """Rewrite documents using web-sourced best practices"""
    print("\n" + "=" * 60)
    print("Example 6: Web-Enhanced Document Rewriting")
    print("=" * 60)
    
    # Rewrite with current trends and best practices
    print("\n✏️ Rewriting document with latest trends...")
    
    rewritten = agent.rewrite_document(
        instructions="Rewrite as a modern blog post, incorporating latest industry trends and best practices",
        output_path="rewritten_with_web_context.txt",
        use_web_search=True  # Use web to find current trends
    )
    
    print(f"\n✓ Document rewritten with web-enhanced context!")
    print(f"Preview:\n{rewritten[:400]}...\n")


# ============================================================================
# Example 7: Advanced Use Cases
# ============================================================================

def example_advanced_use_cases(agent):
    """Advanced scenarios combining multiple features"""
    print("\n" + "=" * 60)
    print("Example 7: Advanced Use Cases")
    print("=" * 60)
    
    # Use case 1: Research assistant
    print("\n📚 Use Case 1: Research Assistant")
    print("Comparing document findings with latest research...")
    
    result = agent.answer_question(
        "What recent developments relate to the methodology described in this paper?",
        use_web_search=True
    )
    print(f"Answer: {result['answer'][:200]}...\n")
    
    # Use case 2: Fact-checking
    print("\n✓ Use Case 2: Fact-Checking with Web Sources")
    result = agent.answer_question(
        "Verify the statistics mentioned in the document against current data",
        use_web_search=True
    )
    print(f"Answer: {result['answer'][:200]}...\n")
    
    # Use case 3: Context-aware summarization
    print("\n📝 Use Case 3: Context-Aware Summarization")
    summary = agent.analyze_document(
        analysis_type="Summarize the document and highlight how it relates to current industry trends",
        use_web_search=True
    )
    print(f"Summary: {summary[:200]}...\n")


# ============================================================================
# Example 8: Different Search Providers
# ============================================================================

def example_different_search_providers():
    """Compare different search providers"""
    print("\n" + "=" * 60)
    print("Example 8: Different Search Providers")
    print("=" * 60)
    
    providers = [
        ("duckduckgo", None),  # Free
        # Uncomment if you have API keys:
        # ("tavily", os.getenv("TAVILY_API_KEY")),
        # ("serpapi", os.getenv("SERPAPI_API_KEY")),
    ]
    
    for provider, api_key in providers:
        print(f"\n🔍 Testing {provider}...")
        try:
            agent = MultiModalRAGAgent(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                model="meituan/longcat-flash-chat:free",
                enable_web_search=True,
                search_provider=provider,
                search_api_key=api_key
            )
            
            results = agent.search_and_augment("AI news", max_results=2)
            print(f"✓ {provider}: Found {len(results)} results")
            for r in results:
                print(f"  - {r['title']}")
        except Exception as e:
            print(f"✗ {provider}: {e}")


# ============================================================================
# Example 9: Multi-Document Analysis with Web Context
# ============================================================================

def example_multi_document_analysis():
    """Analyze multiple documents with web enhancement"""
    print("\n" + "=" * 60)
    print("Example 9: Multi-Document Analysis")
    print("=" * 60)
    
    agent = MultiModalRAGAgent(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="meituan/longcat-flash-chat:free",
        enable_web_search=True,
        search_provider="duckduckgo"
    )
    
    # Ingest multiple documents
    documents = [
        "research_paper_1.pdf",
        "technical_report.docx",
        "industry_analysis.txt"
    ]
    
    print("\n📚 Ingesting multiple documents...")
    for doc in documents:
        if os.path.exists(doc):
            agent.ingest_document(doc)
            print(f"  ✓ {doc}")
    
    # Cross-document analysis with web context
    print("\n🔍 Performing cross-document analysis...")
    result = agent.answer_question(
        "Compare the approaches across all documents and relate them to current industry practices",
        use_web_search=True
    )
    print(f"\nAnalysis:\n{result['answer']}")


# ============================================================================
# Example 10: Real-time Information Integration
# ============================================================================

def example_realtime_integration(agent):
    """Integrate real-time web information with documents"""
    print("\n" + "=" * 60)
    print("Example 10: Real-time Information Integration")
    print("=" * 60)
    
    # Questions that benefit from real-time data
    realtime_questions = [
        "What are the latest developments in this field?",
        "How do current market conditions affect these findings?",
        "What are recent criticisms or validations of this approach?",
    ]
    
    for question in realtime_questions:
        print(f"\n💬 {question}")
        result = agent.answer_question(question, use_web_search=True)
        print(f"Answer: {result['answer'][:250]}...")
        print(f"Web sources used: {len(result['web_results'])}\n")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Multi-Modal RAG Agent - Complete Usage Examples")
    print("=" * 60)
    
    # Initialize agent
    agent = example_basic_setup()
    
    # Run examples (uncomment the ones you want to try)
    
"""
Multi-Modal RAG Agent - Usage Examples with Deep Research
Complete examples showing all features including multi-agent deep research
"""

import os
from dotenv import load_dotenv
from multimodal_rag import MultiModalRAGAgent

# Load environment variables
load_dotenv()

# ============================================================================
# Example 1: Basic Setup with Deep Research
# ============================================================================

def example_basic_setup():
    """Initialize agent with free web search and deep research"""
    print("=" * 60)
    print("Example 1: Setup with Deep Research System")
    print("=" * 60)
    
    agent = MultiModalRAGAgent(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="meituan/longcat-flash-chat:free",
        enable_web_search=True,
        search_provider="duckduckgo",  # FREE!
        enable_deep_research=True  # Enable multi-agent system
    )
    
    print("✓ Agent initialized")
    print("✓ Web search enabled (DuckDuckGo)")
    print("✓ Deep research system active:")
    print("  - Explorer Agent")
    print("  - Fact-Checker Agent")
    print("  - Synthesizer Agent")
    print("  - Critic Agent")
    return agent


# ============================================================================
# Example 2: Standard vs Deep Research Comparison
# ============================================================================

def example_compare_modes(agent):
    """Compare standard and deep research modes"""
    print("\n" + "=" * 60)
    print("Example 2: Standard vs Deep Research")
    print("=" * 60)
    
    question = "What are the key implications of AI in healthcare?"
    
    # Standard mode
    print("\n📝 STANDARD MODE:")
    result_standard = agent.answer_question(
        question,
        use_web_search=True,
        use_deep_research=False
    )
    print(f"Answer length: {len(result_standard['answer'])} chars")
    print(f"Confidence: {int(result_standard['confidence']*100)}%")
    print(f"Sources: {len(result_standard['web_results'])} web results")
    
    # Deep research mode
    print("\n🔬 DEEP RESEARCH MODE:")
    result_deep = agent.answer_question(
        question,
        use_web_search=True,
        use_deep_research=True,
        max_iterations=5
    )
    print(f"Answer length: {len(result_deep['answer'])} chars")
    print(f"Confidence: {int(result_deep['confidence']*100)}%")
    
    if 'deep_research' in result_deep:
        dr = result_deep['deep_research']
        print(f"Findings: {dr['findings_count']}")
        print(f"Contradictions: {len(dr['contradictions'])}")
        print(f"Gaps identified: {len(dr['gaps'])}")
        print(f"\nKey Decisions:")
        for decision in dr['key_decisions'][:3]:
            print(f"  • [{decision['agent']}] {decision['decision']}")


# ============================================================================
# Example 3: Deep Research with Multi-Agent Insights
# ============================================================================

def example_deep_research_insights(agent):
    """Explore deep research insights from all agents"""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Agent Research Insights")
    print("=" * 60)
    
    question = "How does machine learning impact data privacy?"
    
    print(f"\n🔬 Researching: {question}\n")
    
    result = agent.answer_question(
        question,
        use_web_search=True,
        use_deep_research=True,
        max_iterations=5
    )
    
    if 'deep_research' in result:
        dr = result['deep_research']
        
        # Explorer insights
        print("\n📡 EXPLORER AGENT:")
        explorer_decisions = [d for d in dr['key_decisions'] if d['agent'] == 'Explorer']
        for dec in explorer_decisions:
            print(f"  • {dec['decision']}")
            print(f"    Reasoning: {dec['reasoning']}")
        
        # Fact-Checker insights
        print("\n✓ FACT-CHECKER AGENT:")
        checker_decisions = [d for d in dr['key_decisions'] if d['agent'] == 'FactChecker']
        for dec in checker_decisions:
            print(f"  • {dec['decision']}")
        
        if dr['contradictions']:
            print(f"\n  ⚠️ Contradictions found:")
            for contra in dr['contradictions'][:2]:
                print(f"    - {contra.get('description', 'N/A')}")
        
        # Synthesizer insights
        print("\n🔄 SYNTHESIZER AGENT:")
        synth_decisions = [d for d in dr['key_decisions'] if d['agent'] == 'Synthesizer']
        for dec in synth_decisions:
            print(f"  • {dec['decision']}")
        
        if dr.get('structured_summary'):
            summary = dr['structured_summary']
            print(f"\n  Evidence Strength: {summary.get('evidence_strength')}")
            print(f"  Consensus Level: {summary.get('consensus_level')}")
        
        # Critic insights
        print("\n🎯 CRITIC AGENT:")
        critic_decisions = [d for d in dr['key_decisions'] if d['agent'] == 'Critic']
        for dec in critic_decisions:
            print(f"  • {dec['decision']}")
        
        if dr['gaps']:
            print(f"\n  ❓ Gaps identified:")
            for gap in dr['gaps'][:3]:
                print(f"    - {gap}")
        
        if dr['assumptions']:
            print(f"\n  🤔 Assumptions questioned:")
            for assumption in dr['assumptions'][:3]:
                print(f"    - {assumption}")


# ============================================================================
# Example 4: Deep Document Analysis
# ============================================================================

def example_deep_document_analysis(agent):
    """Perform deep analysis on documents"""
    print("\n" + "=" * 60)
    print("Example 4: Deep Document Analysis")
    print("=" * 60)
    
    # Ingest a document
    agent.ingest_document("research_paper.pdf")
    
    print("\n🔬 Performing deep analysis...")
    
    result = agent.analyze_document(
        analysis_type="comprehensive analysis with implications",
        use_web_search=True,
        use_deep_research=True
    )
    
    print("\n📊 Analysis Results:")
    print(result['analysis'][:500] + "...")
    
    if 'key_decisions' in result:
        print(f"\n✓ Analysis Confidence: {int(result['confidence']*100)}%")
        print("\nKey Research Decisions:")
        for decision in result['key_decisions'][:4]:
            print(f"  [{decision['agent']}] {decision['decision']}")


# ============================================================================
# Example 5: Fact-Checking Across Sources
# ============================================================================

def example_fact_checking(agent):
    """Demonstrate fact-checking capabilities"""
    print("\n" + "=" * 60)
    print("Example 5: Cross-Source Fact Checking")
    print("=" * 60)
    
    question = "Verify the claim that AI reduces healthcare costs by 30%"
    
    print(f"\n🔍 Fact-checking: {question}\n")
    
    result = agent.answer_question(
        question,
        use_web_search=True,
        use_deep_research=True,
        max_iterations=5
    )
    
    print(f"Answer:\n{result['answer']}\n")
    
    if 'deep_research' in result:
        dr = result['deep_research']
        
        print(f"✓ Verification Confidence: {int(result['confidence']*100)}%")
        
        if dr['contradictions']:
            print(f"\n⚠️ Contradictions detected:")
            for contra in dr['contradictions']:
                print(f"  - {contra.get('description')}")
            
            print(f"\n📝 Contradiction Analysis:")
            print(dr['contradiction_analysis'][:300] + "...")


# ============================================================================
# Example 6: Research Session Management
# ============================================================================

def example_research_session(agent):
    """Demonstrate research session tracking"""
    print("\n" + "=" * 60)
    print("Example 6: Research Session Management")
    print("=" * 60)
    
    questions = [
        "What is quantum computing?",
        "How does it differ from classical computing?",
        "What are its practical applications?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n🔬 Question {i}: {question}")
        
        result = agent.answer_question(
            question,
            use_web_search=True,
            use_deep_research=True,
            max_iterations=3  # Faster for demo
        )
        
        print(f"Confidence: {int(result['confidence']*100)}%")
        if 'deep_research' in result:
            print(f"Findings: {result['deep_research']['findings_count']}")


# ============================================================================
# Example 7: Critic-Reviewed Document Rewriting
# ============================================================================

def example_critic_reviewed_rewrite(agent):
    """Rewrite document with critic review"""
    print("\n" + "=" * 60)
    print("Example 7: Critic-Reviewed Rewriting")
    print("=" * 60)
    
    agent.ingest_document("old_document.txt")
    
    print("\n✏️ Rewriting with critic review...")
    
    rewritten = agent.rewrite_document(
        instructions="Modernize language and add recent developments",
        output_path="modernized_document.txt",
        use_web_search=True,
        use_critic_review=True
    )
    
    print(f"\n✓ Document rewritten")
    print(f"Preview:\n{rewritten[:300]}...")


# ============================================================================
# Example 8: Complex Research Query
# ============================================================================

def example_complex_research(agent):
    """Handle complex multi-faceted research question"""
    print("\n" + "=" * 60)
    print("Example 8: Complex Research Query")
    print("=" * 60)
    
    complex_question = """
    Compare the environmental impact, cost-effectiveness, and scalability 
    of solar vs nuclear energy for developing countries, considering recent 
    technological advances and policy changes.
    """
    
    print(f"\n🔬 Deep research on complex question...\n")
    
    result = agent.answer_question(
        complex_question,
        use_web_search=True,
        use_deep_research=True,
        max_iterations=7  # More iterations for complex queries
    )
    
    print(f"Answer:\n{result['answer'][:400]}...\n")
    
    if 'deep_research' in result:
        dr = result['deep_research']
        
        print(f"✓ Research Quality:")
        print(f"  Confidence: {int(result['confidence']*100)}%")
        print(f"  Findings: {dr['findings_count']}")
        print(f"  Research iterations: 7")
        
        print(f"\n📊 Structured Summary:")
        if dr.get('structured_summary'):
            summary = dr['structured_summary']
            print(f"  Key points: {len(summary.get('key_points', []))}")
            print(f"  Evidence strength: {summary.get('evidence_strength')}")
            print(f"  Consensus level: {summary.get('consensus_level')}")
        
        print(f"\n❓ Outstanding Questions:")
        for gap in dr['gaps'][:3]:
            print(f"  - {gap}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Multi-Modal RAG Agent - Deep Research Examples")
    print("=" * 60)
    
    # Initialize agent
    agent = example_basic_setup()
    
    # Run examples (uncomment the ones you want to try)
    
    # example_compare_modes(agent)
    # example_deep_research_insights(agent)
    # example_deep_document_analysis(agent)
    # example_fact_checking(agent)
    # example_research_session(agent)
    # example_critic_reviewed_rewrite(agent)
    example_complex_research(agent)
    
    print("\n" + "=" * 60)
    print("✓ Examples completed!")
    print("=" * 60)
    print("\nDeep Research Features:")
    print("- Multi-agent collaboration (Explorer, FactChecker, Synthesizer, Critic)")
    print("- 5+ research iterations per query")
    print("- Cross-source fact verification")
    print("- Contradiction detection")
    print("- Gap analysis")
    print("- Confidence scoring")
    print("\nFor more examples, uncomment the functions above!")