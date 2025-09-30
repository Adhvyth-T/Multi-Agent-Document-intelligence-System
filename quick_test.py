"""
Quick Test Script for Multi-Modal RAG Agent with Web Search
Run this to verify your installation and configuration
"""

import os
import sys
from dotenv import load_dotenv

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = {
        'langchain': 'LangChain',
        'chromadb': 'ChromaDB',
        'sentence_transformers': 'Sentence Transformers',
        'openai': 'OpenAI',
        'duckduckgo_search': 'DuckDuckGo Search',
        'unstructured': 'Unstructured',
        'docx': 'python-docx',
        'streamlit': 'Streamlit'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✓ All dependencies installed!\n")
    return True


def check_environment():
    """Check environment configuration"""
    print("🔍 Checking environment configuration...")
    
    load_dotenv()
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        print(f"  ✓ OPENROUTER_API_KEY found ({openrouter_key[:10]}...)")
    else:
        print("  ✗ OPENROUTER_API_KEY not found")
        print("    Get your key from: https://openrouter.ai/keys")
        return False
    
    print("✓ Environment configured!\n")
    return True


def test_basic_initialization():
    """Test basic agent initialization"""
    print("🔍 Testing basic initialization...")
    
    try:
        from multimodal_rag import MultiModalRAGAgent
        
        agent = MultiModalRAGAgent(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="meituan/longcat-flash-chat:free",
            enable_web_search=False  # Start without web search
        )
        print("  ✓ Agent initialized successfully")
        return True, agent
    except Exception as e:
        print(f"  ✗ Initialization failed: {e}")
        return False, None


def test_web_search():
    """Test web search functionality"""
    print("\n🔍 Testing web search...")
    
    try:
        from multimodal_rag import MultiModalRAGAgent, WebSearchTool
        
        # Test DuckDuckGo (free)
        print("  Testing DuckDuckGo (free)...")
        search_tool = WebSearchTool(provider="duckduckgo")
        results = search_tool.search("Python programming", max_results=2)
        
        if results:
            print(f"  ✓ DuckDuckGo working - Found {len(results)} results")
            print(f"    Example: {results[0]['title']}")
        else:
            print("  ⚠ DuckDuckGo returned no results (might be rate limited)")
        
        return True
    except Exception as e:
        print(f"  ✗ Web search test failed: {e}")
        return False


def test_full_agent_with_web():
    """Test full agent with web search"""
    print("\n🔍 Testing full agent with web search...")
    
    try:
        from multimodal_rag import MultiModalRAGAgent
        
        agent = MultiModalRAGAgent(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="meituan/longcat-flash-chat:free",
            enable_web_search=True,
            search_provider="duckduckgo"
        )
        
        print("  ✓ Agent with web search initialized")
        
        # Test standalone web search
        print("  Testing standalone web search...")
        results = agent.search_and_augment("latest AI news", max_results=2)
        
        if results:
            print(f"  ✓ Web search working - Found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"    {i}. {result['title'][:60]}...")
        else:
            print("  ⚠ No results returned (might be rate limited)")
        
        return True
    except Exception as e:
        print(f"  ✗ Full agent test failed: {e}")
        return False


def test_llm_connection():
    """Test LLM connection"""
    print("\n🔍 Testing LLM connection...")
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        response = client.chat.completions.create(
            model="meituan/longcat-flash-chat:free",
            messages=[
                {"role": "user", "content": "Say 'Hello' in one word"}
            ],
            max_tokens=10
        )
        
        reply = response.choices[0].message.content
        print(f"  ✓ LLM connection working")
        print(f"    Response: {reply}")
        return True
    except Exception as e:
        print(f"  ✗ LLM connection failed: {e}")
        return False


def interactive_test():
    """Run an interactive test"""
    print("\n" + "="*60)
    print("🎯 Interactive Test")
    print("="*60)
    
    try:
        from multimodal_rag import MultiModalRAGAgent
        
        agent = MultiModalRAGAgent(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="meituan/longcat-flash-chat:free",
            enable_web_search=True,
            search_provider="duckduckgo"
        )
        
        print("\nAgent ready! You can now:")
        print("1. Test web search")
        print("2. Test Q&A (without documents)")
        print("3. Exit")
        
        while True:
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == "1":
                query = input("Enter search query: ").strip()
                if query:
                    print("\n🔍 Searching...")
                    results = agent.search_and_augment(query, max_results=3)
                    if results:
                        print(f"\nFound {len(results)} results:")
                        for i, r in enumerate(results, 1):
                            print(f"\n{i}. {r['title']}")
                            print(f"   {r['url']}")
                            print(f"   {r['snippet'][:100]}...")
                    else:
                        print("No results found.")
            
            elif choice == "2":
                question = input("Enter question: ").strip()
                if question:
                    print("\n💭 Thinking...")
                    result = agent.answer_question(question, use_web_search=True)
                    print(f"\n📝 Answer:\n{result['answer']}")
                    if result['web_results']:
                        print(f"\n🌐 Used {len(result['web_results'])} web sources")
            
            elif choice == "3":
                print("\n👋 Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-3.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Multi-Modal RAG Agent - Quick Test Script")
    print("="*60 + "\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Test basic initialization
    success, agent = test_basic_initialization()
    if not success:
        sys.exit(1)
    
    # Test LLM connection
    if not test_llm_connection():
        print("\n⚠ Warning: LLM connection issues. Check your API key and internet connection.")
    
    # Test web search
    if not test_web_search():
        print("\n⚠ Warning: Web search issues. Will continue without web search.")
    
    # Test full agent with web
    test_full_agent_with_web()
    
    # Summary
    print("\n" + "="*60)
    print("✓ Basic tests completed!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Run the Streamlit UI: streamlit run app.py")
    print("2. Try the examples: python usage_examples.py")
    print("3. Run interactive test (below)")
    
    # Ask if user wants interactive test
    response = input("\nWould you like to run the interactive test? (y/n): ").strip().lower()
    if response == 'y':
        interactive_test()
    else:
        print("\n👋 Test complete! Your agent is ready to use.")


if __name__ == "__main__":
    main()