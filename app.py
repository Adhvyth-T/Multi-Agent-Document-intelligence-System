"""
Streamlit UI for Multi-Modal RAG Agent with Deep Research
Run with: streamlit run app.py
"""

import streamlit as st
import os
from pathlib import Path
import tempfile
from multimodal_rag import MultiModalRAGAgent

# Page config
st.set_page_config(
    page_title="Multi-Modal RAG Agent",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Configuration
st.sidebar.title("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "OpenRouter API Key",
    type="password",
    help="Enter your OpenRouter API key from openrouter.ai"
)

model_choice = st.sidebar.selectbox(
    "Model",
    [
        "meituan/longcat-flash-chat:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "google/gemini-flash-1.5:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "qwen/qwen-2-7b-instruct:free"
    ],
    help="Select the LLM model to use"
)

# Web search configuration
st.sidebar.markdown("### 🌐 Web Search Settings")
enable_web_search = st.sidebar.checkbox("Enable Web Search", value=True)

search_provider = st.sidebar.selectbox(
    "Search Provider",
    ["duckduckgo", "tavily", "serpapi", "serper"],
    help="DuckDuckGo is free and requires no API key"
)

search_api_key = None
if search_provider != "duckduckgo":
    search_api_key = st.sidebar.text_input(
        f"{search_provider.title()} API Key",
        type="password",
        help=f"Required for {search_provider}"
    )

# Deep Research configuration
st.sidebar.markdown("### 🔬 Deep Research Settings")
enable_deep_research = st.sidebar.checkbox(
    "Enable Deep Research", 
    value=True,
    help="Multi-agent system with Explorer, FactChecker, Synthesizer, and Critic agents"
)

research_depth = 5
if enable_deep_research:
    research_depth = st.sidebar.slider(
        "Research Depth",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of research iterations (more = thorough but slower)"
    )

# Initialize agent
if api_key and st.session_state.agent is None:
    try:
        st.session_state.agent = MultiModalRAGAgent(
            api_key=api_key,
            model=model_choice,
            enable_web_search=enable_web_search,
            search_provider=search_provider,
            search_api_key=search_api_key,
            enable_deep_research=enable_deep_research
        )
        st.sidebar.success("✅ Agent initialized!")
        if enable_deep_research:
            st.sidebar.info("🔬 Deep research agents ready")
    except Exception as e:
        st.sidebar.error(f"❌ Initialization error: {str(e)}")

# Main interface
st.title("📚 Multi-Modal RAG Agent with Deep Research")
st.markdown("Analyze, query, and rewrite documents with multi-agent research system")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload Documents",
    "💬 Chat & Query",
    "📊 Analyze",
    "✏️ Rewrite"
])

# Tab 1: Upload Documents
with tab1:
    st.header("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'md'],
        help="Upload PDF, DOCX, TXT, or MD files"
    )
    
    if uploaded_files and st.session_state.agent:
        if st.button("🔄 Process Documents"):
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        st.session_state.agent.ingest_document(tmp_path)
                        st.session_state.documents.append(uploaded_file.name)
                        st.success(f"✅ Processed: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        os.unlink(tmp_path)
    
    if st.session_state.documents:
        st.subheader("📁 Loaded Documents")
        for doc in st.session_state.documents:
            st.write(f"- {doc}")

# Tab 2: Chat & Query
with tab2:
    st.header("Chat with Your Documents")
    
    if not st.session_state.agent:
        st.warning("⚠️ Please configure API key in the sidebar first")
    elif not st.session_state.documents:
        st.info("ℹ️ Upload documents in the first tab to start chatting")
    else:
        # Research mode toggles
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            use_web = st.checkbox("🌐 Web Search", value=True, key="chat_web_search")
        with col3:
            use_deep = st.checkbox("🔬 Deep Research", value=False, key="chat_deep_research",
                                  help="Multi-agent deep research (slower but thorough)")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                if message["role"] == "assistant" and "metadata" in message:
                    metadata = message["metadata"]
                    
                    # Confidence indicator
                    if "confidence" in metadata:
                        confidence = int(metadata["confidence"] * 100)
                        st.progress(confidence / 100, text=f"Confidence: {confidence}%")
                    
                    # Sources and research details
                    with st.expander("📚 Sources & Research Details"):
                        if metadata.get("document_sources"):
                            st.write("**Document Sources:**")
                            for src in metadata["document_sources"]:
                                st.write(f"- {src}")
                        
                        if metadata.get("web_results"):
                            st.write("\n**Web Sources:**")
                            for i, result in enumerate(metadata["web_results"][:5], 1):
                                if isinstance(result, dict):
                                    st.markdown(f"{i}. [{result.get('title', 'N/A')}]({result.get('url', '#')})")
                                    if 'snippet' in result:
                                        st.caption(result['snippet'][:150] + "...")
                        
                        # Deep research insights
                        if "deep_research" in metadata and metadata["deep_research"]:
                            dr = metadata["deep_research"]
                            
                            st.markdown("---")
                            st.markdown("### 🔬 Deep Research Insights")
                            
                            # Key decisions
                            if dr.get("key_decisions"):
                                st.write("**Key Decisions:**")
                                for decision in dr["key_decisions"][:5]:
                                    st.write(f"🤖 **{decision['agent']}**: {decision['decision']}")
                                    with st.expander(f"View reasoning"):
                                        st.write(decision['reasoning'])
                                        st.caption(f"Confidence: {int(decision['confidence']*100)}%")
                            
                            # Contradictions
                            if dr.get("contradictions"):
                                st.warning(f"⚠️ **{len(dr['contradictions'])} Contradiction(s) Found**")
                                for contra in dr["contradictions"][:3]:
                                    st.write(f"- {contra.get('description', 'N/A')}")
                            
                            # Research gaps
                            if dr.get("gaps"):
                                st.info(f"❓ **Research Gaps:**")
                                for gap in dr["gaps"][:5]:
                                    st.write(f"- {gap}")
                            
                            # Structured summary
                            if dr.get("structured_summary"):
                                with st.expander("📊 Structured Summary"):
                                    summary = dr["structured_summary"]
                                    if summary.get("key_points"):
                                        st.write("**Key Points:**")
                                        for point in summary["key_points"]:
                                            st.write(f"• {point}")
                                    st.write(f"\n**Evidence Strength:** {summary.get('evidence_strength', 'unknown')}")
                                    st.write(f"**Consensus Level:** {summary.get('consensus_level', 'unknown')}")
        
        # Chat input
        question = st.chat_input("Ask a question about your documents...")
        
        if question:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            
            with st.chat_message("user"):
                st.write(question)
            
            # Get answer
            with st.chat_message("assistant"):
                status_text = "Thinking..."
                if use_deep:
                    status_text = "🔬 Deep research in progress (1-3 minutes)..."
                elif use_web:
                    status_text = "🌐 Searching web..."
                
                with st.spinner(status_text):
                    try:
                        result = st.session_state.agent.answer_question(
                            question, 
                            use_web_search=use_web,
                            use_deep_research=use_deep,
                            max_iterations=research_depth if enable_deep_research else 5
                        )
                        
                        st.write(result['answer'])
                        
                        # Show confidence
                        if 'confidence' in result:
                            confidence = int(result['confidence'] * 100)
                            st.progress(confidence / 100, text=f"Confidence: {confidence}%")
                        
                        # Show sources
                        with st.expander("📚 Sources & Research Details"):
                            if result.get("document_sources"):
                                st.write("**Document Sources:**")
                                for src in result["document_sources"]:
                                    st.write(f"- {src}")
                            
                            if result.get("web_results"):
                                st.write("\n**Web Sources:**")
                                for i, web_result in enumerate(result["web_results"][:5], 1):
                                    if isinstance(web_result, dict):
                                        st.markdown(f"{i}. [{web_result.get('title', 'N/A')}]({web_result.get('url', '#')})")
                                        if 'snippet' in web_result:
                                            st.caption(web_result['snippet'][:150] + "...")
                            
                            # Deep research insights
                            if "deep_research" in result and result["deep_research"]:
                                dr = result["deep_research"]
                                
                                st.markdown("---")
                                st.markdown("### 🔬 Deep Research Insights")
                                
                                if dr.get("key_decisions"):
                                    st.write("**Key Decisions:**")
                                    for decision in dr["key_decisions"][:5]:
                                        st.write(f"🤖 **{decision['agent']}**: {decision['decision']}")
                                        with st.expander(f"View reasoning"):
                                            st.write(decision['reasoning'])
                                            st.caption(f"Confidence: {int(decision['confidence']*100)}%")
                                
                                if dr.get("contradictions"):
                                    st.warning(f"⚠️ **{len(dr['contradictions'])} Contradiction(s) Found**")
                                    for contra in dr["contradictions"][:3]:
                                        st.write(f"- {contra.get('description', 'N/A')}")
                                
                                if dr.get("gaps"):
                                    st.info(f"❓ **Research Gaps:**")
                                    for gap in dr["gaps"][:5]:
                                        st.write(f"- {gap}")
                                
                                if dr.get("structured_summary"):
                                    with st.expander("📊 Structured Summary"):
                                        summary = dr["structured_summary"]
                                        if summary.get("key_points"):
                                            st.write("**Key Points:**")
                                            for point in summary["key_points"]:
                                                st.write(f"• {point}")
                                        st.write(f"\n**Evidence:** {summary.get('evidence_strength', 'unknown')}")
                                        st.write(f"**Consensus:** {summary.get('consensus_level', 'unknown')}")
                        
                        # Save to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result['answer'],
                            "metadata": {
                                "confidence": result.get("confidence", 0.75),
                                "document_sources": result.get("document_sources", []),
                                "web_results": result.get("web_results", []),
                                "deep_research": result.get("deep_research")
                            }
                        })
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Tab 3: Analyze
with tab3:
    st.header("Document Analysis")
    
    if not st.session_state.agent:
        st.warning("⚠️ Please configure API key in the sidebar first")
    elif not st.session_state.documents:
        st.info("ℹ️ Upload documents in the first tab to analyze")
    else:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                [
                    "Summary",
                    "Key Points Extraction",
                    "Structure Analysis",
                    "Topic Identification",
                    "Statistics & Numbers",
                    "Implications & Impact",
                    "Custom Analysis"
                ]
            )
        
        with col2:
            use_web_analysis = st.checkbox("🌐 Web Search", value=False)
        
        with col3:
            use_deep_analysis = st.checkbox("🔬 Deep Research", value=False)
        
        custom_instruction = ""
        if analysis_type == "Custom Analysis":
            custom_instruction = st.text_area(
                "Describe your analysis",
                placeholder="e.g., Identify all technical terms and provide definitions"
            )
        
        if st.button("🔍 Analyze"):
            status_text = "Analyzing..."
            if use_deep_analysis:
                status_text = "🔬 Deep analysis in progress..."
            
            with st.spinner(status_text):
                try:
                    analysis_map = {
                        "Summary": "summary",
                        "Key Points Extraction": "key_points",
                        "Structure Analysis": "structure",
                        "Topic Identification": "topics",
                        "Statistics & Numbers": "statistics",
                        "Implications & Impact": "implications and impact analysis",
                        "Custom Analysis": custom_instruction
                    }
                    
                    result = st.session_state.agent.analyze_document(
                        analysis_type=analysis_map[analysis_type],
                        use_web_search=use_web_analysis,
                        use_deep_research=use_deep_analysis
                    )
                    
                    st.subheader("Analysis Results")
                    
                    if isinstance(result, dict) and 'confidence' in result:
                        confidence = int(result['confidence'] * 100)
                        st.progress(confidence / 100, text=f"Confidence: {confidence}%")
                    
                    analysis_text = result['analysis'] if isinstance(result, dict) else result
                    st.markdown(analysis_text)
                    
                    if isinstance(result, dict) and 'key_decisions' in result:
                        with st.expander("🔬 Deep Research Insights"):
                            st.markdown("### Key Decisions")
                            for decision in result['key_decisions'][:5]:
                                st.write(f"**{decision['agent']}**: {decision['decision']}")
                                st.caption(decision['reasoning'])
                            
                            if result.get('gaps'):
                                st.markdown("### Gaps Identified")
                                for gap in result['gaps']:
                                    st.write(f"• {gap}")
                    
                    st.download_button(
                        label="📥 Download Analysis",
                        data=analysis_text,
                        file_name="document_analysis.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# Tab 4: Rewrite
with tab4:
    st.header("Rewrite Documents")
    
    if not st.session_state.agent:
        st.warning("⚠️ Please configure API key in the sidebar first")
    elif not st.session_state.documents:
        st.info("ℹ️ Upload documents in the first tab to rewrite")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rewrite_preset = st.selectbox(
                "Quick Preset",
                [
                    "Custom",
                    "Make more concise",
                    "Expand with details",
                    "Convert to bullet points",
                    "Simplify language (ELI5)",
                    "Make more formal",
                    "Make more casual"
                ]
            )
        
        with col2:
            output_format = st.selectbox("Output Format", ["Markdown", "Plain Text", "HTML"])
        
        with col3:
            use_web_rewrite = st.checkbox("🌐 Web", value=False)
        
        use_critic_review = st.checkbox("🎯 Critic Review", value=True, 
                                       help="Critic agent reviews rewrite")
        
        if rewrite_preset == "Custom":
            instructions = st.text_area("Custom Instructions", 
                                       placeholder="e.g., Rewrite as a blog post...",
                                       height=150)
        else:
            instructions = rewrite_preset
            st.info(f"Using preset: {rewrite_preset}")
        
        if st.button("✏️ Rewrite Document"):
            if not instructions:
                st.error("Please provide rewriting instructions")
            else:
                with st.spinner("Rewriting..."):
                    try:
                        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w')
                        output_path = output_file.name
                        output_file.close()
                        
                        full_instructions = f"{instructions}. Format as {output_format}."
                        
                        rewritten = st.session_state.agent.rewrite_document(
                            instructions=full_instructions,
                            output_path=output_path,
                            use_web_search=use_web_rewrite,
                            use_critic_review=use_critic_review
                        )
                        
                        st.subheader("Rewritten Document")
                        st.markdown(rewritten)
                        
                        st.download_button(
                            label="📥 Download",
                            data=rewritten,
                            file_name="rewritten_document.txt",
                            mime="text/plain"
                        )
                        
                        os.unlink(output_path)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### Multi-Modal RAG v3.0 🔬
    
    **🔬 Deep Research Agents:**
    - 📡 Explorer: Discovers topics
    - ✓ Fact-Checker: Verifies claims  
    - 🔄 Synthesizer: Combines sources
    - 🎯 Critic: Identifies gaps
    """
)

# Agent status
if st.session_state.agent:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Active Agents")
    st.sidebar.success("✓ Base RAG Agent")
    if st.session_state.agent.enable_web_search:
        st.sidebar.success("✓ Web Search")
    if st.session_state.agent.enable_deep_research:
        st.sidebar.success("✓ Deep Research (4 agents)")