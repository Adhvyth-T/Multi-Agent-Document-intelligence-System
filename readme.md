# 📚 Multi-Agent Document intelligence System with Deep research capabilities

A production-ready RAG (Retrieval-Augmented Generation) agent featuring **4 specialized AI agents** for deep research, multi-modal document processing, and web search integration.

## 🌟 Key Features

### 🔬 **Deep Research System (NEW - Phase 3)**
- **4 Specialized AI Agents:**
  - 📡 **Explorer Agent**: Breaks down complex queries, discovers topics, generates research angles
  - ✓ **Fact-Checker Agent**: Verifies claims across sources, detects contradictions, assesses quality
  - 🔄 **Synthesizer Agent**: Combines multi-source information, resolves conflicts, creates summaries
  - 🎯 **Critic Agent**: Identifies knowledge gaps, questions assumptions, validates reasoning
- **5+ Research Iterations** for thorough investigation
- **85-95% Confidence Scores** (vs 75% for standard mode)
- **Contradiction Detection** with severity analysis
- **Gap Analysis** identifies missing information
- **Clean Decision Display** (shows key decisions, not verbose logs)
- **Structured Summaries** with evidence strength ratings

### 📄 Core Capabilities
- **Multi-format document parsing** (PDF, DOCX, TXT, MD, HTML)
- **Image extraction** and processing from documents
- **Table recognition** and analysis
- **Semantic search** with vector embeddings
- **Interactive Q&A** with context-aware responses
- **Document analysis** (summaries, key points, structure)
- **Document rewriting** with custom instructions

### 🌐 Web Search Integration
- **Real-time web search** to augment document knowledge
- **FREE option available** (DuckDuckGo - no API key needed)
- **Multiple providers** (DuckDuckGo, Tavily, SerpAPI, Serper)
- **Smart query optimization**
- **Source attribution** with URLs and snippets
- **Context fusion** combining document and web information

### 🎨 Streamlit Web Interface
- Beautiful, intuitive UI
- Drag-and-drop document upload
- Deep research toggles
- Confidence indicators
- Agent decision visualization
- Export functionality

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

**Step 1: Clone/Download Project**
```bash
cd multimodal-rag-agent
```

**Step 2: Create Virtual Environment**
```bash
python -m venv venv

# Activate on Linux/Mac:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Install System Dependencies (for PDF support)**
```bash
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils

# macOS:
brew install tesseract poppler

# Windows:
# Download Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# Download Poppler: https://blog.alivate.com.au/poppler-windows/
```

**Step 5: Configure Environment**
```bash
# Copy template
cp .env.example .env

# Edit .env and add your API key:
OPENROUTER_API_KEY=your_key_here
```

**Get your FREE API key:** [OpenRouter Keys](https://openrouter.ai/keys)

### Launch

**Option 1: Streamlit UI (Recommended)**
```bash
streamlit run app.py
```
Then:
1. Enter API key in sidebar
2. Enable "🔬 Deep Research" 
3. Upload documents
4. Start chatting!

**Option 2: Python API**
```python
from multimodal_rag import MultiModalRAGAgent

# Initialize with deep research
agent = MultiModalRAGAgent(
    api_key="your_key",
    enable_deep_research=True
)

# Ingest documents
agent.ingest_document("document.pdf")

# Standard Q&A (fast - 10 seconds)
result = agent.answer_question("What is this about?")

# Deep Research (thorough - 1-3 minutes)
result = agent.answer_question(
    "What are the implications?",
    use_deep_research=True,
    max_iterations=5
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']*100}%")
```

**Option 3: Run Examples**
```bash
python usage_examples.py
```

## 📊 Standard vs Deep Research Comparison

| Feature | Standard Mode | Deep Research Mode |
|---------|---------------|-------------------|
| **Speed** | ~10 seconds | 1-3 minutes |
| **AI Agents** | 0 | 4 specialized agents |
| **Research Iterations** | 1 pass | 5+ iterations |
| **Confidence Score** | ~75% | 85-95% |
| **Sources Analyzed** | 5-10 | 20-50+ |
| **Fact Verification** | ❌ No | ✅ Yes |
| **Contradiction Detection** | ❌ No | ✅ Yes (with analysis) |
| **Gap Analysis** | ❌ No | ✅ Yes (5-7 gaps identified) |
| **API Calls** | 1-3 | 15-30 |
| **Cost (Free Tier)** | Very low | Low |
| **Best For** | Quick questions | Critical research |

## 🤖 The Four Research Agents Explained

### 📡 Explorer Agent
**Role:** Discovers and expands research scope

**Capabilities:**
- Breaks complex questions into 5+ sub-questions
- Generates alternative search terms
- Identifies key research areas (concepts, stakeholders, contexts)
- Maps timeframes (historical, current, future)

**Example Output:**
```
✓ Generated 5 research angles
  Research areas: AI ethics, implementation costs, patient outcomes
```

### ✓ Fact-Checker Agent
**Role:** Verifies claims and detects contradictions

**Capabilities:**
- Extracts verifiable claims from all sources
- Cross-references information across documents and web
- Detects contradictions with severity levels (high/medium/low)
- Assesses source quality (0-100%)
- Assigns confidence scores to findings

**Example Output:**
```
⚠ Found 2 contradictions
  - Cost estimates vary by 40% (Severity: high)
    Source 1: $100K vs Source 2: $140K
  Confidence: 85%
```

### 🔄 Synthesizer Agent
**Role:** Combines information from multiple sources

**Capabilities:**
- Integrates document and web findings
- Resolves contradictions with explanations
- Creates coherent narratives from fragmented information
- Generates structured summaries
- Rates evidence strength (strong/moderate/weak)
- Assesses consensus levels (high/medium/low)

**Example Output:**
```
✓ Synthesized comprehensive answer
  Evidence strength: strong
  Consensus level: high
  Combined 25 sources
```

### 🎯 Critic Agent
**Role:** Questions assumptions and identifies gaps

**Capabilities:**
- Identifies 5-7 specific knowledge gaps
- Questions implicit assumptions
- Assesses research completeness (0-100%)
- Detects biases in sources
- Suggests improvements for future research

**Example Output:**
```
✓ Completeness: 85% - Found 4 gaps
  Gaps identified:
  - Long-term outcomes data missing
  - No developing country context
  - Cost-benefit analysis needed
```

## 💡 Usage Examples

### Basic Standard Q&A
```python
from multimodal_rag import MultiModalRAGAgent

agent = MultiModalRAGAgent(
    api_key="your_key",
    model="meituan/longcat-flash-chat:free"
)

agent.ingest_document("paper.pdf")

# Quick answer
result = agent.answer_question("What are the main findings?")
print(result['answer'])
```

### Deep Research with All Features
```python
# Enable deep research
agent = MultiModalRAGAgent(
    api_key="your_key",
    enable_deep_research=True,
    enable_web_search=True
)

agent.ingest_document("research.pdf")

# Thorough investigation
result = agent.answer_question(
    "What are the long-term implications of these findings?",
    use_web_search=True,
    use_deep_research=True,
    max_iterations=5  # 5 research iterations
)

# Access comprehensive results
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']*100}%")

# View agent insights
dr = result['deep_research']
print(f"\nFindings: {dr['findings_count']}")
print(f"Contradictions: {len(dr['contradictions'])}")
print(f"Gaps: {len(dr['gaps'])}")

# See what each agent decided
for decision in dr['key_decisions']:
    print(f"\n[{decision['agent']}] {decision['decision']}")
    print(f"  Reasoning: {decision['reasoning']}")
    print(f"  Confidence: {decision['confidence']*100}%")

# Check for contradictions
if dr['contradictions']:
    print("\n⚠️ Contradictions Found:")
    for c in dr['contradictions']:
        print(f"  - {c['description']}")
    print(f"\nAnalysis: {dr['contradiction_analysis']}")

# Review research gaps
print("\n❓ Research Gaps:")
for gap in dr['gaps']:
    print(f"  - {gap}")

# View structured summary
summary = dr['structured_summary']
print(f"\n📊 Summary:")
print(f"  Evidence Strength: {summary['evidence_strength']}")
print(f"  Consensus Level: {summary['consensus_level']}")
print(f"  Key Points: {len(summary['key_points'])}")
```

### Document Analysis with Deep Research
```python
analysis = agent.analyze_document(
    analysis_type="comprehensive analysis with implications",
    use_web_search=True,
    use_deep_research=True
)

print(analysis['analysis'])
print(f"Confidence: {analysis['confidence']*100}%")

# View agent decisions
for decision in analysis['key_decisions']:
    print(f"[{decision['agent']}] {decision['decision']}")
```

### Critic-Reviewed Document Rewriting
```python
agent.rewrite_document(
    instructions="Modernize language and add latest research",
    output_path="updated_document.txt",
    use_web_search=True,
    use_critic_review=True  # Critic agent reviews before finalizing
)
```

### Fact Verification
```python
# Verify specific claims
result = agent.answer_question(
    "Verify the claim that AI reduces healthcare costs by 30%",
    use_deep_research=True,
    max_iterations=7
)

# Check if claim is disputed
if result['deep_research']['contradictions']:
    print("⚠️ Claim is disputed!")
    for c in result['deep_research']['contradictions']:
        print(f"  {c['description']}")
```

## 🎯 When to Use Each Mode

### Use Standard Mode ✅ When:
- Simple fact lookups ("What is X?")
- Quick summaries needed
- Well-established information
- Time is limited (~10 seconds)
- Information is stable (rarely changes)

**Example Questions:**
- "What are the key points in this document?"
- "Summarize section 3"
- "Define machine learning"

### Use Deep Research 🔬 When:
- Complex, multi-faceted questions
- Critical decisions depend on answer
- Need high confidence (85-95%)
- Expect conflicting information
- Want comprehensive analysis
- Research quality matters more than speed
- Need to identify knowledge gaps

**Example Questions:**
- "Compare AI safety approaches and identify trade-offs"
- "What are the long-term economic implications of these policies?"
- "Verify these research findings against current literature"
- "Analyze the document and identify what's missing"

## 💰 Cost & Performance

### FREE Tier (Recommended for Getting Started)
- **Model**: `meituan/longcat-flash-chat:free` (or other free models)
- **Web Search**: DuckDuckGo (completely free, no API key)
- **Deep Research**: FREE (just uses more API calls)
- **Total Cost**: $0/month! 🎉

**Performance:**
- Standard: ~10 seconds, 75% confidence
- Deep Research: 1-3 minutes, 85-95% confidence
- Quality: Excellent for most use cases

### Paid Tier (Better Quality/Speed)
- **Models**: `anthropic/claude-3.5-sonnet`, `openai/gpt-4-turbo`
- **Web Search**: Tavily, SerpAPI, Serper
- **Cost**: ~$0.50-1.00 per deep research query
- **Performance**: Faster, even higher quality

## 📁 Project Structure

```
multimodal-rag-agent/
│
├── 🔬 CORE SYSTEM
│   ├── multimodal_rag.py              # Main RAG agent
│   ├── deep_research_agents.py        # 4 AI agents (NEW Phase 3)
│   └── app.py                          # Streamlit UI
│
├── 📋 CONFIGURATION
│   ├── requirements.txt                # Python dependencies
│   ├── .env.example                    # Config template
│   └── .env                            # Your API keys (create this)
│
├── 🧪 TESTING & EXAMPLES
│   ├── quick_test.py                   # Installation test
│   └── usage_examples.py               # Code examples
│
├── 📚 DOCUMENTATION
│   ├── README.md                       # This file
│   ├── DEEP_RESEARCH_GUIDE.md         # Deep research guide
│   ├── QUICK_REFERENCE.md             # Quick cheat sheet
│   ├── GETTING_STARTED_DEEP_RESEARCH.md  # 5-min quickstart
│   ├── PROJECT_SUMMARY.md             # Overview
│   └── INSTALLATION_TROUBLESHOOTING.md  # Setup help
│
└── 💾 DATA (auto-created)
    └── chroma_db/                      # Vector database
```

## 🔧 Configuration

### Environment Variables (.env file)
```bash
# Required
OPENROUTER_API_KEY=your_key_here

# Optional - Web Search APIs
TAVILY_API_KEY=your_key      # If using Tavily
SERPAPI_API_KEY=your_key     # If using SerpAPI
SERPER_API_KEY=your_key      # If using Serper

# Optional - Customization
DEFAULT_MODEL=meta-llama/llama-3.2-3b-instruct:free
CHROMA_PERSIST_DIR=./chroma_db
```

### Code Configuration
```python
agent = MultiModalRAGAgent(
    # Required
    api_key="your_openrouter_key",
    
    # Optional - Model Selection
    model="meta-llama/llama-3.2-3b-instruct:free",  # Free option
    
    # Optional - Web Search
    enable_web_search=True,
    search_provider="duckduckgo",  # Free, no API key
    search_api_key=None,  # Only if using paid provider
    
    # Optional - Deep Research
    enable_deep_research=True  # Enable 4-agent system
)
```

### Research Depth Settings
```python
# Quick (30 seconds)
max_iterations=3

# Balanced (1 minute) - Recommended
max_iterations=5

# Thorough (2-3 minutes)
max_iterations=7

# Comprehensive (3-5 minutes)
max_iterations=10
```

## 🎨 Streamlit UI Features

### Configuration Sidebar
- ⚙️ API key input
- 🎛️ Model selection (free options shown)
- 🌐 Web search provider selection
- 🔬 **Deep research toggle**
- 📊 **Research depth slider** (3-10 iterations)
- 🤖 **Active agents status display**

### Chat Interface Features
- 💬 Interactive Q&A with documents
- 🌐 Web search toggle per query
- 🔬 **Deep research mode toggle**
- 📈 **Confidence progress bars**
- 📚 Expandable source display
- 🤖 **Agent decision viewer** (clean, not verbose)
- ⚠️ **Contradiction alerts**
- ❓ **Gap identification display**
- 📊 **Structured summaries**
- 💾 Chat history with metadata

### Analysis Tab Features
- 📊 Multiple analysis types
- 🌐 Web enhancement option
- 🔬 **Deep research toggle**
- 📈 Confidence indicators
- 🤖 **Agent insights display**
- 📥 Download results

### Rewrite Tab Features
- ✏️ Quick presets + custom instructions
- 📝 Output format selection
- 🌐 Web enhancement
- 🎯 **Critic agent review option**
- 📥 Download rewritten content

## 📖 Complete Documentation

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **README.md** | Main documentation | Start here |
| **GETTING_STARTED_DEEP_RESEARCH.md** | 5-minute quickstart | First time setup |
| **DEEP_RESEARCH_GUIDE.md** | Complete agent guide | Learn deep research |
| **QUICK_REFERENCE.md** | Cheat sheet | Quick lookups |
| **PROJECT_SUMMARY.md** | Project overview | Understand structure |
| **INSTALLATION_TROUBLESHOOTING.md** | Setup help | Having issues? |

## 🧪 Testing & Verification

### Run Comprehensive Tests
```bash
python quick_test.py
```

### Test Deep Research Specifically
```python
python -c "
from multimodal_rag import MultiModalRAGAgent

agent = MultiModalRAGAgent(
    api_key='your_key',
    enable_deep_research=True
)

print('✓ Deep research ready!' if agent.orchestrator else '✗ Failed')
print(f'✓ Orchestrator: {agent.orchestrator}')
print(f'✓ Web Search: {agent.web_search}')
"
```

### Run Example Scripts
```bash
# All examples
python usage_examples.py

# Specific example
python -c "
from usage_examples import example_deep_research_insights, example_basic_setup
agent = example_basic_setup()
example_deep_research_insights(agent)
"
```

## 🐛 Troubleshooting

### "Deep research not working"
```python
# Check if enabled
print(agent.enable_deep_research)  # Should be True
print(agent.orchestrator)  # Should not be None

# Verify file exists
import os
print(os.path.exists('deep_research_agents.py'))  # Should be True
```

### "Cannot import deep_research_agents"
```bash
# Ensure file exists in same directory
ls deep_research_agents.py

# Test import
python -c "from deep_research_agents import ResearchOrchestrator; print('OK')"
```

### "Taking too long"
```python
# Reduce iterations
max_iterations=3  # Faster

# Or disable web search
use_web_search=False
```

### "Low confidence scores"
```python
# Add more documents
agent.ingest_document("doc1.pdf")
agent.ingest_document("doc2.pdf")

# Enable web search
use_web_search=True

# Increase iterations
max_iterations=7
```

### "PDF parsing errors"
See `INSTALLATION_TROUBLESHOOTING.md` for detailed PDF setup help.

## 🎓 Learning Path

### Week 1: Basics
- [ ] Install and configure
- [ ] Run standard Q&A
- [ ] Upload a document
- [ ] Try web search

### Week 2: Deep Research
- [ ] Enable deep research
- [ ] Run 3-iteration research
- [ ] Review agent decisions
- [ ] Check confidence scores

### Week 3: Advanced
- [ ] Identify contradictions
- [ ] Address research gaps
- [ ] Use critic review
- [ ] Multi-document analysis

### Week 4: Production
- [ ] Deploy Streamlit app
- [ ] Optimize for your use case
- [ ] Create custom workflows
- [ ] Monitor performance

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **LangChain** - RAG infrastructure
- **ChromaDB** - Vector storage
- **Unstructured** - Document parsing
- **OpenRouter** - LLM access
- **DuckDuckGo** - Free web search


## 🗺️ Roadmap

- [x] Phase 1: Core RAG with multi-modal support
- [x] Phase 2: Web search integration
- [x] Phase 3: Multi-agent deep research system
- [ ] Phase 4: Knowledge graphs and visualization
- [ ] Phase 5: Advanced caching and optimization
- [ ] Phase 6: Multi-language support
- [ ] Phase 7: API endpoints with FastAPI
- [ ] Phase 8: Collaborative features

---

