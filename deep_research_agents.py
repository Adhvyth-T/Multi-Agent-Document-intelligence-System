"""
Deep Research Multi-Agent System
Advanced research capabilities with specialized agents
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from collections import defaultdict


@dataclass
class ResearchDecision:
    """Represents a key decision made during research"""
    agent: str
    decision: str
    reasoning: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResearchFinding:
    """Represents a finding from research"""
    content: str
    source: str
    source_type: str  # 'document' or 'web'
    confidence: float
    verified: bool = False
    contradictions: List[str] = field(default_factory=list)


@dataclass
class ResearchSession:
    """Tracks an entire research session"""
    query: str
    decisions: List[ResearchDecision] = field(default_factory=list)
    findings: List[ResearchFinding] = field(default_factory=list)
    iterations: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    gaps: List[str] = field(default_factory=list)
    final_confidence: float = 0.0


class ExplorerAgent:
    """Agent responsible for discovering related topics and expanding research scope"""
    
    def __init__(self, llm_client, model: str):
        self.client = llm_client
        self.model = model
        self.name = "Explorer"
    
    def generate_sub_queries(self, query: str, depth: int = 5) -> List[str]:
        """Generate diverse sub-queries for comprehensive research"""
        
        prompt = f"""You are a research explorer. Break down this question into {depth} diverse sub-questions 
that cover different aspects and perspectives. Each sub-question should explore a unique angle.

Main Question: {query}

Generate {depth} sub-questions that explore:
1. Core concepts and definitions
2. Historical context and evolution
3. Current state and recent developments
4. Practical applications and implications
5. Challenges, criticisms, and limitations

Return ONLY the questions, one per line, numbered 1-{depth}."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research question decomposition expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        text = response.choices[0].message.content
        queries = [q.strip() for q in re.findall(r'\d+\.\s*(.+)', text)]
        return queries[:depth]
    
    def expand_search_terms(self, query: str) -> List[str]:
        """Generate alternative search terms and related concepts"""
        
        prompt = f"""Generate 5 alternative search queries for: "{query}"

Include:
- Synonyms and related terms
- Technical and common language variants
- Broader and narrower concepts
- Related fields and applications

Return only the search queries, one per line."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a search query optimization expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        
        text = response.choices[0].message.content
        terms = [line.strip() for line in text.split('\n') if line.strip() and not line.startswith('#')]
        return terms[:5]
    
    def identify_research_areas(self, query: str) -> Dict[str, List[str]]:
        """Identify key research areas and their components"""
        
        prompt = f"""Analyze this research question and identify key areas to investigate: "{query}"

Return a JSON object with these keys:
- "core_concepts": List of fundamental concepts to understand
- "stakeholders": Who is affected or involved
- "contexts": Different contexts or domains to consider
- "timeframes": Relevant time periods (historical, current, future)

Return ONLY valid JSON."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research planning expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {
                "core_concepts": [query],
                "stakeholders": [],
                "contexts": [],
                "timeframes": []
            }


class FactCheckerAgent:
    """Agent responsible for verifying claims and detecting contradictions"""
    
    def __init__(self, llm_client, model: str):
        self.client = llm_client
        self.model = model
        self.name = "FactChecker"
    
    def extract_claims(self, text: str) -> List[str]:
        """Extract verifiable claims from text"""
        
        prompt = f"""Extract all factual claims from this text that can be verified:

Text: {text[:2000]}

Return a numbered list of claims. Include only specific, verifiable statements (facts, statistics, dates, names, etc.).
Exclude opinions and subjective statements."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a claim extraction expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        text_response = response.choices[0].message.content
        claims = [c.strip() for c in re.findall(r'\d+\.\s*(.+)', text_response)]
        return claims
    
    def verify_claim(self, claim: str, sources: List[ResearchFinding]) -> Tuple[bool, float, List[str]]:
        """Verify a claim against available sources"""
        
        source_texts = "\n\n".join([
            f"Source {i+1} ({s.source_type}): {s.content[:500]}"
            for i, s in enumerate(sources)
        ])
        
        prompt = f"""Verify this claim against the provided sources:

Claim: {claim}

Sources:
{source_texts}

Analyze:
1. Is the claim supported by the sources?
2. What's your confidence level (0-100)?
3. Are there any contradictions?

Return a JSON object:
{{
    "verified": true/false,
    "confidence": 0-100,
    "contradictions": ["list of contradicting information if any"]
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a fact verification expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result.get("verified", False), result.get("confidence", 0) / 100, result.get("contradictions", [])
        except:
            return False, 0.0, []
    
    def detect_contradictions(self, findings: List[ResearchFinding]) -> List[Dict[str, Any]]:
        """Detect contradictions across multiple findings"""
        
        if len(findings) < 2:
            return []
        
        contradictions = []
        
        for i, finding1 in enumerate(findings):
            for finding2 in findings[i+1:]:
                prompt = f"""Compare these two pieces of information and identify any contradictions:

Source 1: {finding1.content[:500]}
Source 2: {finding2.content[:500]}

Return JSON:
{{
    "contradicts": true/false,
    "contradiction": "description of contradiction if any",
    "severity": "high/medium/low"
}}"""

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a contradiction detection expert. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                
                try:
                    result = json.loads(response.choices[0].message.content)
                    if result.get("contradicts", False):
                        contradictions.append({
                            "source1": finding1.source,
                            "source2": finding2.source,
                            "description": result.get("contradiction", ""),
                            "severity": result.get("severity", "medium")
                        })
                except:
                    continue
        
        return contradictions
    
    def assess_source_quality(self, finding: ResearchFinding) -> float:
        """Assess the quality and reliability of a source"""
        
        prompt = f"""Assess the quality of this source:

Content: {finding.content[:500]}
Source: {finding.source}
Type: {finding.source_type}

Rate on scale 0-100 based on:
- Specificity and detail
- Internal consistency
- Source credibility indicators
- Recency (if applicable)

Return only a number 0-100."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a source quality assessment expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        try:
            score = float(re.search(r'\d+', response.choices[0].message.content).group())
            return min(max(score / 100, 0.0), 1.0)
        except:
            return 0.5


class SynthesizerAgent:
    """Agent responsible for combining information from multiple sources"""
    
    def __init__(self, llm_client, model: str):
        self.client = llm_client
        self.model = model
        self.name = "Synthesizer"
    
    def synthesize_findings(self, query: str, findings: List[ResearchFinding], 
                          contradictions: List[Dict]) -> str:
        """Synthesize information from multiple sources into coherent answer"""
        
        # Group findings by source type
        doc_findings = [f for f in findings if f.source_type == 'document']
        web_findings = [f for f in findings if f.source_type == 'web']
        
        findings_text = "DOCUMENT SOURCES:\n"
        for i, f in enumerate(doc_findings, 1):
            findings_text += f"{i}. {f.content[:300]}...\n"
        
        findings_text += "\nWEB SOURCES:\n"
        for i, f in enumerate(web_findings, 1):
            findings_text += f"{i}. {f.content[:300]}...\n"
        
        contradiction_text = ""
        if contradictions:
            contradiction_text = "\nNOTE - CONTRADICTIONS FOUND:\n"
            for c in contradictions:
                contradiction_text += f"- {c['description']} (Severity: {c['severity']})\n"
        
        prompt = f"""Synthesize a comprehensive answer to this question using the provided sources:

Question: {query}

{findings_text}

{contradiction_text}

Requirements:
1. Integrate information from multiple sources
2. Address any contradictions explicitly
3. Prioritize document sources but supplement with web sources
4. Be comprehensive yet concise
5. Maintain accuracy and cite key points

Provide a well-structured, synthesis that combines all relevant information."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at synthesizing information from multiple sources into coherent, comprehensive answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        
        return response.choices[0].message.content
    
    def create_structured_summary(self, findings: List[ResearchFinding]) -> Dict[str, Any]:
        """Create structured summary of research findings"""
        
        all_content = "\n\n".join([f.content for f in findings])
        
        prompt = f"""Create a structured summary of these research findings:

{all_content[:3000]}

Return JSON with:
{{
    "key_points": ["list of 5-7 main points"],
    "evidence_strength": "strong/moderate/weak",
    "consensus_level": "high/medium/low",
    "notable_findings": ["list of particularly important or surprising findings"]
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research summarization expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {
                "key_points": ["Summary unavailable"],
                "evidence_strength": "unknown",
                "consensus_level": "unknown",
                "notable_findings": []
            }
    
    def resolve_contradictions(self, contradictions: List[Dict], findings: List[ResearchFinding]) -> str:
        """Attempt to resolve or explain contradictions"""
        
        if not contradictions:
            return "No contradictions detected."
        
        contradiction_text = "\n".join([
            f"- {c['description']}" for c in contradictions
        ])
        
        prompt = f"""Analyze and explain these contradictions found in the research:

{contradiction_text}

Provide:
1. Possible reasons for the contradictions
2. Which sources might be more reliable
3. Any context that explains the differences
4. Recommendations for resolving uncertainty

Be balanced and acknowledge genuine uncertainty where it exists."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing and explaining contradictions in research."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        
        return response.choices[0].message.content


class CriticAgent:
    """Agent responsible for questioning assumptions and identifying gaps"""
    
    def __init__(self, llm_client, model: str):
        self.client = llm_client
        self.model = model
        self.name = "Critic"
    
    def identify_gaps(self, query: str, findings: List[ResearchFinding]) -> List[str]:
        """Identify gaps in the research"""
        
        findings_summary = "\n".join([
            f"- {f.content[:200]}..." for f in findings[:10]
        ])
        
        prompt = f"""Review this research and identify gaps or missing information:

Research Question: {query}

Current Findings:
{findings_summary}

Identify:
1. Important questions that remain unanswered
2. Perspectives or contexts not covered
3. Key information that's missing
4. Areas needing deeper investigation

Return a numbered list of specific gaps (5-7 items)."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a critical research reviewer focused on identifying gaps."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )
        
        text = response.choices[0].message.content
        gaps = [g.strip() for g in re.findall(r'\d+\.\s*(.+)', text)]
        return gaps
    
    def question_assumptions(self, query: str, answer: str) -> List[str]:
        """Question assumptions in the research and answer"""
        
        prompt = f"""Critically examine this research question and answer for assumptions:

Question: {query}
Answer: {answer[:1000]}

Identify:
1. Implicit assumptions in the question itself
2. Assumptions made in the answer
3. Potential biases
4. Alternative perspectives not considered

Return a numbered list of assumptions/concerns (4-6 items)."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a critical thinker who questions assumptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )
        
        text = response.choices[0].message.content
        assumptions = [a.strip() for a in re.findall(r'\d+\.\s*(.+)', text)]
        return assumptions
    
    def assess_completeness(self, query: str, findings: List[ResearchFinding]) -> Tuple[float, str]:
        """Assess how completely the question has been answered"""
        
        findings_summary = "\n".join([
            f"- {f.content[:150]}..." for f in findings[:15]
        ])
        
        prompt = f"""Assess how completely this question has been answered:

Question: {query}

Available Information:
{findings_summary}

Return JSON:
{{
    "completeness_score": 0-100,
    "assessment": "brief explanation of what's covered well and what's lacking"
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research completeness assessor. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result.get("completeness_score", 50) / 100, result.get("assessment", "")
        except:
            return 0.5, "Assessment unavailable"
    
    def suggest_improvements(self, session: ResearchSession) -> List[str]:
        """Suggest improvements to the research approach"""
        
        prompt = f"""Review this research session and suggest improvements:

Question: {session.query}
Iterations completed: {session.iterations}
Findings gathered: {len(session.findings)}
Gaps identified: {len(session.gaps)}

Suggest 3-5 specific ways to improve this research."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research methodology expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )
        
        text = response.choices[0].message.content
        suggestions = [s.strip() for s in re.findall(r'\d+\.\s*(.+)', text)]
        return suggestions


class ResearchOrchestrator:
    """Orchestrates all research agents and manages the research workflow"""
    
    def __init__(self, llm_client, model: str, vector_store, web_search):
        self.client = llm_client
        self.model = model
        self.vector_store = vector_store
        self.web_search = web_search
        
        # Initialize agents
        self.explorer = ExplorerAgent(llm_client, model)
        self.fact_checker = FactCheckerAgent(llm_client, model)
        self.synthesizer = SynthesizerAgent(llm_client, model)
        self.critic = CriticAgent(llm_client, model)
        
        self.current_session = None
    
    def deep_research(self, query: str, max_iterations: int = 5, 
                     use_web_search: bool = True) -> Dict[str, Any]:
        """
        Perform deep research with multi-agent collaboration
        
        Returns comprehensive research results with key decisions shown
        """
        
        # Initialize session
        session = ResearchSession(query=query)
        self.current_session = session
        
        print(f"\n{'='*60}")
        print(f"🔬 DEEP RESEARCH MODE ACTIVATED")
        print(f"{'='*60}")
        print(f"Query: {query}\n")
        
        # PHASE 1: EXPLORATION
        print("📡 PHASE 1: EXPLORATION")
        print("→ Explorer Agent: Analyzing question and planning research...")
        
        sub_queries = self.explorer.generate_sub_queries(query, depth=max_iterations)
        research_areas = self.explorer.identify_research_areas(query)
        
        decision = ResearchDecision(
            agent="Explorer",
            decision=f"Generated {len(sub_queries)} research angles",
            reasoning=f"Identified key areas: {', '.join(research_areas.get('core_concepts', [])[:3])}",
            confidence=0.9
        )
        session.decisions.append(decision)
        print(f"  ✓ {decision.decision}")
        print(f"    Research angles: {len(sub_queries)}")
        
        # PHASE 2: DEEP ITERATIVE SEARCH
        print(f"\n🔍 PHASE 2: ITERATIVE DEEP SEARCH ({max_iterations} iterations)")
        
        all_findings = []
        
        for iteration in range(max_iterations):
            session.iterations += 1
            print(f"\n  Iteration {iteration + 1}/{max_iterations}")
            
            current_query = sub_queries[iteration] if iteration < len(sub_queries) else query
            
            # Search documents
            print(f"    → Searching documents...")
            doc_results = self.vector_store.similarity_search(current_query, k=3)
            
            for doc in doc_results:
                finding = ResearchFinding(
                    content=doc.page_content,
                    source=doc.metadata.get('source', 'Unknown'),
                    source_type='document',
                    confidence=0.8
                )
                all_findings.append(finding)
                session.findings.append(finding)
            
            # Search web if enabled
            if use_web_search and self.web_search:
                print(f"    → Searching web...")
                expanded_terms = self.explorer.expand_search_terms(current_query)
                
                for term in expanded_terms[:2]:
                    web_results = self.web_search.search(term, max_results=2)
                    
                    for result in web_results:
                        finding = ResearchFinding(
                            content=f"{result['title']}: {result['snippet']}",
                            source=result['url'],
                            source_type='web',
                            confidence=0.7
                        )
                        all_findings.append(finding)
                        session.findings.append(finding)
            
            print(f"    ✓ Gathered {len(all_findings)} total findings so far")
        
        decision = ResearchDecision(
            agent="Explorer",
            decision=f"Completed {max_iterations} research iterations",
            reasoning=f"Collected {len(all_findings)} findings from documents and web",
            confidence=0.85
        )
        session.decisions.append(decision)
        
        # PHASE 3: FACT CHECKING
        print(f"\n✓ PHASE 3: FACT CHECKING")
        print("→ Fact-Checker Agent: Verifying claims...")
        
        # Assess source quality
        for finding in all_findings[:10]:  # Check top findings
            quality_score = self.fact_checker.assess_source_quality(finding)
            finding.confidence = quality_score
        
        # Detect contradictions
        contradictions = self.fact_checker.detect_contradictions(all_findings[:15])
        
        if contradictions:
            decision = ResearchDecision(
                agent="FactChecker",
                decision=f"Found {len(contradictions)} contradictions",
                reasoning="Cross-source verification revealed conflicting information",
                confidence=0.9
            )
            print(f"  ⚠ {decision.decision}")
        else:
            decision = ResearchDecision(
                agent="FactChecker",
                decision="No major contradictions detected",
                reasoning="Sources are largely consistent",
                confidence=0.95
            )
            print(f"  ✓ {decision.decision}")
        
        session.decisions.append(decision)
        
        # PHASE 4: SYNTHESIS
        print(f"\n🔄 PHASE 4: SYNTHESIS")
        print("→ Synthesizer Agent: Combining information...")
        
        synthesized_answer = self.synthesizer.synthesize_findings(
            query, all_findings, contradictions
        )
        
        structured_summary = self.synthesizer.create_structured_summary(all_findings)
        
        decision = ResearchDecision(
            agent="Synthesizer",
            decision="Synthesized comprehensive answer",
            reasoning=f"Evidence strength: {structured_summary.get('evidence_strength', 'unknown')}",
            confidence=0.85
        )
        session.decisions.append(decision)
        print(f"  ✓ {decision.decision}")
        
        if contradictions:
            print("→ Synthesizer Agent: Resolving contradictions...")
            contradiction_analysis = self.synthesizer.resolve_contradictions(
                contradictions, all_findings
            )
        else:
            contradiction_analysis = "No contradictions to resolve."
        
        # PHASE 5: CRITICAL REVIEW
        print(f"\n🎯 PHASE 5: CRITICAL REVIEW")
        print("→ Critic Agent: Identifying gaps and assumptions...")
        
        gaps = self.critic.identify_gaps(query, all_findings)
        session.gaps = gaps
        
        completeness, assessment = self.critic.assess_completeness(query, all_findings)
        
        decision = ResearchDecision(
            agent="Critic",
            decision=f"Completeness: {int(completeness*100)}% - Found {len(gaps)} gaps",
            reasoning=assessment,
            confidence=completeness
        )
        session.decisions.append(decision)
        session.final_confidence = completeness
        print(f"  ✓ {decision.decision}")
        
        assumptions = self.critic.question_assumptions(query, synthesized_answer)
        
        # Calculate final confidence
        avg_confidence = sum(f.confidence for f in all_findings) / len(all_findings) if all_findings else 0.5
        final_confidence = (completeness + avg_confidence) / 2
        session.final_confidence = final_confidence
        
        print(f"\n{'='*60}")
        print(f"✅ RESEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"Confidence: {int(final_confidence*100)}%")
        print(f"Findings: {len(all_findings)}")
        print(f"Contradictions: {len(contradictions)}")
        print(f"Gaps Identified: {len(gaps)}\n")
        
        return {
            'answer': synthesized_answer,
            'confidence': final_confidence,
            'key_decisions': [
                {
                    'agent': d.agent,
                    'decision': d.decision,
                    'reasoning': d.reasoning,
                    'confidence': d.confidence
                }
                for d in session.decisions
            ],
            'findings_count': len(all_findings),
            'contradictions': contradictions,
            'contradiction_analysis': contradiction_analysis,
            'gaps': gaps,
            'assumptions': assumptions,
            'structured_summary': structured_summary,
            'session': session
        }