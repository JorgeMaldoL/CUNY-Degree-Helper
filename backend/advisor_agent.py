
# Fix SQLite issues FIRST - before any other imports
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("✅ Successfully replaced sqlite3 with pysqlite3-binary in advisor_agent")
except ImportError:
    print("❌ pysqlite3-binary not available in advisor_agent")

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
from dotenv import load_dotenv
from .chroma_manager import ChromaDBManager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    interests: List[str]
    preferred_location: Optional[str] = None
    budget_max: Optional[float] = None
    career_goals: List[str] = None
    values: List[str] = None
    academic_strengths: List[str] = None
    
@dataclass
class Recommendation:
    # Individual recommendation result
    program_name: str
    college: str
    degree_type: str
    estimated_cost: float
    fit_score: float
    reasoning: str
    pros: List[str]
    cons: List[str]

class CUNYAdvisorAgent:
    """
    🎯 Main AI advisor agent that combines multiple AI models, 
    vector search, and structured data analysis
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.programs_data = []
        self.cost_data = {}
        self.top_majors = {}
        
        # Initialize AI client (Anthropic via LangChain)
        self._setup_ai_client()
        
        # Load and prepare data
        self._load_all_data()
        
        # Initialize vector database
        self._setup_vector_db()
        
        # Initialize LangChain RAG chain
        self._setup_langchain()
        
    def _setup_ai_client(self):
        """Initialize AI model client (Claude via LangChain)."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("❌ ANTHROPIC_API_KEY not found. This is required for the application to work.")
            raise ValueError("ANTHROPIC_API_KEY is required")
        # Allow overriding the model via env; default to a current, supported model
        model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        # ChatAnthropic will read the key from env if not passed explicitly
        self.llm = ChatAnthropic(
            model=model_name,
            anthropic_api_key=api_key,
            timeout=30,
            max_retries=2,
        )
        logger.info(f"🤖 Using Claude (Anthropic) model: {model_name}")
    
    def _load_all_data(self):
        """Load all CUNY data files"""
        try:
            # Load cleaned programs data
            programs_file = self.data_dir / "cleaned_programs.json"
            if programs_file.exists():
                with open(programs_file, 'r', encoding='utf-8') as f:
                    self.programs_data = json.load(f)
                logger.info(f"📊 Loaded {len(self.programs_data)} CUNY programs")
            
            # Load cost data
            cost_file = self.data_dir / "CUNY_Cost_Data.json"
            if cost_file.exists():
                with open(cost_file, 'r', encoding='utf-8') as f:
                    self.cost_data = json.load(f)
                logger.info("💰 Loaded CUNY cost data")
            
            # Load top majors
            majors_file = self.data_dir / "CUNY_Top_15_Majors.json"
            if majors_file.exists():
                with open(majors_file, 'r', encoding='utf-8') as f:
                    self.top_majors = json.load(f)
                logger.info("🎓 Loaded top majors data")
                
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def _setup_vector_db(self):
        """Initialize ChromaDB"""
        try:
            self.chroma_manager = ChromaDBManager()
            logger.info("🔍 ChromaDB search ready")
        except Exception as e:
            logger.error(f"❌ ChromaDB setup failed: {e}")
            raise e  # Don't continue without ChromaDB
    
    def _setup_langchain(self):
        """Initialize a minimal LangChain RAG chain (prompt -> Claude -> text)."""
        system_instructions = (
            "You are a helpful CUNY academic advisor. Use the provided context (programs and data) to answer "
            "clearly and concisely. Prefer specific CUNY programs, campuses, and degree types. If information is "
            "not in context, say so briefly and suggest how to refine the query."
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_instructions),
            (
                "human",
                "Question: {question}\n\n"
                "User profile (optional): {profile}\n\n"
                "Context (top matches):\n{context}\n\n"
                "Please provide a short, helpful answer with 2-3 concrete program suggestions when possible."
            ),
        ])

        self._output_parser = StrOutputParser()
        # Chain will be used via .invoke({question, profile, context})
        self.chain = (self.prompt | self.llm | self._output_parser)

    # -------------------------
    # RAG helpers
    # -------------------------
    def _build_profile_text(self, profile: Optional["UserProfile"]) -> str:
        if not profile:
            return "(none)"
        parts = []
        if profile.interests:
            parts.append(f"interests={', '.join(profile.interests)}")
        if profile.preferred_location:
            parts.append(f"location={profile.preferred_location}")
        if profile.budget_max:
            parts.append(f"budget_max=${profile.budget_max:,.0f}")
        if profile.career_goals:
            parts.append(f"career_goals={', '.join(profile.career_goals)}")
        return "; ".join(parts) if parts else "(none)"

    def _retrieve_program_context(self, query: str, k: int = 5):
        """Retrieve top-k program matches and optionally augment with web snippets.

        Returns (context_text, sources_list).
        """
        results = self.chroma_manager.search_programs(query, n_results=k)
        context_lines = []
        sources = []
        if results and results.get('metadatas') and results['metadatas'][0]:
            for md in results['metadatas'][0]:
                name = md.get('program_name', '')
                college = md.get('college', '')
                degree = md.get('degree_type', '')
                field = md.get('cip_title', '')
                tap = md.get('tap_eligible', '')
                context_lines.append(f"- {name} | {college} | {degree} | {field} | TAP: {tap}")
                sources.append({
                    "program_name": name,
                    "college": college,
                    "degree_type": degree,
                    "cip_title": field,
                })

        return ("\n".join(context_lines) if context_lines else "(no matching programs found)", sources)

    def ask_advisor(self, question: str, user_profile: Optional["UserProfile"] = None) -> Dict[str, Any]:
        """Answer a free-form question using RAG over CUNY programs. Returns {'answer', 'sources'}.

        This keeps history out for simplicity; Streamlit can pass prior messages if needed later.
        """
        if not question or not question.strip():
            return {"answer": "Please enter a question.", "sources": []}

        profile_text = self._build_profile_text(user_profile)
        context_text, sources = self._retrieve_program_context(question, k=5)

        try:
            answer = self.chain.invoke({
                "question": question.strip(),
                "profile": profile_text,
                "context": context_text,
            })
        except Exception as e:
            logger.error(f"❌ LLM error: {e}")
            answer = "I couldn't generate an answer right now. Please try again."

        return {"answer": answer, "sources": sources}
    
    def _search_programs_tool(self, query: str) -> str:
        """Tool: Search for relevant CUNY programs using ChromaDB manager"""
        if not self.chroma_manager:
            raise Exception("ChromaDB manager is not available")
            
        if not self.chroma_manager.is_available():
            raise Exception("ChromaDB collection is not available")
            
        try:
            results = self.chroma_manager.search_programs(query, n_results=10)
            
            if not results or not results.get('metadatas'):
                return "No programs found"
            
            programs = []
            for metadata in results['metadatas'][0]:
                programs.append({
                    'program': metadata.get('program_name'),
                    'college': metadata.get('college'),
                    'degree': metadata.get('degree_type')
                })
            
            return json.dumps(programs, indent=2)
            
        except Exception as e:
            raise Exception(f"Search error: {e}")
    
    def _calculate_costs_tool(self, college_name: str) -> str:
        """Tool: Calculate costs for a specific college"""
        base_tuition = 7560  # Standard CUNY tuition
        
        # Look up specific college costs
        college_costs = {}
        if 'TuitionAndFees' in self.cost_data:
            for college_data in self.cost_data['TuitionAndFees']:
                if college_name.lower() in college_data.get('Campus', '').lower():
                    college_costs = college_data
                    break
        
        return json.dumps({
            'college': college_name,
            'base_tuition': base_tuition,
            'additional_info': college_costs
        }, indent=2)
    
    def _match_major_to_schools_tool(self, major: str) -> str:
        """Tool: Find best schools for a specific major"""
        if 'Top15Majors' not in self.top_majors:
            return "Major matching data not available"
        
        for major_info in self.top_majors['Top15Majors']:
            if major.lower() in major_info.get('Major', '').lower():
                return json.dumps({
                    'major': major_info['Major'],
                    'recommended_schools': major_info.get('Common Schools', [])
                }, indent=2)
        
        return f"No specific school recommendations found for {major}"
    
    # Removed: legacy direct Anthropic prompt method in favor of LangChain chain
    
    def get_recommendations(self, user_profile: UserProfile) -> List[Recommendation]:
        """
        🎯 Main method: Generate personalized recommendations
        """
        recommendations = []
        
        try:
            # 1. Search for relevant programs based on interests
            search_queries = user_profile.interests + (user_profile.career_goals or [])
            relevant_programs = []
            
            for query in search_queries[:3]:  # Limit searches
                results = self.chroma_manager.search_programs(query, n_results=5)
                if results and results.get('metadatas'):
                    relevant_programs.extend(results['metadatas'][0])
            
            # 2. Score and rank programs (remove duplicates)
            scored_programs = []
            seen_programs = set()
            
            for program in relevant_programs:
                # Create unique identifier
                program_id = f"{program.get('program_name', '')}_{program.get('college', '')}_{program.get('degree_type', '')}"
                
                if program_id not in seen_programs:
                    seen_programs.add(program_id)
                    score = self._calculate_fit_score(program, user_profile)
                    if score > 0.3:  # Minimum threshold
                        scored_programs.append((program, score))
            
            # Sort by score
            scored_programs.sort(key=lambda x: x[1], reverse=True)
            
            # 3. Generate recommendations
            for program, score in scored_programs[:5]:  # Top 5
                rec = self._create_recommendation(program, score, user_profile)
                if rec:
                    recommendations.append(rec)
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
        
        return recommendations
    
    def _calculate_fit_score(self, program: Dict, user_profile: UserProfile) -> float:
        """Calculate how well a program fits the user profile"""
        score = 0.0
        
        # Interest matching
        program_text = f"{program.get('program_name', '')} {program.get('cip_title', '')}"
        for interest in user_profile.interests:
            if interest.lower() in program_text.lower():
                score += 0.3
        
        # Location preference
        if user_profile.preferred_location:
            college = program.get('college', '')
            if user_profile.preferred_location.lower() in college.lower():
                score += 0.2
        
        # TAP eligibility (budget consideration)
        if program.get('tap_eligible', '').lower() == 'yes':
            score += 0.1
        
        # Popular programs bonus
        program_name = program.get('program_name', '')
        for major_info in self.top_majors.get('Top15Majors', []):
            if major_info['Major'].lower() in program_name.lower():
                score += 0.15
                break
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _create_recommendation(self, program: Dict, score: float, user_profile: UserProfile) -> Optional[Recommendation]:
        """Create a structured recommendation from program data"""
        try:
            # Estimate costs
            base_cost = 7560
            college_name = program.get('college', '')
            
            # Look up additional costs by major type
            additional_cost = 0
            program_name = program.get('program_name', '').lower()
            
            if 'MajorSpecificCosts' in self.cost_data:
                for cost_info in self.cost_data['MajorSpecificCosts']:
                    examples = cost_info.get('Example Majors', '').lower()
                    if any(keyword in program_name for keyword in examples.split(', ')):
                        # Parse cost range
                        cost_range = cost_info.get('Estimated Extra Costs / Year', '')
                        if '$' in cost_range:
                            try:
                                # Extract first number from range
                                import re
                                numbers = re.findall(r'\d+', cost_range)
                                if numbers:
                                    additional_cost = int(numbers[0])
                            except:
                                additional_cost = 500  # Default
                        break
            
            total_cost = base_cost + additional_cost
            
            # Generate reasoning
            reasoning = f"This program matches your interests in {', '.join(user_profile.interests[:2])} "
            reasoning += f"and is offered at {college_name}. "
            
            if program.get('tap_eligible', '').lower() == 'yes':
                reasoning += "It's eligible for NYS TAP financial aid. "
            
            # Generate pros and cons
            pros = [
                f"Strong fit for your interest in {user_profile.interests[0] if user_profile.interests else 'your field'}",
                f"Available at {college_name}",
                f"Estimated annual cost: ${total_cost:,}"
            ]
            
            if program.get('tap_eligible', '').lower() == 'yes':
                pros.append("Eligible for NYS TAP financial aid")
            
            cons = []
            if user_profile.budget_max and total_cost > user_profile.budget_max:
                cons.append(f"May exceed your budget of ${user_profile.budget_max:,}")
            
            if not user_profile.preferred_location or user_profile.preferred_location.lower() not in college_name.lower():
                cons.append("May not be in your preferred location")
            
            if not cons:
                cons.append("Limited networking opportunities in specialized fields")
            
            return Recommendation(
                program_name=program.get('program_name', ''),
                college=college_name,
                degree_type=program.get('degree_type', ''),
                estimated_cost=total_cost,
                fit_score=score,
                reasoning=reasoning,
                pros=pros,
                cons=cons
            )
            
        except Exception as e:
            logger.error(f"❌ Error creating recommendation: {e}")
            return None
    
    def compare_recommendations(self, rec1: Recommendation, rec2: Recommendation) -> Dict[str, Any]:
        """Generate a detailed comparison between two recommendations"""
        comparison = {
            'programs': {
                'program_1': {
                    'name': f"{rec1.program_name} at {rec1.college}",
                    'degree': rec1.degree_type,
                    'cost': rec1.estimated_cost,
                    'fit_score': rec1.fit_score
                },
                'program_2': {
                    'name': f"{rec2.program_name} at {rec2.college}",
                    'degree': rec2.degree_type,
                    'cost': rec2.estimated_cost,
                    'fit_score': rec2.fit_score
                }
            },
            'cost_difference': abs(rec1.estimated_cost - rec2.estimated_cost),
            'better_fit': rec1.program_name if rec1.fit_score > rec2.fit_score else rec2.program_name,
            'recommendation': ""
        }
        
        # Generate comparison recommendation
        if rec1.fit_score > rec2.fit_score and rec1.estimated_cost <= rec2.estimated_cost:
            comparison['recommendation'] = f"{rec1.program_name} at {rec1.college} offers better fit and value"
        elif rec2.fit_score > rec1.fit_score and rec2.estimated_cost <= rec1.estimated_cost:
            comparison['recommendation'] = f"{rec2.program_name} at {rec2.college} offers better fit and value"
        else:
            comparison['recommendation'] = "Both programs have distinct advantages - consider your priorities"
        
        return comparison
    
    def save_user_interaction(self, user_profile: UserProfile, recommendations: List[Recommendation]):
        """Save user interaction for future analysis"""
        log_file = self.data_dir / "user_logs.json"
        
        # Load existing logs
        logs = []
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # Add new interaction
        interaction = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'user_profile': {
                'interests': user_profile.interests,
                'preferred_location': user_profile.preferred_location,
                'budget_max': user_profile.budget_max,
                'career_goals': user_profile.career_goals
            },
            'recommendations_count': len(recommendations),
            'top_recommendation': recommendations[0].program_name if recommendations else None
        }
        
        logs.append(interaction)
        
        # Save updated logs
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved user interaction to {log_file}")
        except Exception as e:
            logger.error(f"❌ Error saving user interaction: {e}")

# Convenience function for quick usage
def get_cuny_advisor() -> CUNYAdvisorAgent:
    """Factory function to create and return a CUNY advisor agent"""
    return CUNYAdvisorAgent()
