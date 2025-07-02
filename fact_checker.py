OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

import os
import io
import base64
import hashlib
import requests
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import re
import asyncio
import aiohttp
from urllib.parse import quote, urlparse
import time

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import pytesseract

import openai

from bs4 import BeautifulSoup

app = FastAPI(title="AI Fact-Checker Pro", description="Advanced real-time fact-checking with AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = OPENAI_API_KEY

class EnhancedFactChecker:
    def __init__(self):
        self.fact_check_sources = [
            {"domain": "snopes.com", "weight": 0.9, "bias": "neutral"},
            {"domain": "factcheck.org", "weight": 0.85, "bias": "neutral"},
            {"domain": "politifact.com", "weight": 0.8, "bias": "slight_left"},
            {"domain": "reuters.com", "weight": 0.9, "bias": "neutral"},
            {"domain": "apnews.com", "weight": 0.85, "bias": "neutral"},
            {"domain": "bbc.com/reality-check", "weight": 0.8, "bias": "neutral"},
            {"domain": "washingtonpost.com/politics/fact-checker", "weight": 0.75, "bias": "left"},
            {"domain": "cnn.com/factsfirst", "weight": 0.7, "bias": "left"}
        ]
        
        self.misinformation_indicators = [
            "click here to learn more", "doctors hate this", "one weird trick",
            "shocking truth", "they don't want you to know", "miracle cure",
            "big pharma", "mainstream media won't tell you", "wake up",
            "do your own research", "connect the dots", "think for yourself"
        ]
        
        self.credibility_indicators = [
            "peer-reviewed", "published in", "according to researchers",
            "study shows", "clinical trial", "academic journal",
            "university research", "scientific evidence", "meta-analysis",
            "systematic review", "controlled study", "statistical analysis"
        ]
        
    def extract_text_from_image(self, image: Image.Image) -> str:
        try:
            # Enhance image for better OCR
            image = image.convert('RGB')
            # Resize if too small
            if image.width < 300 or image.height < 300:
                image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
            
            text = pytesseract.image_to_string(image, config='--psm 6')
            return text.strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def get_image_hash(self, image: Image.Image) -> str:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return hashlib.md5(img_byte_arr).hexdigest()
    
    async def enhanced_google_search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Enhanced Google Custom Search with better result parsing"""
        results = []
        
        try:
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': GOOGLE_API_KEY,
                'cx': GOOGLE_CSE_ID,
                'q': query,
                'num': min(num_results, 10),
                'safe': 'medium'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for item in data.get('items', []):
                            result = {
                                "title": item.get('title', ''),
                                "link": item.get('link', ''),
                                "snippet": item.get('snippet', ''),
                                "domain": urlparse(item.get('link', '')).netloc,
                                "fact_check_source": self.is_fact_check_source(item.get('link', '')),
                                "credibility_score": self.calculate_domain_credibility(item.get('link', ''))
                            }
                            results.append(result)
                            
        except Exception as e:
            print(f"Enhanced Google search error: {e}")
        
        return results
    
    def is_fact_check_source(self, url: str) -> bool:
        """Check if URL is from a known fact-checking source"""
        for source in self.fact_check_sources:
            if source["domain"] in url.lower():
                return True
        return False
    
    def calculate_domain_credibility(self, url: str) -> float:
        """Calculate credibility score based on domain"""
        domain = urlparse(url).netloc.lower()
        
        # Known fact-check sources
        for source in self.fact_check_sources:
            if source["domain"] in domain:
                return source["weight"]
        
        # News domains scoring
        high_credibility = ["reuters.com", "apnews.com", "bbc.com", "npr.org", "pbs.org"]
        medium_credibility = ["cnn.com", "foxnews.com", "nytimes.com", "washingtonpost.com", "wsj.com"]
        academic = [".edu", ".gov", ".org"]
        
        if any(hc in domain for hc in high_credibility):
            return 0.8
        elif any(mc in domain for mc in medium_credibility):
            return 0.6
        elif any(ac in domain for ac in academic):
            return 0.85
        elif "wikipedia.org" in domain:
            return 0.7
        else:
            return 0.4
    
    async def reverse_image_search(self, image_hash: str, query: str = "") -> List[Dict]:
        """Enhanced reverse image search"""
        try:
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': GOOGLE_API_KEY,
                'cx': GOOGLE_CSE_ID,
                'q': f"{query} image verification fact check" if query else "image verification fact check",
                'searchType': 'image',
                'num': 8,
                'safe': 'medium'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            {
                                "source": item.get('displayLink', 'Unknown'),
                                "title": item.get('title', 'No title'),
                                "url": item.get('link', ''),
                                "context": item.get('snippet', ''),
                                "credibility": self.calculate_domain_credibility(item.get('link', ''))
                            }
                            for item in data.get('items', [])[:5]
                        ]
        except Exception as e:
            print(f"Reverse search error: {e}")
        
        return []
    
    async def comprehensive_fact_check_search(self, content: str) -> Dict:
        """Comprehensive fact-checking across multiple sources"""
        queries = [
            f'"{content[:100]}" fact check',
            f"{content[:50]} misinformation",
            f"{content[:50]} debunked",
            f"{content[:50]} verified true false"
        ]
        
        all_results = []
        fact_check_hits = 0
        credibility_sum = 0
        
        for query in queries:
            results = await self.enhanced_google_search(query, 5)
            all_results.extend(results)
            
            for result in results:
                if result["fact_check_source"]:
                    fact_check_hits += 1
                credibility_sum += result["credibility_score"]
        
        # Analyze results
        avg_credibility = credibility_sum / len(all_results) if all_results else 0.4
        fact_check_ratio = fact_check_hits / len(all_results) if all_results else 0
        
        # Look for debunking indicators
        debunk_indicators = ["false", "debunked", "fake", "misleading", "incorrect"]
        verify_indicators = ["true", "verified", "confirmed", "accurate", "factual"]
        
        debunk_count = 0
        verify_count = 0
        
        for result in all_results:
            text = (result["title"] + " " + result["snippet"]).lower()
            debunk_count += sum(1 for indicator in debunk_indicators if indicator in text)
            verify_count += sum(1 for indicator in verify_indicators if indicator in text)
        
        return {
            "results": all_results[:10],  # Top 10 results
            "fact_check_hits": fact_check_hits,
            "fact_check_ratio": fact_check_ratio,
            "avg_credibility": avg_credibility,
            "debunk_signals": debunk_count,
            "verify_signals": verify_count,
            "total_results": len(all_results)
        }
    
    async def analyze_with_llm(self, content: str, search_context: str = "", image_context: str = "") -> Dict:
        """Enhanced LLM analysis using free Ollama"""
        try:
            context_info = ""
            if search_context:
                context_info += f"\nSearch results context: {search_context}"
            if image_context:
                context_info += f"\nImage text content: {image_context}"

            prompt = f"""
            You are an expert fact-checker and misinformation analyst. Analyze the following content for factual accuracy, credibility, and potential misinformation patterns.

            CONTENT TO ANALYZE: "{content}"
            {context_info}

            Perform a comprehensive analysis considering:
            1. Factual accuracy and verifiability
            2. Source credibility indicators
            3. Misinformation patterns and red flags
            4. Language analysis (sensational, clickbait, etc.)
            5. Logical consistency and evidence quality
            6. Context from search results if provided

            Provide your assessment in this exact JSON format:
            {{
                "confidence": <number 0-100>,
                "status": "<verified|fake|misleading|uncertain>",
                "reasoning": "<detailed 2-3 sentence explanation>",
                "evidence": ["<specific evidence point 1>", "<specific evidence point 2>", "<specific evidence point 3>"],
                "risk_factors": ["<risk factor 1>", "<risk factor 2>"],
                "credibility_signals": ["<positive signal 1>", "<positive signal 2>"]
            }}

            Be precise with confidence scores:
            - 80-100: Strong evidence supporting assessment
            - 60-79: Good evidence with some uncertainty
            - 40-59: Mixed signals, leaning toward assessment
            - 20-39: Weak evidence, high uncertainty
            - 0-19: Very likely opposite of stated assessment
            """

            # Make request to local Ollama server
            response = requests.post('http://localhost:11434/api/generate',
                json={
                    'model': 'llama2',  # or whatever model you installed
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.2,
                        'num_predict': 700
                    }
                }
            )

            if response.status_code == 200:
                result_text = response.json()['response']
            else:
                raise Exception(f"Ollama request failed: {response.status_code}")

            # Try to parse JSON
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return self.validate_llm_result(result)
                else:
                    return self.parse_llm_response_advanced(result_text)
            except json.JSONDecodeError:
                return self.parse_llm_response_advanced(result_text)

        except Exception as e:
            print(f"LLM Analysis Error: {e}")
            return self.enhanced_fallback_analysis(content)

    def validate_llm_result(self, result: Dict) -> Dict:
        """Validate and clean LLM result"""
        return {
            "confidence": max(0, min(100, result.get("confidence", 50))),
            "status": result.get("status", "uncertain"),
            "reasoning": result.get("reasoning", "Analysis completed")[:500],
            "evidence": result.get("evidence", ["Analysis performed"])[:5],
            "risk_factors": result.get("risk_factors", [])[:3],
            "credibility_signals": result.get("credibility_signals", [])[:3]
        }

    def parse_llm_response_advanced(self, response_text: str) -> Dict:
        """Advanced parsing of LLM response when JSON fails"""
        confidence = 50
        status = "uncertain"
        evidence = ["Analysis completed"]
        reasoning = response_text[:200]

        # Extract confidence
        conf_patterns = [r'confidence[:\s]+(\d+)', r'(\d+)%', r'score[:\s]+(\d+)']
        for pattern in conf_patterns:
            match = re.search(pattern, response_text.lower())
            if match:
                confidence = int(match.group(1))
                break
            
        # Determine status
        response_lower = response_text.lower()
        if any(word in response_lower for word in ['false', 'fake', 'misleading', 'debunked']):
            if confidence < 50:
                confidence = 100 - confidence  # Flip confidence if it's indicating false
            status = "fake"
        elif any(word in response_lower for word in ['true', 'verified', 'accurate', 'confirmed']):
            status = "verified"
        elif any(word in response_lower for word in ['misleading', 'partial', 'mixed']):
            status = "misleading"

        return {
            "confidence": confidence,
            "status": status,
            "reasoning": reasoning,
            "evidence": evidence,
            "risk_factors": [],
            "credibility_signals": []
        }

    def enhanced_fallback_analysis(self, content: str) -> Dict:
        """Significantly improved fallback analysis"""
        content_lower = content.lower()
        
        # Scoring system
        credibility_score = 50  # Start neutral
        risk_factors = []
        credibility_signals = []
        
        # Check for misinformation indicators
        misinfo_count = sum(1 for indicator in self.misinformation_indicators if indicator in content_lower)
        credibility_score -= misinfo_count * 15
        if misinfo_count > 0:
            risk_factors.extend([f"Contains {misinfo_count} misinformation indicators"])
        
        # Check for credibility indicators
        credible_count = sum(1 for indicator in self.credibility_indicators if indicator in content_lower)
        credibility_score += credible_count * 12
        if credible_count > 0:
            credibility_signals.extend([f"Contains {credible_count} credibility indicators"])
        
        # Language analysis
        exclamation_count = content.count('!')
        caps_ratio = sum(1 for c in content if c.isupper()) / len(content) if content else 0
        
        if exclamation_count > 3:
            credibility_score -= 10
            risk_factors.append("Excessive use of exclamation marks")
        
        if caps_ratio > 0.1:
            credibility_score -= 15
            risk_factors.append("Excessive use of capital letters")
        
        # URL and source analysis
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        if urls:
            avg_url_credibility = sum(self.calculate_domain_credibility(url) for url in urls) / len(urls)
            credibility_score += (avg_url_credibility - 0.5) * 30
            if avg_url_credibility > 0.7:
                credibility_signals.append("Links to credible sources")
            elif avg_url_credibility < 0.4:
                risk_factors.append("Links to low-credibility sources")
        
        # Length and complexity analysis
        if len(content) < 50:
            credibility_score -= 10
            risk_factors.append("Very short content")
        elif len(content) > 500:
            credibility_score += 5
            credibility_signals.append("Detailed content")
        
        # Finalize score
        credibility_score = max(5, min(95, credibility_score))
        
        # Determine status
        if credibility_score >= 70:
            status = "verified"
        elif credibility_score <= 30:
            status = "fake"
        elif misinfo_count > credible_count:
            status = "misleading"
        else:
            status = "uncertain"
        
        reasoning = f"Fallback analysis based on content patterns. Credibility indicators: {credible_count}, Risk indicators: {misinfo_count}, Language analysis score: {int(credibility_score)}"
        
        return {
            "confidence": int(credibility_score),
            "status": status,
            "reasoning": reasoning,
            "evidence": [f"Pattern-based analysis completed", f"Analyzed {len(content)} characters"],
            "risk_factors": risk_factors[:3],
            "credibility_signals": credibility_signals[:3]
        }
    
    def extract_claims(self, content: str) -> List[str]:
        """Enhanced claim extraction"""
        # Split by sentences and filter
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if len(s.strip()) > 15]
        
        # Prioritize sentences with factual indicators
        factual_indicators = ['said', 'reported', 'according to', 'study', 'research', 'data', 'statistics']
        
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for indicator in factual_indicators if indicator.lower() in sentence.lower())
            scored_sentences.append((sentence, score))
        
        # Sort by score and return top claims
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored_sentences[:4]]
    
    async def comprehensive_analysis(self, content: str, image: Optional[Image.Image] = None) -> Dict:
        """Main analysis function with parallel processing"""
        start_time = time.time()
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "content_analyzed": content[:300] + "..." if len(content) > 300 else content,
            "image_provided": image is not None,
            "analysis_duration": 0
        }
        
        # Extract text from image if provided
        image_text = ""
        if image:
            image_text = self.extract_text_from_image(image)
            if image_text:
                content = f"{content} {image_text}".strip()
                results["extracted_image_text"] = image_text[:200]
        
        # Run analyses in parallel
        tasks = []
        
        # Fact-checking search
        search_task = asyncio.create_task(self.comprehensive_fact_check_search(content))
        tasks.append(("search", search_task))
        
        # Reverse image search if image provided
        if image:
            image_hash = self.get_image_hash(image)
            reverse_search_task = asyncio.create_task(self.reverse_image_search(image_hash, content[:50]))
            tasks.append(("reverse_search", reverse_search_task))
        
        # Execute parallel tasks
        completed_tasks = {}
        for name, task in tasks:
            try:
                completed_tasks[name] = await task
            except Exception as e:
                print(f"Task {name} failed: {e}")
                completed_tasks[name] = {}
        
        # Prepare context for LLM
        search_context = ""
        if "search" in completed_tasks:
            search_data = completed_tasks["search"]
            context_snippets = [r["snippet"] for r in search_data.get("results", [])[:3]]
            search_context = " | ".join(context_snippets)
        
        # LLM Analysis with context
        try:
            llm_result = await self.analyze_with_llm(content, search_context, image_text)
        except:
            llm_result = self.enhanced_fallback_analysis(content)
        
        # Combine results
        results.update(llm_result)
        
        # Add search results
        if "search" in completed_tasks:
            search_data = completed_tasks["search"]
            results["fact_check_sources"] = search_data.get("results", [])[:8]
            results["search_analysis"] = {
                "fact_check_hits": search_data.get("fact_check_hits", 0),
                "credibility_score": search_data.get("avg_credibility", 0.4),
                "debunk_signals": search_data.get("debunk_signals", 0),
                "verify_signals": search_data.get("verify_signals", 0)
            }
            
            # Adjust confidence based on search results
            if search_data.get("fact_check_hits", 0) > 0:
                if search_data.get("debunk_signals", 0) > search_data.get("verify_signals", 0):
                    results["confidence"] = max(results["confidence"], 75)
                    if results["status"] == "uncertain":
                        results["status"] = "fake"
                elif search_data.get("verify_signals", 0) > search_data.get("debunk_signals", 0):
                    results["confidence"] = max(results["confidence"], 70)
                    if results["status"] == "uncertain":
                        results["status"] = "verified"
        
        # Add reverse search results
        if "reverse_search" in completed_tasks:
            results["reverse_search"] = completed_tasks["reverse_search"]
        
        # Extract claims
        results["extracted_claims"] = self.extract_claims(content)
        
        # Calculate final analysis duration
        results["analysis_duration"] = round(time.time() - start_time, 2)
        
        return results

# Initialize fact checker
fact_checker = EnhancedFactChecker()

@app.post("/analyze")
async def analyze_content(
    text_content: str = Form(""),
    image: Optional[UploadFile] = File(None)
):
    if not text_content.strip() and not image:
        raise HTTPException(status_code=400, detail="Please provide text content or upload an image")
    
    try:
        processed_image = None
        if image:
            image_content = await image.read()
            processed_image = Image.open(io.BytesIO(image_content))
        
        results = await fact_checker.comprehensive_analysis(text_content, processed_image)
        
        return {
            "success": True,
            "results": results
        }
    
    except Exception as e:
        print(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Serve the enhanced web interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FactCheck Pro</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --primary-dark: #4f46e5;
                --secondary: #f1f5f9;
                --success: #10b981;
                --warning: #f59e0b;
                --danger: #ef4444;
                --text-primary: #0f172a;
                --text-secondary: #64748b;
                --border: #e2e8f0;
                --bg-overlay: rgba(255, 255, 255, 0.9);
                --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
                --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                min-height: 100vh;
                line-height: 1.6;
                color: var(--text-primary);
            }

            .container {
                max-width: 900px;
                margin: 0 auto;
                padding: 2rem 1rem;
            }

            .header {
                text-align: center;
                margin-bottom: 3rem;
            }

            .logo {
                display: inline-flex;
                align-items: center;
                gap: 0.75rem;
                margin-bottom: 1rem;
            }

            .logo-icon {
                width: 40px;
                height: 40px;
                background: var(--primary);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 1.5rem;
                font-weight: 600;
            }

            .logo-text {
                font-size: 2rem;
                font-weight: 700;
                color: white;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            .tagline {
                color: rgba(255, 255, 255, 0.9);
                font-size: 1.1rem;
                font-weight: 400;
            }

            .main-card {
                background: var(--bg-overlay);
                backdrop-filter: blur(20px);
                border-radius: 24px;
                box-shadow: var(--shadow-lg);
                border: 1px solid rgba(255, 255, 255, 0.2);
                overflow: hidden;
            }

            .card-header {
                padding: 2rem 2rem 0;
            }

            .status-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-bottom: 2rem;
            }

            .status-item {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 1rem;
                background: var(--secondary);
                border-radius: 12px;
                font-size: 0.9rem;
            }

            .status-icon {
                width: 20px;
                height: 20px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.8rem;
                font-weight: 600;
            }

            .status-ready { background: var(--success); color: white; }
            .status-warning { background: var(--warning); color: white; }

            .input-section {
                padding: 0 2rem;
            }

            .upload-area {
                border: 2px dashed var(--border);
                border-radius: 16px;
                padding: 3rem 2rem;
                text-align: center;
                margin-bottom: 2rem;
                transition: all 0.3s ease;
                cursor: pointer;
                position: relative;
            }

            .upload-area:hover {
                border-color: var(--primary);
                background: rgba(99, 102, 241, 0.02);
            }

            .upload-area.dragover {
                border-color: var(--primary);
                background: rgba(99, 102, 241, 0.05);
                transform: scale(1.02);
            }

            .upload-icon {
                width: 48px;
                height: 48px;
                background: var(--primary);
                border-radius: 12px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 1.5rem;
                margin-bottom: 1rem;
            }

            .upload-text {
                font-size: 1.1rem;
                font-weight: 500;
                margin-bottom: 0.5rem;
            }

            .upload-subtext {
                color: var(--text-secondary);
                font-size: 0.9rem;
            }

            .file-input {
                display: none;
            }

            .text-area {
                width: 100%;
                min-height: 120px;
                padding: 1.5rem;
                border: 2px solid var(--border);
                border-radius: 16px;
                font-family: inherit;
                font-size: 1rem;
                resize: vertical;
                transition: all 0.3s ease;
                background: white;
            }

            .text-area:focus {
                outline: none;
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
            }

            .text-area::placeholder {
                color: var(--text-secondary);
            }

            .analyze-btn {
                width: 100%;
                padding: 1.25rem 2rem;
                background: linear-gradient(135deg, var(--primary), var(--primary-dark));
                color: white;
                border: none;
                border-radius: 16px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin: 2rem;
                position: relative;
                overflow: hidden;
            }

            .analyze-btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: var(--shadow-lg);
            }

            .analyze-btn:disabled {
                background: var(--text-secondary);
                cursor: not-allowed;
                transform: none;
            }

            .btn-text {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.5rem;
            }

            .loading {
                display: none;
                padding: 3rem 2rem;
                text-align: center;
            }

            .loading.show {
                display: block;
            }

            .spinner {
                width: 40px;
                height: 40px;
                border: 3px solid var(--secondary);
                border-top: 3px solid var(--primary);
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 1rem;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .results {
                display: none;
                padding: 0 2rem 2rem;
            }

            .results.show {
                display: block;
            }

            .result-card {
                background: white;
                border-radius: 20px;
                padding: 2rem;
                margin-bottom: 1.5rem;
                box-shadow: var(--shadow);
                border-left: 4px solid var(--primary);
            }

            .result-card.verified {
                border-left-color: var(--success);
            }

            .result-card.fake {
                border-left-color: var(--danger);
            }

            .result-card.misleading {
                border-left-color: var(--warning);
            }

            .result-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 1.5rem;
                flex-wrap: wrap;
                gap: 1rem;
            }

            .status-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem 1rem;
                border-radius: 25px;
                font-size: 0.9rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.025em;
            }

            .status-badge.verified {
                background: rgba(16, 185, 129, 0.1);
                color: var(--success);
            }

            .status-badge.fake {
                background: rgba(239, 68, 68, 0.1);
                color: var(--danger);
            }

            .status-badge.misleading {
                background: rgba(245, 158, 11, 0.1);
                color: var(--warning);
            }

            .status-badge.uncertain {
                background: rgba(100, 116, 139, 0.1);
                color: var(--text-secondary);
            }

            .confidence-score {
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--text-primary);
            }

            .confidence-bar {
                width: 100%;
                height: 8px;
                background: var(--secondary);
                border-radius: 4px;
                overflow: hidden;
                margin: 1rem 0;
            }

            .confidence-fill {
                height: 100%;
                border-radius: 4px;
                transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
            }

            .confidence-fill.high {
                background: linear-gradient(90deg, var(--success), #059669);
            }

            .confidence-fill.medium {
                background: linear-gradient(90deg, var(--warning), #d97706);
            }

            .confidence-fill.low {
                background: linear-gradient(90deg, var(--danger), #dc2626);
            }

            .info-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                margin-top: 2rem;
            }

            .info-section {
                background: var(--secondary);
                border-radius: 12px;
                padding: 1.5rem;
            }

            .info-title {
                font-size: 1rem;
                font-weight: 600;
                margin-bottom: 1rem;
                color: var(--text-primary);
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .evidence-list {
                list-style: none;
            }

            .evidence-item {
                background: white;
                padding: 0.75rem 1rem;
                margin: 0.5rem 0;
                border-radius: 8px;
                border-left: 3px solid var(--primary);
                font-size: 0.9rem;
                line-height: 1.5;
            }

            .evidence-item.risk {
                border-left-color: var(--danger);
            }

            .evidence-item.positive {
                border-left-color: var(--success);
            }

            .source-link {
                color: var(--primary);
                text-decoration: none;
                font-weight: 500;
                font-size: 0.9rem;
            }

            .source-link:hover {
                text-decoration: underline;
            }

            .meta-info {
                margin-top: 2rem;
                padding-top: 1.5rem;
                border-top: 1px solid var(--border);
                font-size: 0.85rem;
                color: var(--text-secondary);
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 1rem;
            }

            .file-preview {
                display: none;
                background: var(--secondary);
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 1rem;
                text-align: center;
            }

            .file-preview.show {
                display: block;
            }

            .file-preview img {
                max-width: 200px;
                max-height: 150px;
                border-radius: 8px;
                margin-bottom: 0.5rem;
            }

            @media (max-width: 768px) {
                .container {
                    padding: 1rem 0.5rem;
                }

                .main-card {
                    border-radius: 16px;
                }

                .card-header {
                    padding: 1.5rem 1rem 0;
                }

                .input-section {
                    padding: 0 1rem;
                }

                .analyze-btn {
                    margin: 1.5rem 1rem;
                }

                .results {
                    padding: 0 1rem 1.5rem;
                }

                .result-card {
                    padding: 1.5rem;
                }

                .status-grid {
                    grid-template-columns: 1fr;
                }

                .info-grid {
                    grid-template-columns: 1fr;
                }

                .result-header {
                    flex-direction: column;
                    align-items: flex-start;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">
                    <div class="logo-icon">✓</div>
                    <div class="logo-text">FactCheck Pro</div>
                </div>
                <div class="tagline">Advanced AI-powered fact verification</div>
            </div>

            <div class="main-card">
                <div class="card-header">
                    <div class="status-grid">
                        <div class="status-item">
                            <div class="status-icon status-ready">✓</div>
                            <span>OCR Engine Ready</span>
                        </div>
                        <div class="status-item">
                            <div class="status-icon status-warning">⚠</div>
                            <span>OpenAI API Key Required</span>
                        </div>
                        <div class="status-item">
                            <div class="status-icon status-warning">⚠</div>
                            <span>Google API Key Required</span>
                        </div>
                        <div class="status-item">
                            <div class="status-icon status-ready">✓</div>
                            <span>Fact-Check Sources Ready</span>
                        </div>
                    </div>
                </div>

                <div class="input-section">
                    <div class="upload-area" id="uploadArea">
                        <input type="file" id="fileInput" class="file-input" accept="image/*">
                        <div class="upload-icon">📷</div>
                        <div class="upload-text">Upload an image to analyze</div>
                        <div class="upload-subtext">Drag & drop or click to select • PNG, JPG, WEBP</div>
                    </div>

                    <div class="file-preview" id="filePreview">
                        <img id="previewImage" alt="Preview">
                        <div id="previewText"></div>
                    </div>

                    <textarea 
                        id="textInput" 
                        class="text-area" 
                        placeholder="Paste news text, claims, or any content you want to fact-check..."
                    ></textarea>
                </div>

                <button class="analyze-btn" id="analyzeBtn">
                    <span class="btn-text">
                        <span>🔍</span>
                        <span>Analyze Content</span>
                    </span>
                </button>

                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <div>Performing comprehensive analysis...</div>
                    <div style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 0.5rem;">
                        Checking sources, analyzing patterns, verifying claims
                    </div>
                </div>

                <div class="results" id="results">
                    <div id="resultsContainer"></div>
                </div>
            </div>
        </div>

        <script>
            let currentFile = null;

            // File upload handling
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const filePreview = document.getElementById('filePreview');
            const previewImage = document.getElementById('previewImage');
            const previewText = document.getElementById('previewText');

            // Drag and drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            });

            uploadArea.addEventListener('click', () => {
                fileInput.click();
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFile(e.target.files[0]);
                }
            });

            function handleFile(file) {
                if (!file.type.startsWith('image/')) {
                    alert('Please select an image file');
                    return;
                }

                currentFile = file;
                
                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImage.src = e.target.result;
                    previewText.textContent = file.name;
                    filePreview.classList.add('show');
                };
                reader.readAsDataURL(file);

                // Update upload area
                uploadArea.innerHTML = `
                    <div class="upload-icon">✅</div>
                    <div class="upload-text">Image selected: ${file.name}</div>
                    <div class="upload-subtext">Click to change image</div>
                `;
            }

            // Analysis function
            async function analyzeContent() {
                const textContent = document.getElementById('textInput').value.trim();
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');
                const analyzeBtn = document.getElementById('analyzeBtn');

                if (!textContent && !currentFile) {
                    alert('Please provide text content or upload an image');
                    return;
                }

                // Show loading state
                loading.classList.add('show');
                results.classList.remove('show');
                analyzeBtn.disabled = true;

                try {
                    const formData = new FormData();
                    formData.append('text_content', textContent);
                    if (currentFile) {
                        formData.append('image', currentFile);
                    }

                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const data = await response.json();
                    displayResults(data.results);

                } catch (error) {
                    console.error('Analysis error:', error);
                    alert(`Error during analysis: ${error.message}`);
                } finally {
                    loading.classList.remove('show');
                    analyzeBtn.disabled = false;
                }
            }

            function displayResults(result) {
                const resultsContainer = document.getElementById('resultsContainer');
                const results = document.getElementById('results');

                const confidenceClass = result.confidence >= 70 ? 'high' : result.confidence >= 40 ? 'medium' : 'low';
                const statusClass = result.status;

                const evidenceItems = result.evidence.map(item => 
                    `<li class="evidence-item">${item}</li>`
                ).join('');

                const riskItems = (result.risk_factors || []).map(item => 
                    `<li class="evidence-item risk">⚠️ ${item}</li>`
                ).join('');

                const credibilityItems = (result.credibility_signals || []).map(item => 
                    `<li class="evidence-item positive">✅ ${item}</li>`
                ).join('');

                const claimsItems = (result.extracted_claims || []).map(claim => 
                    `<li class="evidence-item">${claim}</li>`
                ).join('');

                const sourcesItems = (result.fact_check_sources || []).slice(0, 5).map(source => 
                    `<li class="evidence-item">
                        <a href="${source.link}" target="_blank" class="source-link">${source.title}</a>
                        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem;">
                            ${source.domain} • Credibility: ${(source.credibility_score * 100).toFixed(0)}%
                        </div>
                    </li>`
                ).join('');

                let searchAnalysis = '';
                if (result.search_analysis) {
                    const sa = result.search_analysis;
                    searchAnalysis = `
                        <div class="info-section">
                            <div class="info-title">🔍 Search Analysis</div>
                            <ul class="evidence-list">
                                <li class="evidence-item">Fact-check sources found: ${sa.fact_check_hits}</li>
                                <li class="evidence-item">Average source credibility: ${(sa.credibility_score * 100).toFixed(0)}%</li>
                                <li class="evidence-item">Debunking signals: ${sa.debunk_signals}</li>
                                <li class="evidence-item">Verification signals: ${sa.verify_signals}</li>
                            </ul>
                        </div>
                    `;
                }

                const resultHTML = `
                    <div class="result-card ${statusClass}">
                        <div class="result-header">
                            <div class="status-badge ${statusClass}">
                                ${getStatusIcon(statusClass)} ${statusClass.toUpperCase()}
                            </div>
                            <div class="confidence-score">${result.confidence}%</div>
                        </div>
                        
                        <div class="confidence-bar">
                            <div class="confidence-fill ${confidenceClass}" style="width: ${result.confidence}%"></div>
                        </div>

                        <div style="margin-bottom: 1.5rem;">
                            <strong>Analysis:</strong> ${result.reasoning}
                        </div>

                        <div class="info-grid">
                            <div class="info-section">
                                <div class="info-title">📋 Evidence Found</div>
                                <ul class="evidence-list">
                                    ${evidenceItems}
                                </ul>
                            </div>

                            ${riskItems ? `
                            <div class="info-section">
                                <div class="info-title">⚠️ Risk Factors</div>
                                <ul class="evidence-list">
                                    ${riskItems}
                                </ul>
                            </div>
                            ` : ''}

                            ${credibilityItems ? `
                            <div class="info-section">
                                <div class="info-title">✅ Credibility Signals</div>
                                <ul class="evidence-list">
                                    ${credibilityItems}
                                </ul>
                            </div>
                            ` : ''}

                            ${claimsItems ? `
                            <div class="info-section">
                                <div class="info-title">🎯 Key Claims</div>
                                <ul class="evidence-list">
                                    ${claimsItems}
                                </ul>
                            </div>
                            ` : ''}

                            ${sourcesItems ? `
                            <div class="info-section">
                                <div class="info-title">🔗 Fact-Check Sources</div>
                                <ul class="evidence-list">
                                    ${sourcesItems}
                                </ul>
                            </div>
                            ` : ''}

                            ${searchAnalysis}
                        </div>

                        <div class="meta-info">
                            <span>Analysis completed: ${new Date(result.timestamp).toLocaleString()}</span>
                            <span>Duration: ${result.analysis_duration}s</span>
                        </div>
                    </div>
                `;

                resultsContainer.innerHTML = resultHTML;
                results.classList.add('show');
                results.scrollIntoView({ behavior: 'smooth', block: 'start' });

                // Animate confidence bar
                setTimeout(() => {
                    const confidenceFill = document.querySelector('.confidence-fill');
                    if (confidenceFill) {
                        confidenceFill.style.width = result.confidence + '%';
                    }
                }, 100);
            }

            function getStatusIcon(status) {
                switch (status) {
                    case 'verified': return '✅';
                    case 'fake': return '❌';
                    case 'misleading': return '⚠️';
                    default: return '❓';
                }
            }

            // Set up analyze button
            document.getElementById('analyzeBtn').addEventListener('click', analyzeContent);

            // Allow Enter key in textarea to trigger analysis
            document.getElementById('textInput').addEventListener('keydown', (e) => {
                if (e.ctrlKey && e.key === 'Enter') {
                    analyzeContent();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced AI Fact-Checker Server...")
    print("📋 Required dependencies:")
    print("   pip install fastapi uvicorn pytesseract pillow openai requests beautifulsoup4 python-multipart aiohttp")
    print("🔑 Configure your API keys in the code before running!")
    print("🌐 Server will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)