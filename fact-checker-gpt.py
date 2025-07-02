import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

LLM_PROVIDER = "local"  # "openai" or "local"
OLLAMA_MODEL = "llama3"
OPENAI_MODEL = "gpt-4o"
OLLAMA_BASE_URL = "http://localhost:11434"

import io
import base64
import hashlib
import requests
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import json
import re
import asyncio
import aiohttp
from urllib.parse import quote, urlparse
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import easyocr

import openai
from openai import OpenAI

from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

app = FastAPI(title="ValiData", description="Advanced, reliable, and fast fact-checking with AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if LLM_PROVIDER == "openai":
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None

ocr_reader = easyocr.Reader(['en'])

class LLMManager:
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.openai_client = openai_client
        self.ollama_base_url = OLLAMA_BASE_URL
        self.openai_model = OPENAI_MODEL
        self.ollama_model = OLLAMA_MODEL
    
    async def generate_response(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
        if self.provider == "openai":
            return await self._openai_generate(prompt, max_tokens, temperature)
        else:
            return await self._ollama_generate(prompt, max_tokens, temperature)
    
    async def _openai_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def _ollama_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        'model': self.ollama_model,
                        'prompt': prompt,
                        'stream': False,
                        'options': {
                            'temperature': temperature,
                            'num_predict': max_tokens
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['response']
                    else:
                        error_body = await response.text()
                        raise Exception(f"Ollama request failed: {response.status} - {error_body}")
        except asyncio.TimeoutError:
            raise Exception(f"Ollama API timeout error: The request took longer than 120 seconds.")
        except Exception as e:
                raise Exception(f"Ollama API error: {str(e)}")

class AdvancedImageProcessor:
    def __init__(self):
        self.ocr_engines = {
            'tesseract': self._tesseract_extract,
            'easyocr': self._easyocr_extract
        }
    
    def preprocess_image(self, image: Image.Image) -> List[Image.Image]:
        processed_images = []
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        processed_images.append(Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
        
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        processed_images.append(Image.fromarray(enhanced))
        
        denoised = cv2.fastNlMeansDenoising(gray)
        processed_images.append(Image.fromarray(denoised))
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(Image.fromarray(binary))
        
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel)
        processed_images.append(Image.fromarray(morphed))
        
        scaled_images = []
        for proc_img in processed_images:
            if proc_img.width < 800 or proc_img.height < 600:
                scale_factor = max(800/proc_img.width, 600/proc_img.height)
                new_size = (int(proc_img.width * scale_factor), int(proc_img.height * scale_factor))
                scaled_images.append(proc_img.resize(new_size, Image.Resampling.LANCZOS))
            else:
                scaled_images.append(proc_img)
        
        return scaled_images
    
    def _tesseract_extract(self, image: Image.Image) -> str:
        configs = [
            '--psm 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?-',
            '--psm 6',
            '--psm 8',
            '--psm 11',
            '--psm 12'
        ]
        
        texts = []
        for config in configs:
            try:
                text = pytesseract.image_to_string(image, config=config)
                if text.strip():
                    texts.append(text.strip())
            except:
                continue
        
        return max(texts, key=len) if texts else ""
    
    def _easyocr_extract(self, image: Image.Image) -> str:
        try:
            img_array = np.array(image)
            results = ocr_reader.readtext(img_array, detail=0, paragraph=True)
            return ' '.join(results) if results else ""
        except:
            return ""
    
    def extract_text_comprehensive(self, image: Image.Image) -> Dict[str, str]:
        processed_images = self.preprocess_image(image)
        all_extractions = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for i, proc_img in enumerate(processed_images[:3]):
                for engine_name, engine_func in self.ocr_engines.items():
                    future = executor.submit(engine_func, proc_img)
                    futures.append((f"{engine_name}_v{i}", future))
            
            for name, future in futures:
                try:
                    result = future.result(timeout=30)
                    if result and len(result.strip()) > 5:
                        all_extractions[name] = result.strip()
                except:
                    continue
        
        if not all_extractions:
            return {"text": "", "confidence": 0.0, "method": "none"}
        
        best_extraction = max(all_extractions.items(), key=lambda x: len(x[1]))
        
        return {
            "text": best_extraction[1],
            "confidence": min(100.0, len(best_extraction[1]) / 10),
            "method": best_extraction[0],
            "all_extractions": all_extractions
        }
    
    def analyze_image_metadata(self, image: Image.Image) -> Dict[str, Any]:
        metadata = {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "has_exif": bool(getattr(image, '_getexif', lambda: None)()),
            "color_analysis": self._analyze_colors(image),
            "quality_score": self._assess_quality(image)
        }
        
        if hasattr(image, '_getexif') and image._getexif():
            try:
                exif = image._getexif()
                metadata["exif_data"] = dict(exif) if exif else {}
            except:
                metadata["exif_data"] = {}
        
        return metadata
    
    def _analyze_colors(self, image: Image.Image) -> Dict[str, Any]:
        img_array = np.array(image.convert('RGB'))
        
        return {
            "dominant_colors": self._get_dominant_colors(img_array),
            "brightness": float(np.mean(img_array)),
            "contrast": float(np.std(img_array))
        }
    
    def _get_dominant_colors(self, img_array: np.ndarray, k: int = 5) -> List[List[int]]:
        pixels = img_array.reshape(-1, 3)
        
        from sklearn.cluster import KMeans
        try:
            kmeans = KMeans(n_clusters=k, n_init=10)
            kmeans.fit(pixels)
            return kmeans.cluster_centers_.astype(int).tolist()
        except:
            return [[128, 128, 128]]
    
    def _assess_quality(self, image: Image.Image) -> float:
        img_array = np.array(image.convert('L'))
        
        laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
        
        quality_score = min(100.0, laplacian_var / 10)
        
        return quality_score

import asyncio
import aiohttp
import json
import re
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Set
from urllib.parse import urlparse, quote
from collections import defaultdict, Counter
import hashlib
import math

class EnhancedSearchEngine:
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.search_cache = {}
        self.cache_duration = 3600  # 1 hour
        
        # Enhanced source credibility matrix
        self.credibility_matrix = {
            # Tier 1: Premium fact-checking and authoritative sources
            "snopes.com": {"weight": 0.95, "bias": "neutral", "type": "fact_check", "tier": 1},
            "factcheck.org": {"weight": 0.93, "bias": "neutral", "type": "fact_check", "tier": 1},
            "reuters.com": {"weight": 0.92, "bias": "neutral", "type": "news", "tier": 1},
            "apnews.com": {"weight": 0.91, "bias": "neutral", "type": "news", "tier": 1},
            "nature.com": {"weight": 0.96, "bias": "neutral", "type": "academic", "tier": 1},
            "science.org": {"weight": 0.95, "bias": "neutral", "type": "academic", "tier": 1},
            "nejm.org": {"weight": 0.94, "bias": "neutral", "type": "academic", "tier": 1},
            
            # Tier 2: High-quality sources
            "bbc.com": {"weight": 0.88, "bias": "neutral", "type": "news", "tier": 2},
            "npr.org": {"weight": 0.85, "bias": "slight_left", "type": "news", "tier": 2},
            "pbs.org": {"weight": 0.87, "bias": "neutral", "type": "news", "tier": 2},
            "politifact.com": {"weight": 0.83, "bias": "slight_left", "type": "fact_check", "tier": 2},
            "afp.com": {"weight": 0.86, "bias": "neutral", "type": "news", "tier": 2},
            
            # Tier 3: Mainstream sources
            "washingtonpost.com": {"weight": 0.78, "bias": "left", "type": "news", "tier": 3},
            "nytimes.com": {"weight": 0.80, "bias": "left", "type": "news", "tier": 3},
            "wsj.com": {"weight": 0.82, "bias": "slight_right", "type": "news", "tier": 3},
            "economist.com": {"weight": 0.84, "bias": "slight_right", "type": "news", "tier": 3},
            
            # Tier 4: Lower reliability
            "cnn.com": {"weight": 0.72, "bias": "left", "type": "news", "tier": 4},
            "foxnews.com": {"weight": 0.65, "bias": "right", "type": "news", "tier": 4},
        }
        
        # Enhanced pattern recognition
        self.content_patterns = {
            'high_credibility_indicators': [
                r'peer[- ]reviewed', r'published in.*journal', r'clinical trial', r'meta[- ]analysis',
                r'systematic review', r'scientific study', r'research shows', r'according to.*study',
                r'data from.*survey', r'longitudinal study', r'randomized.*trial', r'evidence[- ]based'
            ],
            'misinformation_indicators': [
                r'doctors hate', r'one weird trick', r'shocking truth', r'they don\'t want you',
                r'big pharma.*hide', r'mainstream media.*ignore', r'wake up.*people',
                r'do your own research', r'question everything', r'\\bhoax\\b', r'false flag',
                r'conspiracy', r'cover[- ]up', r'\\bfake\\b.*news'
            ],
            'verification_indicators': [
                r'verified by', r'confirmed by', r'fact[- ]checked', r'debunked', r'false.*claim',
                r'misleading.*information', r'accurate.*information', r'evidence shows',
                r'studies.*confirm', r'research.*supports', r'experts.*agree'
            ],
            'temporal_indicators': [
                r'breaking', r'just in', r'update', r'developing', r'latest', r'recent',
                r'new study', r'fresh evidence', r'updated.*findings'
            ]
        }
        
        # Domain authority mappings
        self.domain_authority = {
            '.edu': 0.85, '.gov': 0.90, '.org': 0.70, '.mil': 0.88,
            'wikipedia.org': 0.75, 'scholar.google.com': 0.80
        }

    async def multi_stage_search_pipeline(self, content: str) -> Dict[str, Any]:
        """
        Enhanced multi-stage search pipeline with claim extraction and strategic querying
        """
        pipeline_start = time.time()
        
        # Stage 1: Content Analysis and Claim Extraction
        claims_analysis = await self._extract_and_analyze_claims(content)
        
        # Stage 2: Strategic Query Generation with Multiple Approaches
        query_strategies = await self._generate_strategic_queries(content, claims_analysis)
        
        # Stage 3: Parallel Multi-Modal Search
        search_results = await self._execute_parallel_searches(query_strategies)
        
        # Stage 4: Advanced Result Processing and Scoring
        processed_results = await self._advanced_result_processing(search_results, content, claims_analysis)
        
        # Stage 5: Context-Aware Source Selection
        final_sources = await self._context_aware_source_selection(processed_results, claims_analysis)
        
        return {
            "claims_analysis": claims_analysis,
            "search_strategies": query_strategies,
            "total_results_found": len(search_results),
            "processed_results": len(processed_results),
            "final_sources": final_sources,
            "pipeline_duration": round(time.time() - pipeline_start, 2),
            "confidence_metrics": self._calculate_search_confidence(final_sources)
        }

    async def _extract_and_analyze_claims(self, content: str) -> Dict[str, Any]:
        """Extract and categorize factual claims from content"""
        claims_prompt = f"""
        Analyze this content and extract distinct factual claims that can be verified or fact-checked.
        Categorize each claim and assess its verifiability.

        Content: "{content}"

        Return a JSON object with this structure:
        {{
            "primary_claims": [
                {{
                    "claim": "specific factual assertion",
                    "category": "scientific|political|historical|statistical|medical|general",
                    "verifiability": "high|medium|low",
                    "specificity": "specific|vague|ambiguous",
                    "temporal_relevance": "current|historical|timeless"
                }}
            ],
            "supporting_claims": ["claim1", "claim2"],
            "content_type": "news|opinion|academic|social_media|advertisement|other",
            "dominant_topic": "brief topic description",
            "urgency_level": "high|medium|low",
            "controversy_potential": "high|medium|low"
        }}
        """
        
        try:
            response = await self.llm_manager.generate_response(claims_prompt, max_tokens=800, temperature=0.1)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                claims_data = json.loads(json_match.group())
                # Add content fingerprint for caching
                claims_data["content_hash"] = hashlib.md5(content.encode()).hexdigest()[:16]
                return claims_data
        except Exception as e:
            print(f"Claims extraction error: {e}")
        
        # Fallback analysis
        return self._fallback_claims_analysis(content)

    async def _generate_strategic_queries(self, content: str, claims_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate multiple types of strategic search queries"""
        strategies = {
            "verification_queries": [],
            "contradiction_queries": [],
            "source_validation_queries": [],
            "context_queries": [],
            "expert_opinion_queries": []
        }
        
        # Generate queries for each strategy
        strategy_prompts = {
            "verification_queries": f"""
            Generate 4-5 search queries to VERIFY these claims. Focus on finding authoritative sources that confirm or deny:
            
            Claims: {[claim['claim'] for claim in claims_analysis.get('primary_claims', [])]}
            Topic: {claims_analysis.get('dominant_topic', 'general')}
            
            Return only the queries, one per line:
            """,
            
            "contradiction_queries": f"""
            Generate 3-4 search queries to find CONTRADICTORY information or debunking of these claims:
            
            Claims: {[claim['claim'] for claim in claims_analysis.get('primary_claims', [])]}
            
            Use terms like: debunked, false, myth, incorrect, misleading
            Return only the queries, one per line:
            """,
            
            "source_validation_queries": f"""
            Generate 3-4 queries to validate the ORIGINAL SOURCES mentioned in this content:
            
            Content: "{content[:400]}"
            
            Focus on finding the original studies, reports, or authorities cited.
            Return only the queries, one per line:
            """,
            
            "expert_opinion_queries": f"""
            Generate 3-4 queries to find EXPERT OPINIONS on this topic:
            
            Topic: {claims_analysis.get('dominant_topic', 'general')}
            Category: {claims_analysis.get('primary_claims', [{}])[0].get('category', 'general') if claims_analysis.get('primary_claims') else 'general'}
            
            Target experts, institutions, and authoritative bodies.
            Return only the queries, one per line:
            """
        }
        
        # Execute query generation in parallel
        tasks = [
            self._generate_query_set(prompt, strategy)
            for strategy, prompt in strategy_prompts.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (strategy, result) in enumerate(zip(strategy_prompts.keys(), results)):
            if isinstance(result, list):
                strategies[strategy] = result
            else:
                strategies[strategy] = self._fallback_queries(content, strategy)
        
        return strategies

    async def _generate_query_set(self, prompt: str, strategy: str) -> List[str]:
        """Generate a set of queries for a specific strategy"""
        try:
            response = await self.llm_manager.generate_response(prompt, max_tokens=300, temperature=0.3)
            queries = [q.strip() for q in response.split('\n') if q.strip() and len(q.strip()) > 10]
            return queries[:5]  # Limit to 5 queries per strategy
        except Exception as e:
            print(f"Query generation error for {strategy}: {e}")
            return []

    async def _execute_parallel_searches(self, query_strategies: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Execute all search queries in parallel with rate limiting"""
        all_queries = []
        for strategy, queries in query_strategies.items():
            for query in queries:
                all_queries.append({"query": query, "strategy": strategy})
        
        # Rate limiting: batch searches
        batch_size = 8
        all_results = []
        
        for i in range(0, len(all_queries), batch_size):
            batch = all_queries[i:i + batch_size]
            
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self._enhanced_single_search(session, item["query"], item["strategy"])
                    for item in batch
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for results in batch_results:
                    if isinstance(results, list):
                        all_results.extend(results)
            
            # Rate limiting delay
            if i + batch_size < len(all_queries):
                await asyncio.sleep(1)
        
        return all_results

    async def _enhanced_single_search(self, session: aiohttp.ClientSession, query: str, strategy: str) -> List[Dict[str, Any]]:
        """Enhanced single search with strategy-specific parameters"""
        try:
            # Check cache first
            cache_key = hashlib.md5(f"{query}_{strategy}".encode()).hexdigest()
            if cache_key in self.search_cache:
                cached_result, timestamp = self.search_cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    return cached_result
            
            # Strategy-specific search parameters
            search_params = self._get_strategy_search_params(strategy, query)
            
            async with session.get(
                "https://www.googleapis.com/customsearch/v1",
                params=search_params,
                timeout=25
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for item in data.get('items', []):
                        result = await self._create_enhanced_result_object(item, query, strategy)
                        if result:
                            results.append(result)
                    
                    # Cache results
                    self.search_cache[cache_key] = (results, time.time())
                    return results
                    
        except Exception as e:
            print(f"Enhanced search error for '{query}' ({strategy}): {e}")
        
        return []

    def _get_strategy_search_params(self, strategy: str, query: str) -> Dict[str, Any]:
        """Get search parameters optimized for specific strategies"""
        base_params = {
            'key': GOOGLE_API_KEY,
            'cx': GOOGLE_CSE_ID,
            'q': query,
            'num': 10,
            'safe': 'medium'
        }
        
        # Strategy-specific modifications
        if strategy == "verification_queries":
            base_params.update({
                'dateRestrict': 'y3',  # Last 3 years
                'siteSearch': 'site:edu OR site:gov OR site:org'
            })
        elif strategy == "contradiction_queries":
            base_params.update({
                'dateRestrict': 'y5',
                'q': f"{query} debunked OR false OR myth"
            })
        elif strategy == "expert_opinion_queries":
            base_params.update({
                'dateRestrict': 'y2',
                'siteSearch': 'site:edu OR site:gov'
            })
        elif strategy == "source_validation_queries":
            base_params.update({
                'dateRestrict': 'y10'
            })
        
        return base_params

    async def _create_enhanced_result_object(self, item: Dict, query: str, strategy: str) -> Optional[Dict[str, Any]]:
        """Create an enhanced result object with comprehensive metadata"""
        try:
            url = item.get('link', '')
            domain = urlparse(url).netloc.lower()
            
            # Skip low-quality domains
            if any(skip in domain for skip in ['pinterest.com', 'quora.com', 'yahoo.com/answers']):
                return None
            
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            content_text = f"{title} {snippet}".lower()
            
            result = {
                "title": title,
                "link": url,
                "snippet": snippet,
                "domain": domain,
                "query_used": query,
                "search_strategy": strategy,
                "timestamp": datetime.now().isoformat(),
                
                # Enhanced credibility analysis
                "credibility_score": self._calculate_enhanced_credibility(url, content_text),
                "source_metadata": self._analyze_source_metadata(item, domain),
                
                # Content analysis
                "content_indicators": self._comprehensive_content_analysis(content_text),
                
                # Relevance scoring
                "relevance_metrics": self._calculate_relevance_metrics(content_text, query, strategy),
                
                # Temporal analysis
                "temporal_indicators": self._extract_temporal_indicators(content_text, item),
                
                # Bias and perspective analysis
                "perspective_analysis": self._analyze_perspective_bias(content_text, domain)
            }
            
            return result
            
        except Exception as e:
            print(f"Result object creation error: {e}")
            return None

    def _analyze_source_metadata(self, item: Dict, domain: str) -> Dict[str, Any]:
        """Analyze source metadata from search result item"""
        return {
        "page_rank": item.get('pagemap', {}).get('metatags', [{}])[0].get('page_rank', 'unknown'),
        "author": item.get('pagemap', {}).get('metatags', [{}])[0].get('author', 'unknown'),
        "publication_date": item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time', 'unknown'),
        "domain_age": self._estimate_domain_age(domain),
        "ssl_enabled": item.get('link', '').startswith('https'),
        "meta_description": item.get('pagemap', {}).get('metatags', [{}])[0].get('description', ''),
        "content_type": item.get('pagemap', {}).get('metatags', [{}])[0].get('og:type', 'article')
    }

    def _comprehensive_content_analysis(self, content_text: str) -> Dict[str, Any]:
        """Comprehensive analysis of content quality indicators"""
        indicators = {
            "high_credibility_matches": 0,
            "misinformation_matches": 0,
            "verification_matches": 0,
            "temporal_matches": 0,
            "emotional_language_score": 0,
            "technical_depth_score": 0
        }
    
    # Count pattern matches
        for pattern in self.content_patterns['high_credibility_indicators']:
            indicators["high_credibility_matches"] += len(re.findall(pattern, content_text, re.IGNORECASE))
    
        for pattern in self.content_patterns['misinformation_indicators']:
            indicators["misinformation_matches"] += len(re.findall(pattern, content_text, re.IGNORECASE))
    
        for pattern in self.content_patterns['verification_indicators']:
            indicators["verification_matches"] += len(re.findall(pattern, content_text, re.IGNORECASE))
    
        for pattern in self.content_patterns['temporal_indicators']:
            indicators["temporal_matches"] += len(re.findall(pattern, content_text, re.IGNORECASE))
    
    # Emotional language analysis
        emotional_words = ['shocking', 'amazing', 'incredible', 'unbelievable', 'devastating', 'outrageous']
        indicators["emotional_language_score"] = sum(1 for word in emotional_words if word in content_text.lower())
    
    # Technical depth analysis
        technical_words = ['study', 'research', 'analysis', 'data', 'methodology', 'peer-reviewed', 'systematic']
        indicators["technical_depth_score"] = sum(1 for word in technical_words if word in content_text.lower())                
    
        return indicators

    def _calculate_relevance_metrics(self, content_text: str, query: str, strategy: str) -> Dict[str, float]:
        """Calculate relevance metrics for search result"""
        query_words = set(query.lower().split())
        content_words = set(content_text.lower().split())
        
        # Basic overlap
        word_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
    
    # Strategy-specific relevance
        strategy_bonus = 0
        if strategy == "verification_queries" and any(word in content_text.lower() for word in ['confirmed', 'verified', 'proven']):
            strategy_bonus = 0.2
        elif strategy == "contradiction_queries" and any(word in content_text.lower() for word in ['false', 'debunked', 'myth']):
            strategy_bonus = 0.2
        elif strategy == "expert_opinion_queries" and any(word in content_text.lower() for word in ['expert', 'professor', 'researcher']):
            strategy_bonus = 0.2
    
    # Query term frequency
        query_frequency = sum(content_text.lower().count(word) for word in query_words) / len(content_text.split()) if content_text else 0
    
        total_score = (word_overlap * 0.5) + (strategy_bonus * 0.3) + (query_frequency * 0.2)
    
        return {
        "word_overlap": round(word_overlap, 3),
        "strategy_bonus": round(strategy_bonus, 3),
        "query_frequency": round(query_frequency, 3),
        "total_score": round(total_score, 3)
    }

    def _extract_temporal_indicators(self, content_text: str, item: Dict) -> Dict[str, Any]:
        """Extract temporal relevance indicators"""
        temporal_info = {
            "has_recent_indicators": False,
            "has_breaking_indicators": False,
            "estimated_recency": "unknown",
            "temporal_keywords": []
        }

        # Check for temporal patterns
        for pattern in self.content_patterns['temporal_indicators']:
            matches = re.findall(pattern, content_text, re.IGNORECASE)
            if matches:
                temporal_info["temporal_keywords"].extend(matches)
    
    # Specific checks
        temporal_info["has_recent_indicators"] = any(word in content_text.lower() for word in ['recent', 'new', 'latest', 'updated'])
        temporal_info["has_breaking_indicators"] = any(word in content_text.lower() for word in ['breaking', 'just in', 'developing'])
    
    # Extract dates from content
        date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
        r'\b\d{4}-\d{1,2}-\d{1,2}\b',  # YYYY-MM-DD
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
    ]
    
        dates_found = []
        for pattern in date_patterns:
            dates_found.extend(re.findall(pattern, content_text, re.IGNORECASE))
    
        temporal_info["dates_mentioned"] = dates_found[:3]  # Limit to first 3 dates
    
        return temporal_info

    def _analyze_perspective_bias(self, content_text: str, domain: str) -> Dict[str, Any]:
        """Analyze potential bias and perspective indicators"""
        bias_analysis = {
            "domain_bias": "unknown",
            "language_bias_indicators": [],
            "perspective_type": "neutral",
            "certainty_level": "moderate"
        }

        # Domain-based bias (from credibility matrix)
        for source_domain, metadata in self.credibility_matrix.items():
            if source_domain in domain:
                bias_analysis["domain_bias"] = metadata.get("bias", "unknown")
                break
            
        # Language bias indicators
        left_bias_words = ['progressive', 'social justice', 'inequality', 'systemic', 'marginalized']
        right_bias_words = ['traditional', 'conservative', 'freedom', 'liberty', 'free market']
        neutral_words = ['analysis', 'research', 'study', 'data', 'evidence']

        left_count = sum(1 for word in left_bias_words if word in content_text.lower())
        right_count = sum(1 for word in right_bias_words if word in content_text.lower())
        neutral_count = sum(1 for word in neutral_words if word in content_text.lower())

        if neutral_count > max(left_count, right_count):
            bias_analysis["perspective_type"] = "neutral"
        elif left_count > right_count:
            bias_analysis["perspective_type"] = "left_leaning"
        elif right_count > left_count:
            bias_analysis["perspective_type"] = "right_leaning"

        # Certainty level analysis
        certain_words = ['definitely', 'certainly', 'absolutely', 'without doubt', 'proven']
        uncertain_words = ['might', 'could', 'possibly', 'suggests', 'indicates']

        certain_count = sum(1 for word in certain_words if word in content_text.lower())
        uncertain_count = sum(1 for word in uncertain_words if word in content_text.lower())

        if certain_count > uncertain_count:
            bias_analysis["certainty_level"] = "high"
        elif uncertain_count > certain_count:
            bias_analysis["certainty_level"] = "low"

        return bias_analysis

    def _estimate_domain_age(self, domain: str) -> str:
        """Estimate domain age category"""
        well_established = ['bbc.com', 'reuters.com', 'apnews.com', 'nytimes.com', 'washingtonpost.com']
        if any(est in domain for est in well_established):
            return "well_established"
        elif '.edu' in domain or '.gov' in domain:
            return "institutional"
        else:
            return "unknown"

    def _fallback_queries(self, content: str, strategy: str) -> List[str]:
        """Generate fallback queries when LLM query generation fails"""
        content_words = content.split()[:10]  # First 10 words
        base_query = " ".join(content_words)

        if strategy == "verification_queries":
            return [f"{base_query} verified", f"{base_query} fact check", f"{base_query} confirmed"]
        elif strategy == "contradiction_queries":
            return [f"{base_query} false", f"{base_query} debunked", f"{base_query} myth"]
        elif strategy == "expert_opinion_queries":
            return [f"{base_query} expert opinion", f"{base_query} researcher", f"{base_query} scientist"]
        elif strategy == "source_validation_queries":
            return [f"{base_query} original source", f"{base_query} study", f"{base_query} research"]
        else:
            return [base_query, f"{base_query} information", f"{base_query} facts"]

    def _cluster_similar_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster similar results to avoid redundancy"""
        if len(results) <= 5:
            return results

        clusters = []
        used_indices = set()

        for i, result in enumerate(results):
            if i in used_indices:
                continue

            cluster = [result]
            result_words = set(result.get('title', '').lower().split())

            for j, other_result in enumerate(results[i+1:], i+1):
                if j in used_indices:
                    continue

                other_words = set(other_result.get('title', '').lower().split())
                similarity = len(result_words.intersection(other_words)) / max(len(result_words), len(other_words), 1)

                if similarity > 0.6 and result.get('domain') != other_result.get('domain'):
                    cluster.append(other_result)
                    used_indices.add(j)

            # Keep the best result from each cluster
            best_result = max(cluster, key=lambda x: x.get('credibility_score', {}).get('overall_score', 0))
            clusters.append(best_result)
            used_indices.add(i)

        return clusters

    async def _calculate_advanced_relevance_scores(self, results: List[Dict[str, Any]], 
                                                 content: str, claims_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate advanced relevance scores considering claims and context"""
        claims = [claim.get('claim', '') for claim in claims_analysis.get('primary_claims', [])]
        content_type = claims_analysis.get('content_type', 'general')

        for result in results:
            base_relevance = result.get('relevance_metrics', {}).get('total_score', 0)

            # Claim-specific relevance
            claim_relevance = 0
            result_text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()

            for claim in claims:
                claim_words = set(claim.lower().split())
                result_words = set(result_text.split())
                overlap = len(claim_words.intersection(result_words)) / max(len(claim_words), 1)
                claim_relevance = max(claim_relevance, overlap)

            # Content type bonus
            type_bonus = 0
            if content_type == 'medical' and any(word in result_text for word in ['medical', 'health', 'doctor', 'study']):
                type_bonus = 0.1
            elif content_type == 'scientific' and any(word in result_text for word in ['research', 'study', 'peer-reviewed']):
                type_bonus = 0.1

            # Update relevance metrics
            result['relevance_metrics']['claim_relevance'] = round(claim_relevance, 3)
            result['relevance_metrics']['type_bonus'] = round(type_bonus, 3)
            result['relevance_metrics']['advanced_score'] = round(base_relevance + claim_relevance * 0.3 + type_bonus, 3)

        return results

    def _optimize_result_diversity(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize result diversity to avoid echo chambers"""
        if len(results) <= 8:
            return results

        diversified = []
        used_domains = set()
        used_bias_types = set()

        # Sort by advanced relevance score
        sorted_results = sorted(results, key=lambda x: x.get('relevance_metrics', {}).get('advanced_score', 0), reverse=True)

        for result in sorted_results:
            domain = result.get('domain', '')
            bias_type = result.get('perspective_analysis', {}).get('domain_bias', 'unknown')

            # Always include top results
            if len(diversified) < 3:
                diversified.append(result)
                used_domains.add(domain)
                used_bias_types.add(bias_type)
                continue
            
            # Prioritize diversity
            domain_diversity_bonus = 0 if domain in used_domains else 0.2
            bias_diversity_bonus = 0 if bias_type in used_bias_types else 0.1

            diversity_score = (result.get('relevance_metrics', {}).get('advanced_score', 0) + 
                             domain_diversity_bonus + bias_diversity_bonus)

            result['diversity_score'] = round(diversity_score, 3)
            diversified.append(result)
            used_domains.add(domain)
            used_bias_types.add(bias_type)

            if len(diversified) >= 12:
                break
            
        return diversified
    def _calculate_enhanced_credibility(self, url: str, content_text: str) -> Dict[str, float]:
        """Calculate multi-dimensional credibility score"""
        domain = urlparse(url).netloc.lower()
        
        # Base credibility from domain
        base_score = 0.4
        source_type = "other"
        bias_rating = "unknown"
        
        # Check credibility matrix
        for source_domain, metadata in self.credibility_matrix.items():
            if source_domain in domain:
                base_score = metadata["weight"]
                source_type = metadata["type"]
                bias_rating = metadata["bias"]
                break
        
        # Check domain authority patterns
        for authority_pattern, score in self.domain_authority.items():
            if authority_pattern in domain:
                base_score = max(base_score, score)
                break
        
        # Content-based credibility adjustments
        content_score = self._analyze_content_credibility(content_text)
        
        # Combine scores
        final_score = (base_score * 0.7) + (content_score * 0.3)
        
        return {
            "overall_score": round(final_score, 3),
            "domain_score": round(base_score, 3),
            "content_score": round(content_score, 3),
            "source_type": source_type,
            "bias_rating": bias_rating
        }

    def _analyze_content_credibility(self, content_text: str) -> float:
        """Analyze content for credibility indicators"""
        score = 0.5  # Neutral starting point
        
        # Positive indicators
        for pattern in self.content_patterns['high_credibility_indicators']:
            if re.search(pattern, content_text, re.IGNORECASE):
                score += 0.05
        
        # Negative indicators
        for pattern in self.content_patterns['misinformation_indicators']:
            if re.search(pattern, content_text, re.IGNORECASE):
                score -= 0.1
        
        # Verification indicators
        for pattern in self.content_patterns['verification_indicators']:
            if re.search(pattern, content_text, re.IGNORECASE):
                score += 0.03
        
        return max(0.0, min(1.0, score))

    async def _advanced_result_processing(self, search_results: List[Dict[str, Any]], 
                                        content: str, claims_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Advanced processing with deduplication, clustering, and enhanced scoring"""
        if not search_results:
            return []
        
        # Stage 1: Deduplication with semantic similarity
        deduplicated = self._semantic_deduplication(search_results)
        
        # Stage 2: Cluster similar results
        clustered_results = self._cluster_similar_results(deduplicated)
        
        # Stage 3: Enhanced relevance scoring
        scored_results = await self._calculate_advanced_relevance_scores(
            clustered_results, content, claims_analysis
        )
        
        # Stage 4: Diversity optimization
        diversified_results = self._optimize_result_diversity(scored_results)
        
        return diversified_results

    def _semantic_deduplication(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove semantically similar results"""
        seen_domains = set()
        seen_titles = set()
        deduplicated = []
        
        # Sort by credibility first
        results_sorted = sorted(results, key=lambda x: x.get('credibility_score', {}).get('overall_score', 0), reverse=True)
        
        for result in results_sorted:
            domain = result.get('domain', '')
            title = result.get('title', '').lower()
            
            # Skip if we already have a result from this domain (unless it's a high-authority domain)
            if domain in seen_domains and result.get('credibility_score', {}).get('overall_score', 0) < 0.8:
                continue
            
            # Skip if we have a very similar title
            title_words = set(title.split())
            is_similar = any(
                len(title_words.intersection(set(seen_title.split()))) / max(len(title_words), len(seen_title.split())) > 0.8
                for seen_title in seen_titles
            )
            
            if not is_similar:
                deduplicated.append(result)
                seen_domains.add(domain)
                seen_titles.add(title)
        
        return deduplicated

    async def _context_aware_source_selection(self, processed_results: List[Dict[str, Any]], 
                                            claims_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select sources using advanced context awareness"""
        if not processed_results:
            return []
        
        selection_prompt = f"""
        You are a professional fact-checker selecting the most valuable sources for verification.
        
        CLAIMS TO VERIFY:
        {json.dumps([claim['claim'] for claim in claims_analysis.get('primary_claims', [])], indent=2)}
        
        CONTENT TYPE: {claims_analysis.get('content_type', 'unknown')}
        CONTROVERSY LEVEL: {claims_analysis.get('controversy_potential', 'unknown')}
        
        AVAILABLE SOURCES (showing top candidates):
        {self._format_sources_for_selection(processed_results[:20])}
        
        Select and rank the most valuable sources considering:
        1. Direct relevance to specific claims
        2. Source credibility and authority
        3. Complementary perspectives (pro/con evidence)
        4. Recency and temporal relevance
        5. Source diversity (avoid echo chambers)
        
        Return a JSON array of source indices (0-based) in order of value for fact-checking:
        {{"selected_indices": [1, 5, 12, 8, ...]}}
        
        Select 8-12 sources maximum.
        """
        
        try:
            response = await self.llm_manager.generate_response(selection_prompt, max_tokens=400, temperature=0.1)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                selection_data = json.loads(json_match.group())
                indices = selection_data.get('selected_indices', [])
                
                selected_sources = []
                for idx in indices:
                    if 0 <= idx < len(processed_results):
                        source = processed_results[idx].copy()
                        source['selection_rank'] = len(selected_sources) + 1
                        selected_sources.append(source)
                
                return selected_sources[:12]
                
        except Exception as e:
            print(f"Context-aware selection error: {e}")
        
        # Fallback to score-based selection
        return sorted(processed_results, 
                     key=lambda x: x.get('relevance_metrics', {}).get('total_score', 0), 
                     reverse=True)[:10]

    def _format_sources_for_selection(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for LLM selection prompt"""
        formatted = []
        for i, source in enumerate(sources):
            cred = source.get('credibility_score', {})
            rel = source.get('relevance_metrics', {})
            
            formatted.append(f"""
            [{i}] {source.get('title', 'Untitled')}
                Domain: {source.get('domain', 'unknown')} | Strategy: {source.get('search_strategy', 'unknown')}
                Credibility: {cred.get('overall_score', 0):.2f} | Relevance: {rel.get('total_score', 0):.2f}
                Type: {cred.get('source_type', 'unknown')} | Bias: {cred.get('bias_rating', 'unknown')}
                Snippet: {source.get('snippet', 'No snippet')[:100]}...
            """.strip())
        
        return '\n'.join(formatted)

    # Additional helper methods...
    def _fallback_claims_analysis(self, content: str) -> Dict[str, Any]:
        """Fallback claims analysis when LLM fails"""
        return {
            "primary_claims": [{"claim": content[:100], "category": "general", "verifiability": "medium"}],
            "content_type": "unknown",
            "dominant_topic": "general",
            "urgency_level": "medium",
            "controversy_potential": "medium",
            "content_hash": hashlib.md5(content.encode()).hexdigest()[:16]
        }

    def _calculate_search_confidence(self, sources: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence metrics for the search results"""
        if not sources:
            return {"overall_confidence": 0.0}
        
        avg_credibility = sum(s.get('credibility_score', {}).get('overall_score', 0) for s in sources) / len(sources)
        source_diversity = len(set(s.get('domain', '') for s in sources)) / len(sources)
        fact_check_ratio = len([s for s in sources if s.get('credibility_score', {}).get('source_type') == 'fact_check']) / len(sources)
        
        overall_confidence = (avg_credibility * 0.5) + (source_diversity * 0.3) + (fact_check_ratio * 0.2)
        
        return {
            "overall_confidence": round(overall_confidence, 3),
            "average_credibility": round(avg_credibility, 3),
            "source_diversity": round(source_diversity, 3),
            "fact_check_ratio": round(fact_check_ratio, 3)
        }


class AdvancedFactChecker:
    def __init__(self):
        self.llm_manager = LLMManager()
        self.image_processor = AdvancedImageProcessor()
        self.search_engine = EnhancedSearchEngine(self.llm_manager)
        
        # Enhanced analysis frameworks
        self.verification_frameworks = {
            "scientific": {
                "criteria": ["peer_review", "replication", "methodology", "sample_size", "statistical_significance"],
                "weight_distribution": [0.25, 0.20, 0.20, 0.15, 0.20]
            },
            "news": {
                "criteria": ["source_diversity", "primary_sources", "expert_quotes", "contradictory_evidence", "temporal_relevance"],
                "weight_distribution": [0.20, 0.25, 0.20, 0.20, 0.15]
            },
            "statistical": {
                "criteria": ["data_source", "methodology", "sample_representativeness", "statistical_methods", "confidence_intervals"],
                "weight_distribution": [0.25, 0.20, 0.20, 0.20, 0.15]
            },
            "historical": {
                "criteria": ["primary_sources", "scholarly_consensus", "archaeological_evidence", "document_authenticity", "cross_references"],
                "weight_distribution": [0.30, 0.25, 0.15, 0.15, 0.15]
            }
        }
        
        # Multi-dimensional verdict system
        self.verdict_criteria = {
            "factual_accuracy": {"weight": 0.35, "description": "Alignment with established facts"},
            "source_reliability": {"weight": 0.25, "description": "Quality and credibility of sources"},
            "evidence_strength": {"weight": 0.20, "description": "Strength and consistency of evidence"},
            "expert_consensus": {"weight": 0.15, "description": "Agreement among domain experts"},
            "methodological_rigor": {"weight": 0.05, "description": "Quality of research methodology"}
        }

    async def comprehensive_fact_check_pipeline(self, content: str, image: Optional[Any] = None) -> Dict[str, Any]:
        """
        Professional multi-stage fact-checking pipeline with advanced analysis
        """
        pipeline_start = time.time()
        
        # Initialize comprehensive analysis result
        analysis_result = {
            "analysis_id": hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:16],
            "timestamp": datetime.now().isoformat(),
            "content_metadata": await self._analyze_content_metadata(content),
            "image_analysis": None,
            "pipeline_stages": {},
            "confidence_tracking": {}
        }
        
        try:
            # Stage 1: Content Preprocessing and Enhancement
            stage_start = time.time()
            enhanced_content = await self._preprocess_and_enhance_content(content, image)
            analysis_result["pipeline_stages"]["content_preprocessing"] = {
                "duration": round(time.time() - stage_start, 2),
                "enhanced_content_length": len(enhanced_content["final_content"]),
                "image_processed": enhanced_content["image_processed"]
            } 
             
            print("\n\n")
            print("content preprocessing -> ")
            print(analysis_result)
            
            if enhanced_content["image_analysis"]:
                analysis_result["image_analysis"] = enhanced_content["image_analysis"]
            
            # Stage 2: Advanced Search Pipeline
            stage_start = time.time()
            search_pipeline_result = await self.search_engine.multi_stage_search_pipeline(
                enhanced_content["final_content"]
            )
            analysis_result["pipeline_stages"]["search_pipeline"] = {
                "duration": round(time.time() - stage_start, 2),
                **search_pipeline_result["confidence_metrics"]
            }
            
            print("\n\n")
            print("stage 2 -> ")
            print(analysis_result["pipeline_stages"]["search_pipeline"])
            
            stage_start = time.time()
            evidence_analysis = await self._multi_perspective_evidence_analysis(
        enhanced_content["final_content"],
        search_pipeline_result["final_sources"],
        search_pipeline_result["claims_analysis"]
)

            # Debug: Print the actual structure
            print(f"Evidence analysis structure: {evidence_analysis.keys() if evidence_analysis else 'None'}")

            # Fix: Safely access all keys with proper fallbacks
            perspective_analyses = evidence_analysis.get("perspective_analyses", {})
            perspectives_count = 0
            if isinstance(perspective_analyses, dict):
                perspectives_count = len([k for k in perspective_analyses.keys() if isinstance(perspective_analyses[k], dict)])

            analysis_result["pipeline_stages"]["evidence_analysis"] = {
                "duration": round(time.time() - stage_start, 2),
                "perspectives_analyzed": perspectives_count,
                "evidence_strength": evidence_analysis.get("overall_evidence_strength", "unknown")
            }
            
            print("\n\n")
            print("stage 3 -> ")
            print(analysis_result["pipeline_stages"]["evidence_analysis"])
            
            # Stage 4: Advanced Credibility Assessment
            stage_start = time.time()
            credibility_assessment = await self._advanced_credibility_assessment(
               enhanced_content["final_content"],
               search_pipeline_result["final_sources"],
               evidence_analysis
           )
            analysis_result["pipeline_stages"]["credibility_assessment"] = {
               "duration": round(time.time() - stage_start, 2),
               "credibility_dimensions": len(credibility_assessment["dimensional_scores"]),
               "overall_credibility": credibility_assessment["overall_credibility_score"]
           }
            
            print("\n\n")
            print("stage 4 -> ")
            print(analysis_result["pipeline_stages"]["credibility_assessment"])
            
           # Stage 5: Expert Domain Analysis
            stage_start = time.time()
            domain_analysis = await self._expert_domain_analysis(
               enhanced_content["final_content"],
               search_pipeline_result["claims_analysis"],
               evidence_analysis
           )
            analysis_result["pipeline_stages"]["domain_analysis"] = {
               "duration": round(time.time() - stage_start, 2),
               "domains_analyzed": len(domain_analysis["domain_assessments"]),
               "expert_consensus_level": domain_analysis["consensus_metrics"]["overall_consensus"]
           }
            
            print("\n\n")
            print("stage 5 -> ")
            print(analysis_result["pipeline_stages"]["domain_analysis"])
            
           # Stage 6: Comprehensive Verdict Generation
            stage_start = time.time()
            final_verdict = await self._generate_comprehensive_verdict(
               enhanced_content["final_content"],
               search_pipeline_result,
               evidence_analysis,
               credibility_assessment,
               domain_analysis
           )
            analysis_result["pipeline_stages"]["verdict_generation"] = {
               "duration": round(time.time() - stage_start, 2),
               "verdict_confidence": final_verdict["confidence_score"],
               "evidence_items_analyzed": final_verdict["evidence_summary"]["total_evidence_items"]
           }
            
            print("\n\n")
            print("stage 6 -> ")
            print(analysis_result["pipeline_stages"]["verdict_generation"])
            
           # Compile final comprehensive result
            analysis_result.update({
               "search_results": search_pipeline_result,
               "evidence_analysis": evidence_analysis,
               "credibility_assessment": credibility_assessment,
               "domain_analysis": domain_analysis,
               "final_verdict": final_verdict,
               "total_pipeline_duration": round(time.time() - pipeline_start, 2),
               "analysis_completeness": self._calculate_analysis_completeness(analysis_result)
           })
            
        except Exception as e:
            analysis_result["error"] = {
               "type": type(e).__name__,
               "message": str(e),
               "stage": "pipeline_execution"
           }
           
        print("\n\ndone\n\n") 
        return analysis_result

    async def _analyze_content_metadata(self, content: str) -> Dict[str, Any]:
       """Analyze content metadata and characteristics"""
       word_count = len(content.split())
       char_count = len(content)
       
       # Detect content type patterns
       content_type_indicators = {
           "news_article": ["breaking", "reported", "according to", "sources say"],
           "social_media": ["#", "@", "RT", "share", "like"],
           "academic": ["study", "research", "journal", "peer-reviewed"],
           "opinion": ["i think", "believe", "opinion", "my view"],
           "advertisement": ["buy", "sale", "discount", "offer", "click here"]
       }
       
       detected_types = []
       for content_type, indicators in content_type_indicators.items():
           if any(indicator.lower() in content.lower() for indicator in indicators):
               detected_types.append(content_type)
       
       return {
           "word_count": word_count,
           "character_count": char_count,
           "estimated_reading_time": round(word_count / 200, 1),  # Average reading speed
           "detected_content_types": detected_types,
           "complexity_score": self._calculate_content_complexity(content),
           "urgency_indicators": self._detect_urgency_indicators(content),
           "emotional_indicators": self._detect_emotional_language(content)
       }

    def _calculate_content_complexity(self, content: str) -> float:
       """Calculate content complexity score"""
       words = content.split()
       if not words:
           return 0.0
       
       # Average word length
       avg_word_length = sum(len(word) for word in words) / len(words)
       
       # Sentence complexity (average words per sentence)
       sentences = content.split('.')
       avg_sentence_length = len(words) / max(len(sentences), 1)
       
       # Technical terms (words > 8 characters)
       technical_terms = len([w for w in words if len(w) > 8]) / len(words)
       
       complexity = (avg_word_length / 10) + (avg_sentence_length / 30) + technical_terms
       return min(1.0, complexity)

    def _detect_urgency_indicators(self, content: str) -> List[str]:
       """Detect urgency indicators in content"""
       urgency_patterns = [
           "breaking", "urgent", "immediate", "now", "today", "just in",
           "developing", "alert", "warning", "critical", "emergency"
       ]
       return [pattern for pattern in urgency_patterns if pattern.lower() in content.lower()]

    def _detect_emotional_language(self, content: str) -> Dict[str, int]:
       """Detect emotional language patterns"""
       emotion_patterns = {
           "anger": ["outraged", "furious", "angry", "mad", "disgusted"],
           "fear": ["terrifying", "scary", "frightening", "alarming", "worried"],
           "excitement": ["amazing", "incredible", "fantastic", "wonderful", "exciting"],
           "sadness": ["tragic", "sad", "heartbreaking", "devastating", "terrible"]
       }
       
       detected_emotions = {}
       for emotion, patterns in emotion_patterns.items():
           count = sum(1 for pattern in patterns if pattern.lower() in content.lower())
           if count > 0:
               detected_emotions[emotion] = count
       
       return detected_emotions

    async def _preprocess_and_enhance_content(self, content: str, image: Optional[Any] = None) -> Dict[str, Any]:
        enhanced_result = {
            "original_content": content,
            "final_content": content,
            "image_processed": False,
            "image_analysis": None,
            "content_enhancements": [],
            "processing_metadata": {
                "timestamp": datetime.now().isoformat(),
                "processing_time": 0.0,
                "success": True,
                "errors": []
            }
        }
        start_time = time.time()
        if image:
            try:
                if not isinstance(image, Image.Image):
                    if hasattr(image, 'read'):
                        image = Image.open(image)
                    elif isinstance(image, (str, Path)):
                        image = Image.open(image)
                    elif isinstance(image, np.ndarray):
                        image = Image.fromarray(image)
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                text_extraction_result = self.image_processor.extract_text_comprehensive(image)
                image_metadata = self.image_processor.analyze_image_metadata(image)
                image_analysis = {
                    "text_extraction": text_extraction_result,
                    "metadata": image_metadata,
                    "processing_details": {
                        "preprocessing_variants": len(self.image_processor.preprocess_image(image)),
                        "ocr_engines_used": list(self.image_processor.ocr_engines.keys()),
                        "best_extraction_method": text_extraction_result.get("method", "none"),
                        "confidence_score": text_extraction_result.get("confidence", 0.0)
                    }
                }
                enhanced_result["image_analysis"] = image_analysis
                enhanced_result["image_processed"] = True
                extracted_text = text_extraction_result.get("text", "").strip()
                confidence = text_extraction_result.get("confidence", 0.0)
                if extracted_text and len(extracted_text) > 5:
                    if confidence > 50:
                        enhanced_content = f"{content}\n\n--- Extracted Text (High Confidence) ---\n{extracted_text}"
                        enhanced_result["content_enhancements"].append("high_confidence_text_extraction")
                    elif confidence > 20:
                        enhanced_content = f"{content}\n\n--- Extracted Text (Medium Confidence) ---\n{extracted_text}"
                        enhanced_result["content_enhancements"].append("medium_confidence_text_extraction")
                    else:
                        all_extractions = text_extraction_result.get("all_extractions", {})
                        if all_extractions:
                            extraction_summary = "\n".join([
                                f"{method}: {text[:100]}..." 
                                for method, text in all_extractions.items() 
                                if text.strip()
                            ])
                            enhanced_content = f"{content}\n\n--- Multiple Text Extraction Attempts ---\n{extraction_summary}"
                            enhanced_result["content_enhancements"].append("multiple_extraction_attempts")
                        else:
                            enhanced_content = content
                    enhanced_result["final_content"] = enhanced_content
                quality_score = image_metadata.get("quality_score", 0)
                color_analysis = image_metadata.get("color_analysis", {})
                technical_context = []
                if quality_score < 30:
                    technical_context.append(f"Image quality: Low ({quality_score:.1f}/100) - may affect text extraction accuracy")
                elif quality_score > 70:
                    technical_context.append(f"Image quality: High ({quality_score:.1f}/100) - optimal for text extraction")
                brightness = color_analysis.get("brightness", 128)
                contrast = color_analysis.get("contrast", 0)
                if brightness < 80:
                    technical_context.append("Image appears dark - preprocessing applied for better OCR")
                elif brightness > 200:
                    technical_context.append("Image appears bright - preprocessing applied for better OCR")
                if contrast < 30:
                    technical_context.append("Low contrast detected - enhancement algorithms applied")
                if technical_context:
                    enhanced_result["final_content"] += f"\n\n--- Image Processing Notes ---\n" + "\n".join(technical_context)
                    enhanced_result["content_enhancements"].append("technical_context_analysis")
                dominant_colors = color_analysis.get("dominant_colors", [])
                if dominant_colors:
                    color_desc = f"Dominant colors detected: {len(dominant_colors)} color clusters"
                    enhanced_result["final_content"] += f"\n\n--- Visual Analysis ---\n{color_desc}"
                    enhanced_result["content_enhancements"].append("color_analysis")
            except Exception as e:
                error_msg = f"Image processing error: {str(e)}"
                enhanced_result["processing_metadata"]["errors"].append(error_msg)
                enhanced_result["processing_metadata"]["success"] = False
                try:
                    if isinstance(image, Image.Image):
                        basic_info = f"Image detected: {image.size[0]}x{image.size[1]} pixels, {image.mode} mode"
                        enhanced_result["final_content"] += f"\n\n--- Basic Image Info ---\n{basic_info}"
                        enhanced_result["content_enhancements"].append("basic_image_info")
                except:
                    pass
        else:
            content_stats = {
                "length": len(content),
                "words": len(content.split()),
                "lines": len(content.splitlines()),
                "has_urls": bool(re.search(r'https?://', content)),
                "has_emails": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content))
            }
            enhanced_result["content_analysis"] = content_stats
            enhanced_result["content_enhancements"].append("content_statistical_analysis")
        enhanced_result["processing_metadata"]["processing_time"] = time.time() - start_time
        return enhanced_result

    async def _multi_perspective_evidence_analysis(self, content: str, sources: List[Dict[str, Any]], 
                                                claims_analysis: Dict[str, Any]) -> Dict[str, Any]:
       """Analyze evidence from multiple perspectives"""
       evidence_prompt = f"""
       As a professional fact-checker, analyze the evidence for these claims from multiple perspectives:

       CLAIMS TO ANALYZE:
       {json.dumps([claim['claim'] for claim in claims_analysis.get('primary_claims', [])], indent=2)}

       AVAILABLE EVIDENCE SOURCES:
       {self._format_sources_for_analysis(sources[:15])}

       Provide a comprehensive analysis considering:
       1. SUPPORTING EVIDENCE - What evidence supports each claim?
       2. CONTRADICTING EVIDENCE - What evidence contradicts each claim?
       3. EXPERT CONSENSUS - What do experts in relevant fields say?
       4. METHODOLOGICAL ANALYSIS - How strong are the research methods?
       5. SOURCE RELIABILITY - How reliable are the sources?

       Return a JSON object with this structure:
       {{
           "perspective_analyses": {{
               "supporting_evidence": {{
                   "strength": "strong|moderate|weak",
                   "key_sources": ["source1", "source2"],
                   "evidence_quality": "high|medium|low",
                   "summary": "brief summary"
               }},
               "contradicting_evidence": {{
                   "strength": "strong|moderate|weak",
                   "key_sources": ["source1", "source2"],
                   "evidence_quality": "high|medium|low",
                   "summary": "brief summary"
               }},
               "expert_consensus": {{
                   "consensus_level": "strong|moderate|weak|none",
                   "expert_sources": ["source1", "source2"],
                   "disagreement_areas": ["area1", "area2"]
               }},
               "methodological_assessment": {{
                   "overall_quality": "high|medium|low",
                   "strengths": ["strength1", "strength2"],
                   "limitations": ["limitation1", "limitation2"]
               }}
           }},
           "overall_evidence_strength": "strong|moderate|weak",
           "confidence_level": 0.85,
           "evidence_gaps": ["gap1", "gap2"]
       }}
       """
       
       try:
           response = await self.llm_manager.generate_response(evidence_prompt, max_tokens=1200, temperature=0.1)
           json_match = re.search(r'\{.*\}', response, re.DOTALL)
           
           if json_match:
               evidence_analysis = json.loads(json_match.group())
               evidence_analysis["analysis_timestamp"] = datetime.now().isoformat()
               evidence_analysis["sources_analyzed"] = len(sources)
               return evidence_analysis
               
       except Exception as e:
           print(f"Evidence analysis error: {e}")
       
       # Fallback analysis
       return self._fallback_evidence_analysis(sources)

    def _format_sources_for_analysis(self, sources: List[Dict[str, Any]]) -> str:
       """Format sources for evidence analysis"""
       formatted = []
       for i, source in enumerate(sources):
           cred = source.get('credibility_score', {})
           formatted.append(f"""
           Source {i+1}: {source.get('title', 'Untitled')}
           Domain: {source.get('domain', 'unknown')}
           Credibility: {cred.get('overall_score', 0):.2f} ({cred.get('source_type', 'unknown')})
           Strategy: {source.get('search_strategy', 'unknown')}
           Content: {source.get('snippet', 'No content')[:200]}...
           """.strip())
       
       return '\n\n'.join(formatted)

    def _fallback_evidence_analysis(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
       """Fallback evidence analysis when LLM fails"""
       avg_credibility = sum(s.get('credibility_score', {}).get('overall_score', 0) for s in sources) / max(len(sources), 1)
       
       return {
           "perspective_analyses": {
               "supporting_evidence": {
                   "strength": "moderate",
                   "key_sources": [s.get('domain', 'unknown') for s in sources[:3]],
                   "evidence_quality": "medium",
                   "summary": "Analysis based on available sources"
               },
               "contradicting_evidence": {
                   "strength": "weak",
                   "key_sources": [],
                   "evidence_quality": "low",
                   "summary": "Limited contradicting evidence found"
               },
               "expert_consensus": {
                   "consensus_level": "moderate",
                   "expert_sources": [],
                   "disagreement_areas": []
               },
               "methodological_assessment": {
                   "overall_quality": "medium",
                   "strengths": ["Multiple sources consulted"],
                   "limitations": ["Limited expert analysis"]
               }
           },
           "overall_evidence_strength": "moderate",
           "confidence_level": avg_credibility,
           "evidence_gaps": ["Limited expert analysis", "Methodological assessment needed"]
       }

    async def _advanced_credibility_assessment(self, content: str, sources: List[Dict[str, Any]], 
                                            evidence_analysis: Dict[str, Any]) -> Dict[str, Any]:
       """Advanced multi-dimensional credibility assessment"""
       credibility_prompt = f"""
       Conduct a professional credibility assessment of this content and its supporting evidence:

       CONTENT TO ASSESS: "{content[:500]}..."

       EVIDENCE ANALYSIS SUMMARY:
       - Supporting Evidence Strength: {evidence_analysis.get('perspective_analyses', {}).get('supporting_evidence', {}).get('strength', 'unknown')}
       - Contradicting Evidence Strength: {evidence_analysis.get('perspective_analyses', {}).get('contradicting_evidence', {}).get('strength', 'unknown')}
       - Expert Consensus Level: {evidence_analysis.get('perspective_analyses', {}).get('expert_consensus', {}).get('consensus_level', 'unknown')}

       SOURCE QUALITY SUMMARY:
       {self._summarize_source_quality(sources)}

       Assess credibility across multiple dimensions:

       Return a JSON object:
       {{
           "dimensional_scores": {{
               "factual_accuracy": {{"score": 0.85, "justification": "reason"}},
               "source_reliability": {{"score": 0.78, "justification": "reason"}},
               "evidence_consistency": {{"score": 0.92, "justification": "reason"}},
               "expert_validation": {{"score": 0.70, "justification": "reason"}},
               "methodological_rigor": {{"score": 0.65, "justification": "reason"}},
               "temporal_relevance": {{"score": 0.88, "justification": "reason"}}
           }},
           "overall_credibility_score": 0.80,
           "credibility_level": "high|moderate|low",
           "key_strengths": ["strength1", "strength2"],
           "key_weaknesses": ["weakness1", "weakness2"],
           "improvement_recommendations": ["rec1", "rec2"]
       }}
       """
       
       try:
           response = await self.llm_manager.generate_response(credibility_prompt, max_tokens=1000, temperature=0.1)
           json_match = re.search(r'\{.*\}', response, re.DOTALL)
           
           if json_match:
               credibility_assessment = json.loads(json_match.group())
               credibility_assessment["assessment_timestamp"] = datetime.now().isoformat()
               return credibility_assessment
               
       except Exception as e:
           print(f"Credibility assessment error: {e}")
       
       return self._fallback_credibility_assessment(sources)

    def _summarize_source_quality(self, sources: List[Dict[str, Any]]) -> str:
       """Summarize overall source quality"""
       if not sources:
           return "No sources available"
       
       total_sources = len(sources)
       high_cred_sources = len([s for s in sources if s.get('credibility_score', {}).get('overall_score', 0) > 0.8])
       fact_check_sources = len([s for s in sources if s.get('credibility_score', {}).get('source_type') == 'fact_check'])
       academic_sources = len([s for s in sources if s.get('credibility_score', {}).get('source_type') == 'academic'])
       
       return f"""
       Total Sources: {total_sources}
       High Credibility (>0.8): {high_cred_sources} ({high_cred_sources/total_sources*100:.1f}%)
       Fact-Checking Sources: {fact_check_sources}
       Academic Sources: {academic_sources}
       Average Credibility: {sum(s.get('credibility_score', {}).get('overall_score', 0) for s in sources) / total_sources:.2f}
       """

    def _fallback_credibility_assessment(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
       """Fallback credibility assessment"""
       avg_credibility = sum(s.get('credibility_score', {}).get('overall_score', 0) for s in sources) / max(len(sources), 1)
       
       return {
           "dimensional_scores": {
               "factual_accuracy": {"score": avg_credibility, "justification": "Based on source credibility"},
               "source_reliability": {"score": avg_credibility, "justification": "Average of source scores"},
               "evidence_consistency": {"score": 0.6, "justification": "Limited analysis available"},
               "expert_validation": {"score": 0.5, "justification": "Insufficient expert sources"},
               "methodological_rigor": {"score": 0.5, "justification": "Cannot assess methodology"},
               "temporal_relevance": {"score": 0.7, "justification": "Recent sources included"}
           },
           "overall_credibility_score": avg_credibility,
           "credibility_level": "moderate" if avg_credibility > 0.6 else "low",
           "key_strengths": ["Multiple sources consulted"],
           "key_weaknesses": ["Limited expert analysis"],
           "improvement_recommendations": ["Seek expert validation", "Review methodology"]
       }

    async def _expert_domain_analysis(self, content: str, claims_analysis: Dict[str, Any], 
                                   evidence_analysis: Dict[str, Any]) -> Dict[str, Any]:
       """Analyze content from expert domain perspectives"""
       domain_categories = claims_analysis.get('primary_claims', [{}])[0].get('category', 'general')
       
       domain_prompt = f"""
       As a domain expert in {domain_categories}, analyze this content and claims:

       CONTENT: "{content[:400]}..."

       CLAIMS: {json.dumps([claim['claim'] for claim in claims_analysis.get('primary_claims', [])], indent=2)}

       EVIDENCE STRENGTH: {evidence_analysis.get('overall_evidence_strength', 'unknown')}

       Provide expert analysis from the perspective of {domain_categories} domain:

       Return JSON:
       {{
           "domain_assessments": {{
               "{domain_categories}": {{
                   "expert_consensus": "strong|moderate|weak|divided",
                   "established_facts": ["fact1", "fact2"],
                   "areas_of_uncertainty": ["area1", "area2"],
                   "methodological_concerns": ["concern1", "concern2"],
                   "domain_specific_credibility": 0.75
               }}
           }},
           "consensus_metrics": {{
               "overall_consensus": 0.70,
               "consensus_strength": "moderate",
               "dissenting_views": ["view1", "view2"]
           }},
           "expert_recommendations": ["rec1", "rec2"],
           "domain_confidence": 0.80
       }}
       """
       
       try:
           response = await self.llm_manager.generate_response(domain_prompt, max_tokens=800, temperature=0.1)
           json_match = re.search(r'\{.*\}', response, re.DOTALL)
           
           if json_match:
               domain_analysis = json.loads(json_match.group())
               domain_analysis["analysis_timestamp"] = datetime.now().isoformat()
               return domain_analysis
               
       except Exception as e:
           print(f"Domain analysis error: {e}")
       
       return self._fallback_domain_analysis(domain_categories)

    def _fallback_domain_analysis(self, domain_category: str) -> Dict[str, Any]:
       """Fallback domain analysis"""
       return {
           "domain_assessments": {
               domain_category: {
                   "expert_consensus": "moderate",
                   "established_facts": [],
                   "areas_of_uncertainty": ["Analysis incomplete"],
                   "methodological_concerns": ["Limited expert input"],
                   "domain_specific_credibility": 0.5
               }
           },
           "consensus_metrics": {
               "overall_consensus": 0.5,
               "consensus_strength": "weak",
               "dissenting_views": []
           },
           "expert_recommendations": ["Seek additional expert validation"],
           "domain_confidence": 0.4
       }

    async def _generate_comprehensive_verdict(self, content: str, search_results: Dict[str, Any],
                                           evidence_analysis: Dict[str, Any], credibility_assessment: Dict[str, Any],
                                           domain_analysis: Dict[str, Any]) -> Dict[str, Any]:
       """Generate comprehensive final verdict with detailed justification"""
       verdict_prompt = f"""
       Generate a comprehensive fact-checking verdict based on this analysis:

       CONTENT ANALYZED: "{content[:300]}..."

       ANALYSIS RESULTS:
       - Search Confidence: {search_results.get('confidence_metrics', {}).get('overall_confidence', 0):.2f}
       - Evidence Strength: {evidence_analysis.get('overall_evidence_strength', 'unknown')}
       - Overall Credibility: {credibility_assessment.get('overall_credibility_score', 0):.2f}
       - Expert Consensus: {domain_analysis.get('consensus_metrics', {}).get('overall_consensus', 0):.2f}

       KEY EVIDENCE SUMMARY:
       Supporting: {evidence_analysis.get('perspective_analyses', {}).get('supporting_evidence', {}).get('strength', 'unknown')}
       Contradicting: {evidence_analysis.get('perspective_analyses', {}).get('contradicting_evidence', {}).get('strength', 'unknown')}

       Generate a professional fact-checking verdict:

       Return JSON:
       {{
           "verdict": "TRUE|MOSTLY_TRUE|MIXED|MOSTLY_FALSE|FALSE|UNVERIFIABLE",
           "confidence_score": 0.85,
           "verdict_explanation": "detailed explanation of the verdict",
           "key_findings": [
               "finding1",
               "finding2",
               "finding3"
           ],
           "evidence_summary": {{
               "total_evidence_items": 15,
               "supporting_evidence_count": 8,
               "contradicting_evidence_count": 3,
               "neutral_evidence_count": 4,
               "evidence_quality_score": 0.78
           }},
           "limitations": [
               "limitation1",
               "limitation2"
           ],
           "recommendations": [
               "recommendation1",
               "recommendation2"
           ],
           "confidence_factors": {{
               "source_quality": 0.80,
               "evidence_consistency": 0.75,
               "expert_consensus": 0.70,
               "methodological_rigor": 0.65
           }}
       }}
       """
       
       try:
           response = await self.llm_manager.generate_response(verdict_prompt, max_tokens=1200, temperature=0.1)
           json_match = re.search(r'\{.*\}', response, re.DOTALL)
           
           if json_match:
               verdict_data = json.loads(json_match.group())
               verdict_data["verdict_timestamp"] = datetime.now().isoformat()
               verdict_data["analysis_completeness"] = self._calculate_verdict_completeness(verdict_data)
               return verdict_data
               
       except Exception as e:
           print(f"Verdict generation error: {e}")
       
       return self._fallback_verdict_generation(credibility_assessment, evidence_analysis)

    def _calculate_verdict_completeness(self, verdict_data: Dict[str, Any]) -> float:
       """Calculate completeness score for the verdict"""
       required_fields = ["verdict", "confidence_score", "verdict_explanation", "key_findings", "evidence_summary"]
       present_fields = sum(1 for field in required_fields if verdict_data.get(field))
       return present_fields / len(required_fields)

    def _fallback_verdict_generation(self, credibility_assessment: Dict[str, Any], 
                                  evidence_analysis: Dict[str, Any]) -> Dict[str, Any]:
       """Fallback verdict generation"""
       overall_credibility = credibility_assessment.get('overall_credibility_score', 0.5)
       evidence_strength = evidence_analysis.get('overall_evidence_strength', 'moderate')
       
       # Simple verdict logic
       if overall_credibility > 0.8 and evidence_strength == 'strong':
           verdict = "MOSTLY_TRUE"
       elif overall_credibility > 0.6 and evidence_strength in ['strong', 'moderate']:
           verdict = "MIXED"
       elif overall_credibility < 0.4:
           verdict = "MOSTLY_FALSE"
       else:
           verdict = "UNVERIFIABLE"
       
       return {
           "verdict": verdict,
           "confidence_score": overall_credibility,
           "verdict_explanation": "Verdict based on automated analysis due to processing limitations",
           "key_findings": [
               f"Overall credibility score: {overall_credibility:.2f}",
               f"Evidence strength: {evidence_strength}",
               "Analysis completed with limited expert input"
           ],
           "evidence_summary": {
               "total_evidence_items": len(credibility_assessment.get('key_strengths', [])) + len(credibility_assessment.get('key_weaknesses', [])),
               "supporting_evidence_count": len(credibility_assessment.get('key_strengths', [])),
               "contradicting_evidence_count": len(credibility_assessment.get('key_weaknesses', [])),
               "neutral_evidence_count": 0,
               "evidence_quality_score": overall_credibility
           },
           "limitations": [
               "Limited expert analysis",
               "Automated processing only",
               "May require additional verification"
           ],
           "recommendations": [
               "Seek additional expert validation",
               "Review primary sources",
               "Consider alternative perspectives"
           ],
           "confidence_factors": {
               "source_quality": overall_credibility,
               "evidence_consistency": 0.6,
               "expert_consensus": 0.4,
               "methodological_rigor": 0.5
           },
           "verdict_timestamp": datetime.now().isoformat(),
           "analysis_completeness": 0.7
       }

    def _calculate_analysis_completeness(self, analysis_result: Dict[str, Any]) -> float:
       """Calculate overall analysis completeness score"""
       required_stages = ["content_preprocessing", "search_pipeline", "evidence_analysis", 
                         "credibility_assessment", "domain_analysis", "verdict_generation"]
       
       completed_stages = sum(1 for stage in required_stages 
                            if stage in analysis_result.get("pipeline_stages", {}))
       
       base_completeness = completed_stages / len(required_stages)
       
       # Bonus for additional components
       bonus = 0
       if analysis_result.get("image_analysis"):
           bonus += 0.1
       if analysis_result.get("search_results", {}).get("confidence_metrics", {}).get("overall_confidence", 0) > 0.7:
           bonus += 0.1
       
       return min(1.0, base_completeness + bonus)

llm_manager = LLMManager()
image_processor = AdvancedImageProcessor()
fact_checker = AdvancedFactChecker()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ValiData</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --secondary: #0f172a;
            --accent: #06b6d4;
            --success: #059669;
            --warning: #d97706;
            --error: #dc2626;
            --neutral: #64748b;
            --surface: #ffffff;
            --surface-alt: #f8fafc;
            --border: #e2e8f0;
            --text: #0f172a;
            --text-muted: #64748b;
            --shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-surface: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f172a;
            background-image: 
                radial-gradient(at 40% 20%, hsla(228, 62%, 59%, 0.12) 0px, transparent 50%),
                radial-gradient(at 80% 0%, hsla(189, 94%, 43%, 0.12) 0px, transparent 50%),
                radial-gradient(at 0% 50%, hsla(355, 85%, 93%, 0.12) 0px, transparent 50%),
                radial-gradient(at 80% 50%, hsla(340, 100%, 76%, 0.12) 0px, transparent 50%),
                radial-gradient(at 0% 100%, hsla(22, 100%, 77%, 0.12) 0px, transparent 50%),
                radial-gradient(at 80% 100%, hsla(242, 100%, 70%, 0.12) 0px, transparent 50%),
                radial-gradient(at 0% 0%, hsla(343, 100%, 76%, 0.12) 0px, transparent 50%);
            min-height: 100vh;
            color: var(--text);
            overflow-x: hidden;
        }
        
        .main-container {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            position: relative;
        }
        
        .background-elements {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            overflow: hidden;
        }
        
        .floating-shape {
            position: absolute;
            border-radius: 50%;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(167, 139, 253, 0.1));
            animation: float 6s ease-in-out infinite;
        }
        
        .shape-1 {
            width: 300px;
            height: 300px;
            top: 10%;
            left: 10%;
            animation-delay: 0s;
        }
        
        .shape-2 {
            width: 200px;
            height: 200px;
            top: 60%;
            right: 15%;
            animation-delay: 2s;
        }
        
        .shape-3 {
            width: 150px;
            height: 150px;
            bottom: 20%;
            left: 20%;
            animation-delay: 4s;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }
        
        .platform-container {
            background: var(--gradient-surface);
            backdrop-filter: blur(20px);
            border-radius: 32px;
            box-shadow: var(--shadow-lg);
            padding: 3rem;
            max-width: 900px;
            width: 100%;
            border: 1px solid rgba(255, 255, 255, 0.2);
            position: relative;
            z-index: 1;
        }
        
        .header-section {
            text-align: center;
            margin-bottom: 3rem;
        }
        
        .brand-logo {
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }
        
        .logo-icon {
            width: 48px;
            height: 48px;
            background: var(--gradient-primary);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 24px;
            font-weight: bold;
        }
        
        .brand-name {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .tagline {
            color: var(--text-muted);
            font-size: 1.125rem;
            font-weight: 400;
            margin-top: 0.5rem;
        }
        
        .verification-form {
            margin-bottom: 2rem;
        }
        
        .input-group {
            margin-bottom: 2rem;
        }
        
        .input-label {
            display: block;
            margin-bottom: 0.75rem;
            font-weight: 600;
            color: var(--text);
            font-size: 0.95rem;
        }
        
        .content-input {
            width: 100%;
            padding: 1.25rem;
            border: 2px solid var(--border);
            border-radius: 16px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            min-height: 140px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            background: var(--surface);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .content-input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 
                0 0 0 3px rgba(37, 99, 235, 0.1),
                0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        .upload-zone {
            position: relative;
            border: 2px dashed var(--border);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
            background: var(--surface-alt);
            cursor: pointer;
        }
        
        .upload-zone:hover {
            border-color: var(--primary);
            background: rgba(37, 99, 235, 0.02);
        }
        
        .upload-zone.dragover {
            border-color: var(--accent);
            background: rgba(6, 182, 212, 0.05);
            transform: scale(1.02);
        }
        
        .upload-zone.has-file {
            border-color: var(--success);
            background: rgba(5, 150, 105, 0.05);
        }
        
        .upload-input {
            position: absolute;
            opacity: 0;
            pointer-events: none;
        }
        
        .upload-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.75rem;
        }
        
        .upload-icon {
            width: 48px;
            height: 48px;
            background: var(--gradient-primary);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 20px;
        }
        
        .upload-text {
            font-weight: 500;
            color: var(--text);
        }
        
        .upload-subtext {
            font-size: 0.875rem;
            color: var(--text-muted);
        }
        
        .verify-button {
            width: 100%;
            padding: 1.25rem 2rem;
            background: var(--gradient-primary);
            color: white;
            border: none;
            border-radius: 16px;
            font-size: 1.125rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .verify-button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(37, 99, 235, 0.4);
        }
        
        .verify-button:active {
            transform: translateY(0);
        }
        
        .verify-button:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
        }
        
        .button-content {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .analysis-progress {
            display: none;
            text-align: center;
            margin: 2rem 0;
            padding: 2rem;
            background: var(--surface-alt);
            border-radius: 20px;
            border: 1px solid var(--border);
        }
        
        .progress-spinner {
            width: 48px;
            height: 48px;
            border: 4px solid var(--border);
            border-top: 4px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .progress-text {
            color: var(--text-muted);
            font-weight: 500;
        }
        
        .results-panel {
            display: none;
            margin-top: 2rem;
            border-radius: 24px;
            overflow: hidden;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
        }
        
        .results-header {
            padding: 2rem;
            background: var(--gradient-surface);
            border-bottom: 1px solid var(--border);
        }
        
        .verdict-display {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .verdict-badge {
            padding: 0.5rem 1rem;
            border-radius: 24px;
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .verdict-badge.verified {
            background: var(--success);
            color: white;
        }
        
        .verdict-badge.false {
            background: var(--error);
            color: white;
        }
        
        .verdict-badge.misleading {
            background: var(--warning);
            color: white;
        }
        
        .verdict-badge.mixed {
            background: #8b5cf6;
            color: white;
        }
        
        .verdict-badge.insufficient_evidence {
            background: var(--neutral);
            color: white;
        }
        
        .confidence-meter {
            flex: 1;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .confidence-track {
            flex: 1;
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--error), var(--warning), var(--success));
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 4px;
        }
        
        .confidence-value {
            font-weight: 600;
            color: var(--text);
            min-width: 3rem;
        }
        
        .results-content {
            padding: 2rem;
            background: var(--surface);
        }
        
        .result-section {
            margin-bottom: 2rem;
        }
        
        .result-section:last-child {
            margin-bottom: 0;
        }
        
        .section-title {
            color: var(--text);
            margin-bottom: 0.75rem;
            font-size: 1.125rem;
            font-weight: 600;
        }
        
        .section-content {
            color: var(--text-muted);
            line-height: 1.7;
        }
        
        .section-list {
            color: var(--text-muted);
            line-height: 1.7;
            padding-left: 1.25rem;
        }
        
        .sources-grid {
            display: grid;
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .source-card {
            padding: 1.25rem;
            background: var(--surface-alt);
            border-radius: 12px;
            border: 1px solid var(--border);
            transition: all 0.2s ease;
        }
        
        .source-card:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        .source-header {
            font-weight: 600;
            color: var(--text);
            margin-bottom: 0.5rem;
            font-size: 0.95rem;
        }
        
        .source-domain {
            font-size: 0.875rem;
            color: var(--primary);
            margin-bottom: 0.75rem;
            font-weight: 500;
        }
        
        .source-excerpt {
            font-size: 0.875rem;
            color: var(--text-muted);
            line-height: 1.6;
            margin-bottom: 0.75rem;
        }
        
        .source-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .credibility-score {
            padding: 0.25rem 0.75rem;
            background: rgba(37, 99, 235, 0.1);
            color: var(--primary);
            border-radius: 16px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .analysis-metadata {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            padding: 1.5rem;
            background: var(--surface-alt);
            border-top: 1px solid var(--border);
            font-size: 0.875rem;
        }
        
        .metadata-item {
            text-align: center;
        }
        
        .metadata-value {
            font-weight: 600;
            color: var(--text);
            font-size: 1.25rem;
            display: block;
        }
        
        .metadata-label {
            color: var(--text-muted);
            margin-top: 0.25rem;
        }
        
        .error-display {
            padding: 1.5rem;
            background: rgba(220, 38, 38, 0.05);
            border: 1px solid rgba(220, 38, 38, 0.2);
            border-radius: 12px;
            color: var(--error);
            text-align: center;
        }
        
        @media (max-width: 768px) {
            .main-container {
                padding: 1rem;
            }
            
            .platform-container {
                padding: 2rem;
                border-radius: 24px;
            }
            
            .brand-name {
                font-size: 2rem;
            }
            
            .content-input {
                min-height: 120px;
            }
            
            .upload-zone {
                padding: 1.5rem;
            }
            
            .sources-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="background-elements">
        <div class="floating-shape shape-1"></div>
        <div class="floating-shape shape-2"></div>
        <div class="floating-shape shape-3"></div>
    </div>
    
    <div class="main-container">
        <div class="platform-container">
            <div class="header-section">
                <div class="brand-logo">
                    <div class="logo-icon">✓</div>
                    <h1 class="brand-name">ValiData</h1>
                </div>
                <p class="tagline">A fact-verification and source analysis platform for the 21st century.</p>
            </div>
            
            <form class="verification-form" id="verificationForm" enctype="multipart/form-data">
                <div class="input-group">
                    <label class="input-label" for="contentInput">Content to Verify</label>
                    <textarea 
                        class="content-input" 
                        id="contentInput" 
                        name="content" 
                        required 
                        placeholder="Enter the statement, claim, or information you'd like to verify..."
                    ></textarea>
                </div>
                
                <div class="input-group">
                    <label class="input-label" for="imageInput">Supporting Media (Optional)</label>
                    <div class="upload-zone" id="uploadZone">
                        <input type="file" class="upload-input" id="imageInput" name="image" accept="image/*">
                        <div class="upload-content">
                            <div class="upload-icon">📎</div>
                            <div class="upload-text" id="uploadText">Drop image here or click to browse</div>
                            <div class="upload-subtext">Supports JPG, PNG, WebP up to 10MB</div>
                        </div>
                    </div>
                </div>
                
                <button type="submit" class="verify-button" id="verifyButton">
                    <div class="button-content">
                        <span>Verify Information</span>
                    </div>
                </button>
            </form>
            
            <div class="analysis-progress" id="analysisProgress">
                <div class="progress-spinner"></div>
                <div class="progress-text">Analyzing content and cross-referencing sources...</div>
            </div>
            
            <div class="results-panel" id="resultsPanel">
                <!-- Results will be populated here -->
            </div>
        </div>
    </div>
    
    <script>
        const form = document.getElementById('verificationForm');
        const analysisProgress = document.getElementById('analysisProgress');
        const resultsPanel = document.getElementById('resultsPanel');
        const verifyButton = document.getElementById('verifyButton');
        const imageInput = document.getElementById('imageInput');
        const uploadZone = document.getElementById('uploadZone');
        const uploadText = document.getElementById('uploadText');
        
        // File upload handling
        imageInput.addEventListener('change', handleFileSelect);
        uploadZone.addEventListener('click', () => imageInput.click());
        uploadZone.addEventListener('dragover', handleDragOver);
        uploadZone.addEventListener('dragleave', handleDragLeave);
        uploadZone.addEventListener('drop', handleFileDrop);
        
        function handleFileSelect() {
            if (imageInput.files && imageInput.files[0]) {
                const file = imageInput.files[0];
                uploadText.textContent = file.name;
                uploadZone.classList.add('has-file');
            } else {
                uploadText.textContent = 'Drop image here or click to browse';
                uploadZone.classList.remove('has-file');
            }
        }
        
        function handleDragOver(e) {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        }
        
        function handleDragLeave(e) {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
        }
        
        function handleFileDrop(e) {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                imageInput.files = files;
                handleFileSelect();
            }
        }
        
        // Form submission
        form.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(form);
    
    verifyButton.disabled = true;
    analysisProgress.style.display = 'block';
    resultsPanel.style.display = 'none';
    
    try {
        const response = await fetch('http://localhost:8000/fact-check', {
    method: 'POST',
    body: formData
});

        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data);
        } else {
            displayError(data.detail || data.error?.message || 'Verification process encountered an error');
        }
    } catch (error) {
        console.error(error);
        displayError('Unable to connect to verification service');
    } finally {
        verifyButton.disabled = false;
        analysisProgress.style.display = 'none';
    }
});

function displayResults(data) {
    // Handle error case
    if (data.error) {
        displayError(`${data.error.type}: ${data.error.message}`);
        return;
    }
    
    // Extract data from complex nested structure
    const finalVerdict = data.final_verdict || {};
    const evidenceAnalysis = data.evidence_analysis || {};
    const credibilityAssessment = data.credibility_assessment || {};
    const searchResults = data.search_results || {};
    const domainAnalysis = data.domain_analysis || {};
    
    // Extract basic verdict info
    const verdict = finalVerdict.verdict_category || finalVerdict.verdict || 'unknown';
    const confidence = Math.round(finalVerdict.confidence_score || 0);
    const summary = finalVerdict.explanation || finalVerdict.summary || 'No summary available';
    
    // Extract key findings
    const keyFindings = [];
    if (finalVerdict.key_points) {
        keyFindings.push(...finalVerdict.key_points);
    }
    if (evidenceAnalysis.key_insights) {
        keyFindings.push(...evidenceAnalysis.key_insights);
    }
    if (domainAnalysis.key_findings) {
        keyFindings.push(...domainAnalysis.key_findings);
    }
    
    // Extract evidence for and against
    const evidenceFor = [];
    const evidenceAgainst = [];
    
    if (evidenceAnalysis.supporting_evidence) {
        evidenceFor.push(...evidenceAnalysis.supporting_evidence);
    }
    if (evidenceAnalysis.contradicting_evidence) {
        evidenceAgainst.push(...evidenceAnalysis.contradicting_evidence);
    }
    if (finalVerdict.evidence_summary?.supporting_points) {
        evidenceFor.push(...finalVerdict.evidence_summary.supporting_points);
    }
    if (finalVerdict.evidence_summary?.contradicting_points) {
        evidenceAgainst.push(...finalVerdict.evidence_summary.contradicting_points);
    }
    
    // Extract context and recommendations
    const context = finalVerdict.context || 
                   credibilityAssessment.context_analysis || 
                   domainAnalysis.contextual_factors?.join('. ') || '';
    
    const recommendations = finalVerdict.recommendations || 
                           credibilityAssessment.recommendations || 
                           domainAnalysis.recommendations || '';
    
    // Extract and format sources
    const sources = [];
    if (searchResults.final_sources) {
        sources.push(...searchResults.final_sources.map(source => ({
            title: source.title || source.name || 'Unknown Source',
            domain: source.domain || source.url || source.source || 'Unknown Domain',
            snippet: source.snippet || source.summary || source.content?.substring(0, 200) || 'No preview available',
            credibility_score: source.credibility_score || source.reliability_score || 0.5
        })));
    }
    if (searchResults.credible_sources) {
        sources.push(...searchResults.credible_sources.map(source => ({
            title: source.title || 'Credible Source',
            domain: source.domain || source.url || 'Unknown Domain',
            snippet: source.snippet || source.summary || 'No preview available',
            credibility_score: source.credibility_score || 0.8
        })));
    }
    
    // Extract metadata
    const totalResults = searchResults.search_metadata?.total_results || 
                        searchResults.total_sources_found || 
                        sources.length || 0;
    
    const factCheckSources = searchResults.search_metadata?.fact_check_sources ||
                            searchResults.fact_check_sources?.length ||
                            sources.filter(s => s.credibility_score > 0.7).length || 0;
    
    const analysisTime = data.total_pipeline_duration || 
                        Object.values(data.pipeline_stages || {})
                              .reduce((sum, stage) => sum + (stage.duration || 0), 0) || 0;
    
    resultsPanel.innerHTML = `
        <div class="results-header">
            <div class="verdict-display">
                <span class="verdict-badge ${verdict.toLowerCase()}">${formatVerdict(verdict)}</span>
                <div class="confidence-meter">
                    <div class="confidence-track">
                        <div class="confidence-fill" style="width: ${confidence}%"></div>
                    </div>
                    <span class="confidence-value">${confidence}%</span>
                </div>
            </div>
        </div>
        
        <div class="results-content">
            <div class="result-section">
                <h3 class="section-title">Analysis Summary</h3>
                <p class="section-content">${summary}</p>
            </div>
            
            ${keyFindings.length > 0 ? `
                <div class="result-section">
                    <h3 class="section-title">Key Findings</h3>
                    <ul class="section-list">
                        ${keyFindings.map(finding => `<li>${finding}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
            
            ${evidenceFor.length > 0 ? `
                <div class="result-section">
                    <h3 class="section-title">Supporting Evidence</h3>
                    <ul class="section-list">
                        ${evidenceFor.map(evidence => `<li>${evidence}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
            
            ${evidenceAgainst.length > 0 ? `
                <div class="result-section">
                    <h3 class="section-title">Contradicting Evidence</h3>
                    <ul class="section-list">
                        ${evidenceAgainst.map(evidence => `<li>${evidence}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
            
            ${context ? `
                <div class="result-section">
                    <h3 class="section-title">Additional Context</h3>
                    <p class="section-content">${context}</p>
                </div>
            ` : ''}
            
            ${recommendations ? `
                <div class="result-section">
                    <h3 class="section-title">Recommendations</h3>
                    <p class="section-content">${recommendations}</p>
                </div>
            ` : ''}
            
            ${sources.length > 0 ? `
                <div class="result-section">
                    <h3 class="section-title">Source Analysis</h3>
                    <div class="sources-grid">
                        ${sources.slice(0, 5).map(source => `
                            <div class="source-card">
                                <div class="source-header">${source.title}</div>
                                <div class="source-domain">${source.domain}</div>
                                <div class="source-excerpt">${source.snippet}</div>
                                <div class="source-meta">
                                    <span class="credibility-score">Reliability: ${Math.round(source.credibility_score * 100)}%</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
            
            ${data.pipeline_stages ? `
                <div class="result-section">
                    <h3 class="section-title">Analysis Pipeline</h3>
                    <div class="pipeline-stages">
                        ${Object.entries(data.pipeline_stages).map(([stage, info]) => `
                            <div class="pipeline-stage">
                                <span class="stage-name">${formatStageName(stage)}</span>
                                <span class="stage-duration">${info.duration}s</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
        
        <div class="analysis-metadata">
            <div class="metadata-item">
                <span class="metadata-value">${analysisTime.toFixed(1)}s</span>
                <div class="metadata-label">Processing Time</div>
            </div>
            <div class="metadata-item">
                <span class="metadata-value">${totalResults}</span>
                <div class="metadata-label">Sources Analyzed</div>
            </div>
            <div class="metadata-item">
                <span class="metadata-value">${factCheckSources}</span>
                <div class="metadata-label">Fact-Check Sources</div>
            </div>
            ${data.analysis_completeness ? `
                <div class="metadata-item">
                    <span class="metadata-value">${Math.round(data.analysis_completeness * 100)}%</span>
                    <div class="metadata-label">Analysis Completeness</div>
                </div>
            ` : ''}
        </div>
    `;
    
    resultsPanel.style.display = 'block';
    
    // Smooth scroll to results
    setTimeout(() => {
        resultsPanel.scrollIntoView({ behavior: 'smooth' });
    }, 100);
}

function displayError(message) {
    resultsPanel.innerHTML = `
        <div class="results-content">
            <div class="error-display">
                <h3>Verification Error</h3>
                <p>${message}</p>
            </div>
        </div>
    `;
    resultsPanel.style.display = 'block';
}

// Helper function to format stage names for display
function formatStageName(stageName) {
    return stageName
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
}

// Helper function to format verdict display (assuming this exists)
function formatVerdict(verdict) {
    const verdictMap = {
        'true': 'TRUE',
        'false': 'FALSE',
        'mostly-true': 'MOSTLY TRUE',
        'mostly-false': 'MOSTLY FALSE',
        'mixed': 'MIXED',
        'unverified': 'UNVERIFIED',
        'unknown': 'UNKNOWN'
    };
    return verdictMap[verdict.toLowerCase()] || verdict.toUpperCase();
}
    </script>
</body>
</html>
    """

@app.post("/fact-check")
async def fact_check_endpoint(
    content: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    try:
        if not content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty")
        
        if len(content) > 5000:
            raise HTTPException(status_code=400, detail="Content too long (max 5000 characters)")
        
        image_obj = None
        if image is not None and image.filename:
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            image_data = await image.read()
            if len(image_data) > 10 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
            
            try:
                image_obj = Image.open(io.BytesIO(image_data))
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image file")
        
        result = await fact_checker.comprehensive_fact_check_pipeline(content, image_obj)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Fact-check error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during fact-checking")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "llm_provider": LLM_PROVIDER,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)