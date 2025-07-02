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

class EnhancedSearchEngine:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.fact_check_sources = [
            {"domain": "snopes.com", "weight": 0.95, "bias": "neutral", "type": "fact_check"},
            {"domain": "factcheck.org", "weight": 0.90, "bias": "neutral", "type": "fact_check"},
            {"domain": "politifact.com", "weight": 0.85, "bias": "slight_left", "type": "fact_check"},
            {"domain": "reuters.com", "weight": 0.92, "bias": "neutral", "type": "news"},
            {"domain": "apnews.com", "weight": 0.90, "bias": "neutral", "type": "news"},
            {"domain": "bbc.com", "weight": 0.88, "bias": "neutral", "type": "news"},
            {"domain": "npr.org", "weight": 0.85, "bias": "slight_left", "type": "news"},
            {"domain": "pbs.org", "weight": 0.87, "bias": "neutral", "type": "news"},
            {"domain": "washingtonpost.com", "weight": 0.78, "bias": "left", "type": "news"},
            {"domain": "nytimes.com", "weight": 0.80, "bias": "left", "type": "news"},
            {"domain": "wsj.com", "weight": 0.82, "bias": "slight_right", "type": "news"},
            {"domain": "cnn.com", "weight": 0.72, "bias": "left", "type": "news"},
            {"domain": "foxnews.com", "weight": 0.65, "bias": "right", "type": "news"}
        ]
        
        self.academic_domains = [".edu", ".gov", ".org"]
        self.misinformation_keywords = [
            "debunked", "false", "fake", "hoax", "misleading", "incorrect", 
            "conspiracy", "disinformation", "fabricated", "unsubstantiated"
        ]
        self.verification_keywords = [
            "verified", "confirmed", "accurate", "factual", "true", "legitimate", 
            "authenticated", "corroborated", "substantiated", "validated"
        ]
    
    async def intelligent_query_generation(self, content: str) -> List[str]:
        prompt = f"""
        Generate 6-8 diverse and strategic search queries to fact-check this content. Create queries that would find:
        1. Direct fact-checking results
        2. Original source verification
        3. Contradictory information
        4. Expert opinions
        5. Recent developments
        
        Content: "{content[:500]}"
        
        Return only the search queries, one per line, without numbering or explanation:
        """
        
        try:
            response = await self.llm_manager.generate_response(prompt, max_tokens=300)
            if not response or not isinstance(response, str):
                raise ValueError("LLM returned empty or invalid response")
            
            queries = [q.strip() for q in response.split('\n') if q.strip() and len(q.strip()) > 5]
            
            if len(queries) < 4:
                queries.extend(self._fallback_queries(content))
            
            return queries[:8]
        except Exception as e:
            print(f"[QueryGen Error] {e}")
            return self._fallback_queries(content)

    def _fallback_queries(self, content: str) -> List[str]:
        words = content.split()
        key_phrases = []
        
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if len(phrase) > 10:
                key_phrases.append(phrase)
        
        queries = [
            f'"{content[:80]}" fact check',
            f"{' '.join(words[:8])} debunked false",
            f"{' '.join(words[:8])} verified true",
            f"{' '.join(words[:8])} misinformation",
        ]
        
        if key_phrases:
            queries.extend([f'"{phrase}" verification' for phrase in key_phrases[:2]])
        
        return queries
    
    async def enhanced_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        all_results = []
        
        queries =[q for q in queries if isinstance(q, (str, int, float)) and q is not None and str(q).strip() != ""]
        async with aiohttp.ClientSession() as session:
            tasks = [self._single_search(session, str(query)) for query in queries]
            search_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for results in search_results:
                if isinstance(results, list):
                    all_results.extend(results)
        
        deduplicated = self._deduplicate_results(all_results)
        scored_results = self._score_results(deduplicated)
        
        return sorted(scored_results, key=lambda x: x['relevance_score'], reverse=True)[:20]
    
    async def _single_search(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        try:
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': GOOGLE_API_KEY,
                'cx': GOOGLE_CSE_ID,
                'q': query,
                'num': 10,
                'safe': 'medium',
                'dateRestrict': 'y5'
            }
            
            async with session.get(search_url, params=params, timeout=20) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for item in data.get('items', []):
                        result = {
                            "title": item.get('title', ''),
                            "link": item.get('link', ''),
                            "snippet": item.get('snippet', ''),
                            "domain": urlparse(item.get('link', '')).netloc.lower(),
                            "query_used": query,
                            "publish_date": self._extract_date(item),
                            "credibility_score": self._calculate_credibility(item.get('link', '')),
                            "source_type": self._classify_source(item.get('link', '')),
                            "content_indicators": self._analyze_content_indicators(item.get('title', '') + ' ' + item.get('snippet', ''))
                        }
                        results.append(result)
                    
                    return results
        except Exception as e:
            print(f"Search error for query '{query}': {e}")
            return []
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_urls = set()
        seen_titles = set()
        deduplicated = []
        
        for result in results:
            url = result['link']
            title = result['title'].lower()
            
            title_words = set(title.split())
            is_duplicate = any(len(title_words.intersection(set(seen_title.split()))) > len(title_words) * 0.7 
                              for seen_title in seen_titles)
            
            if url not in seen_urls and not is_duplicate:
                seen_urls.add(url)
                seen_titles.add(title)
                deduplicated.append(result)
        
        return deduplicated
    
    def _score_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for result in results:
            score = result['credibility_score']
            
            if result['source_type'] == 'fact_check':
                score += 0.3
            elif result['source_type'] == 'academic':
                score += 0.25
            elif result['source_type'] == 'news':
                score += 0.1
            
            content_text = (result['title'] + ' ' + result['snippet']).lower()
            
            misinfo_count = sum(1 for keyword in self.misinformation_keywords if keyword in content_text)
            verify_count = sum(1 for keyword in self.verification_keywords if keyword in content_text)
            
            if misinfo_count > verify_count:
                score += 0.2
            elif verify_count > misinfo_count:
                score += 0.15
            
            age_factor = self._calculate_age_factor(result.get('publish_date'))
            score += age_factor
            
            result['relevance_score'] = min(1.0, score)
        
        return results
    
    def _calculate_credibility(self, url: str) -> float:
        domain = urlparse(url).netloc.lower()
        
        for source in self.fact_check_sources:
            if source["domain"] in domain:
                return source["weight"]
        
        if any(academic in domain for academic in self.academic_domains):
            return 0.85
        elif "wikipedia.org" in domain:
            return 0.75
        else:
            return 0.4
    
    def _classify_source(self, url: str) -> str:
        domain = urlparse(url).netloc.lower()
        
        for source in self.fact_check_sources:
            if source["domain"] in domain:
                return source["type"]
        
        if any(academic in domain for academic in self.academic_domains):
            return "academic"
        elif "wikipedia.org" in domain:
            return "encyclopedia"
        else:
            return "other"
    
    def _analyze_content_indicators(self, text: str) -> Dict[str, int]:
        text_lower = text.lower()
        
        return {
            "misinformation_signals": sum(1 for keyword in self.misinformation_keywords if keyword in text_lower),
            "verification_signals": sum(1 for keyword in self.verification_keywords if keyword in text_lower),
            "emotional_language": len(re.findall(r'\b(shocking|amazing|unbelievable|urgent|breaking)\b', text_lower)),
            "factual_language": len(re.findall(r'\b(study|research|data|evidence|report|analysis)\b', text_lower))
        }
    
    def _extract_date(self, item: Dict) -> Optional[str]:
        date_fields = ['pagemap', 'metatags']
        for field in date_fields:
            if field in item:
                if isinstance(item[field], dict):
                    for key, value in item[field].items():
                        if 'date' in key.lower() and isinstance(value, str):
                            return value
        return None
    
    def _calculate_age_factor(self, date_str: Optional[str]) -> float:
        if not date_str:
            return 0.0
        
        try:
            from dateutil import parser
            date_obj = parser.parse(date_str)
            days_old = (datetime.now() - date_obj).days
            
            if days_old <= 30:
                return 0.1
            elif days_old <= 365:
                return 0.05
            else:
                return 0.0
        except:
            return 0.0
    
    async def filter_relevant_sources(self, search_results: List[Dict[str, Any]], original_content: str) -> List[Dict[str, Any]]:
        if not search_results:
            return []
        
        results_text = "\n".join([
            f"Title: {r['title']}\nSnippet: {r['snippet']}\nDomain: {r['domain']}\nCredibility: {r['credibility_score']:.2f}\n---"
            for r in search_results[:15]
        ])
        
        prompt = f"""
        Analyze these search results and identify which ones are most relevant for fact-checking this content:
        
        ORIGINAL CONTENT: "{original_content[:400]}"
        
        SEARCH RESULTS:
        {results_text}
        
        Return a JSON array with the indices (0-based) of the most relevant results for fact-checking, ordered by relevance.
        Consider: direct contradictions, confirmations, authoritative sources, and recent information.
        
        Format: [0, 3, 7, 12, ...]
        """
        
        try:
            response = await self.llm_manager.generate_response(prompt, max_tokens=200)
            
            indices_match = re.search(r'\[([\d,\s]+)\]', response)
            if indices_match:
                indices = [int(x.strip()) for x in indices_match.group(1).split(',') if x.strip().isdigit()]
                relevant_results = [search_results[i] for i in indices if i < len(search_results)]
                return relevant_results[:10]
        except:
            pass
        
        return search_results[:8]

class AdvancedFactChecker:
    def __init__(self):
        self.llm_manager = LLMManager()
        self.image_processor = AdvancedImageProcessor()
        self.search_engine = EnhancedSearchEngine(self.llm_manager)
        
        self.credibility_patterns = {
            'high_credibility': [
                r'according to (?:a )?(?:study|research|report)',
                r'peer-reviewed',
                r'published in',
                r'university of',
                r'institute of',
                r'clinical trial',
                r'meta-analysis'
            ],
            'low_credibility': [
                r'doctors hate (?:this|him)',
                r'one weird trick',
                r'shocking (?:truth|discovery)',
                r'they don\'t want you to know',
                r'big pharma (?:doesn\'t want|hates)',
                r'mainstream media (?:won\'t tell|ignores)',
                r'wake up (?:people|sheeple)',
                r'do your own research'
            ]
        }
    
    async def comprehensive_fact_check(self, content: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "content_analyzed": content[:300] + ("..." if len(content) > 300 else ""),
            "image_provided": image is not None,
            "processing_steps": []
        }
        
        if image:
            analysis_result["processing_steps"].append("image_processing")
            image_analysis = self.image_processor.extract_text_comprehensive(image)
            image_metadata = self.image_processor.analyze_image_metadata(image)
            
            if image_analysis["text"]:
                content = f"{content} {image_analysis['text']}".strip()
                analysis_result["image_text"] = image_analysis["text"]
                analysis_result["image_confidence"] = image_analysis["confidence"]
                analysis_result["image_metadata"] = image_metadata
        
        analysis_result["processing_steps"].append("query_generation")
        search_queries = await self.search_engine.intelligent_query_generation(content)
        
        analysis_result["processing_steps"].append("web_search")
        search_results = await self.search_engine.enhanced_search(search_queries)
        
        analysis_result["processing_steps"].append("source_filtering")
        relevant_sources = await self.search_engine.filter_relevant_sources(search_results, content)
        
        analysis_result["processing_steps"].append("llm_analysis")
        fact_check_result = await self._comprehensive_llm_analysis(content, relevant_sources)
        
        analysis_result.update(fact_check_result)
        analysis_result["search_metadata"] = {
            "queries_generated": len(search_queries),
            "total_results": len(search_results),
            "relevant_sources": len(relevant_sources),
            "fact_check_sources": len([s for s in relevant_sources if s.get('source_type') == 'fact_check']),
            "average_credibility": sum(s.get('credibility_score', 0) for s in relevant_sources) / len(relevant_sources) if relevant_sources else 0
        }
        
        analysis_result["sources"] = relevant_sources[:10]
        analysis_result["analysis_duration"] = round(time.time() - start_time, 2)
        
        return analysis_result
    
    async def _comprehensive_llm_analysis(self, content: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        sources_context = self._format_sources_for_llm(sources)
        
        analysis_prompt = f"""
        You are an expert fact-checker with access to multiple reliable sources. Analyze this content for factual accuracy.

        CONTENT TO FACT-CHECK:
        "{content}"

        RELEVANT SOURCES FOUND:
        {sources_context}

        Provide a comprehensive fact-check analysis in this exact JSON format:
        {{
            "verdict": "<verified|false|misleading|mixed|insufficient_evidence>",
            "confidence": <integer 0-100>,
            "summary": "<2-3 sentence explanation of your verdict>",
            "key_findings": [
                "<specific finding 1>",
                "<specific finding 2>",
                "<specific finding 3>"
            ],
            "evidence_for": [
                "<evidence supporting the claim>",
                "<another supporting evidence>"
            ],
            "evidence_against": [
                "<evidence contradicting the claim>",
                "<another contradicting evidence>"
            ],
            "credibility_assessment": {{
                "source_quality": <integer 1-10>,
                "consistency": <integer 1-10>,
                "corroboration": <integer 1-10>
            }},
            "risk_indicators": [
                "<red flag 1>",
                "<red flag 2>"
            ],
            "context": "<important context or nuance>",
            "recommendations": "<what readers should know>"
        }}

        Guidelines:
        - Use "verified" only for well-established facts with strong evidence
        - Use "false" for clearly debunked or fabricated claims
        - Use "misleading" for partially true but distorted information
        - Use "mixed" for claims with both true and false elements
        - Use "insufficient_evidence" when sources are inadequate
        - Confidence should reflect the strength of available evidence
        - Be specific about what makes sources credible or questionable
        """
        
        try:
            response = await self.llm_manager.generate_response(analysis_prompt, max_tokens=1500, temperature=0.1)
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return self._validate_and_enhance_result(result, content, sources)
            else:
                return self._parse_unstructured_response(response, content, sources)
                
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return self._fallback_analysis(content, sources)
    
    def _format_sources_for_llm(self, sources: List[Dict[str, Any]]) -> str:
        if not sources:
            return "No reliable sources found."
        
        formatted_sources = []
        for i, source in enumerate(sources[:8], 1):
            source_info = f"""
            Source {i}:
            Title: {source.get('title', 'Unknown')}
            Domain: {source.get('domain', 'unknown')}
            Type: {source.get('source_type', 'unknown')}
            Credibility Score: {source.get('credibility_score', 0):.2f}
            Content: {source.get('snippet', 'No content available')}
            Indicators: Misinformation signals: {source.get('content_indicators', {}).get('misinformation_signals', 0)}, Verification signals: {source.get('content_indicators', {}).get('verification_signals', 0)}
            """
            formatted_sources.append(source_info.strip())
            formatted_sources.append(source_info)
        
        return "\n".join(formatted_sources)
    
    def _validate_and_enhance_result(self, result: Dict[str, Any], content: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        required_fields = ["verdict", "confidence", "summary", "key_findings"]
        for field in required_fields:
            if field not in result:
                return self._fallback_analysis(content, sources)
        
        if result["verdict"] not in ["verified", "false", "misleading", "mixed", "insufficient_evidence"]:
            result["verdict"] = "insufficient_evidence"
        
        result["confidence"] = max(0, min(100, int(result.get("confidence", 50))))
        
        result["source_breakdown"] = {
            "fact_check_sites": len([s for s in sources if s.get('source_type') == 'fact_check']),
            "news_sources": len([s for s in sources if s.get('source_type') == 'news']),
            "academic_sources": len([s for s in sources if s.get('source_type') == 'academic']),
            "total_sources": len(sources)
        }
        
        return result
    
    def _parse_unstructured_response(self, response: str, content: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        verdict_patterns = {
            "verified": r"(?:verified|true|accurate|correct|confirmed)",
            "false": r"(?:false|fake|fabricated|debunked|untrue)",
            "misleading": r"(?:misleading|distorted|partial|biased)",
            "mixed": r"(?:mixed|partially|some truth)",
            "insufficient_evidence": r"(?:insufficient|unclear|unknown|cannot determine)"
        }
        
        verdict = "insufficient_evidence"
        confidence = 50
        
        response_lower = response.lower()
        for v, pattern in verdict_patterns.items():
            if re.search(pattern, response_lower):
                verdict = v
                break
        
        confidence_match = re.search(r"confidence[:\s]*(\d+)", response_lower)
        if confidence_match:
            confidence = int(confidence_match.group(1))
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "summary": response[:200] + "..." if len(response) > 200 else response,
            "key_findings": ["Analysis available in summary"],
            "evidence_for": [],
            "evidence_against": [],
            "credibility_assessment": {"source_quality": 5, "consistency": 5, "corroboration": 5},
            "risk_indicators": [],
            "context": "Limited analysis due to parsing issues",
            "recommendations": "Manual verification recommended"
        }
    
    def _fallback_analysis(self, content: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        avg_credibility = sum(s.get('credibility_score', 0) for s in sources) / len(sources) if sources else 0.3
        
        fact_check_sources = [s for s in sources if s.get('source_type') == 'fact_check']
        misinformation_signals = sum(s.get('content_indicators', {}).get('misinformation_signals', 0) for s in sources)
        verification_signals = sum(s.get('content_indicators', {}).get('verification_signals', 0) for s in sources)
        
        if fact_check_sources and misinformation_signals > verification_signals:
            verdict = "false"
            confidence = 75
        elif verification_signals > misinformation_signals and avg_credibility > 0.7:
            verdict = "verified"
            confidence = 70
        elif avg_credibility < 0.4:
            verdict = "insufficient_evidence"
            confidence = 30
        else:
            verdict = "mixed"
            confidence = 50
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "summary": f"Automated analysis based on {len(sources)} sources with average credibility {avg_credibility:.2f}",
            "key_findings": [f"Found {len(fact_check_sources)} fact-checking sources", f"Credibility score: {avg_credibility:.2f}"],
            "evidence_for": [f"Verification signals: {verification_signals}"],
            "evidence_against": [f"Misinformation signals: {misinformation_signals}"],
            "credibility_assessment": {"source_quality": int(avg_credibility * 10), "consistency": 5, "corroboration": len(sources)},
            "risk_indicators": ["Automated analysis - manual verification recommended"],
            "context": "Fallback analysis due to LLM processing issues",
            "recommendations": "Consider multiple sources and expert opinions"
        }

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
        <title>AI Fact-Checker Pro</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .container {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border-radius: 24px;
                box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
                padding: 40px;
                max-width: 800px;
                width: 100%;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
            }
            
            .header h1 {
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
            }
            
            .header p {
                color: #666;
                font-size: 1.1rem;
                font-weight: 400;
            }
            
            .form-group {
                margin-bottom: 25px;
            }
            
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #333;
                font-size: 0.95rem;
            }
            
            textarea {
                width: 100%;
                padding: 16px;
                border: 2px solid #e1e5e9;
                border-radius: 12px;
                font-size: 16px;
                font-family: inherit;
                resize: vertical;
                min-height: 120px;
                transition: all 0.3s ease;
                background: rgba(255, 255, 255, 0.8);
            }
            
            textarea:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            .file-input-wrapper {
                position: relative;
                display: inline-block;
                width: 100%;
            }
            
            .file-input {
                position: absolute;
                left: -9999px;
                opacity: 0;
            }
            
            .file-input-label {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 16px;
                border: 2px dashed #d1d5db;
                border-radius: 12px;
                cursor: pointer;
                transition: all 0.3s ease;
                background: rgba(255, 255, 255, 0.8);
                color: #666;
                font-weight: 500;
            }
            
            .file-input-label:hover {
                border-color: #667eea;
                background: rgba(102, 126, 234, 0.05);
                color: #667eea;
            }
            
            .file-input-label.has-file {
                border-color: #10b981;
                background: rgba(16, 185, 129, 0.05);
                color: #10b981;
            }
            
            .submit-btn {
                width: 100%;
                padding: 16px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            
            .submit-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            }
            
            .submit-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            
            .spinner {
                width: 40px;
                height: 40px;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .result {
                display: none;
                margin-top: 30px;
                padding: 25px;
                border-radius: 16px;
                border-left: 4px solid;
                background: rgba(255, 255, 255, 0.9);
                backdrop-filter: blur(10px);
            }
            
            .result.verified {
                border-left-color: #10b981;
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
            }
            
            .result.false {
                border-left-color: #ef4444;
                background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05));
            }
            
            .result.misleading {
                border-left-color: #f59e0b;
                background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05));
            }
            
            .result.mixed {
                border-left-color: #8b5cf6;
                background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(139, 92, 246, 0.05));
            }
            
            .result.insufficient_evidence {
                border-left-color: #6b7280;
                background: linear-gradient(135deg, rgba(107, 114, 128, 0.1), rgba(107, 114, 128, 0.05));
            }
            
            .verdict-header {
                display: flex;
                align-items: center;
                margin-bottom: 15px;
            }
            
            .verdict-badge {
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-right: 15px;
            }
            
            .verdict-badge.verified {
                background: #10b981;
                color: white;
            }
            
            .verdict-badge.false {
                background: #ef4444;
                color: white;
            }
            
            .verdict-badge.misleading {
                background: #f59e0b;
                color: white;
            }
            
            .verdict-badge.mixed {
                background: #8b5cf6;
                color: white;
            }
            
            .verdict-badge.insufficient_evidence {
                background: #6b7280;
                color: white;
            }
            
            .confidence-bar {
                flex: 1;
                height: 8px;
                background: rgba(0, 0, 0, 0.1);
                border-radius: 4px;
                overflow: hidden;
            }
            
            .confidence-fill {
                height: 100%;
                background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);
                transition: width 0.8s ease;
                border-radius: 4px;
            }
            
            .confidence-text {
                margin-left: 10px;
                font-weight: 600;
                color: #333;
            }
            
            .result-section {
                margin-bottom: 20px;
            }
            
            .result-section h3 {
                color: #333;
                margin-bottom: 10px;
                font-size: 1.1rem;
                font-weight: 600;
            }
            
            .result-section p {
                color: #666;
                line-height: 1.6;
            }
            
            .result-section ul {
                color: #666;
                line-height: 1.6;
                padding-left: 20px;
            }
            
            .sources {
                display: grid;
                gap: 15px;
                margin-top: 20px;
            }
            
            .source-item {
                padding: 15px;
                background: rgba(255, 255, 255, 0.7);
                border-radius: 10px;
                border: 1px solid rgba(0, 0, 0, 0.05);
            }
            
            .source-title {
                font-weight: 600;
                color: #333;
                margin-bottom: 5px;
            }
            
            .source-domain {
                font-size: 0.9rem;
                color: #667eea;
                margin-bottom: 8px;
            }
            
            .source-snippet {
                font-size: 0.9rem;
                color: #666;
                line-height: 1.5;
            }
            
            .source-credibility {
                display: inline-block;
                margin-top: 8px;
                padding: 2px 8px;
                background: rgba(102, 126, 234, 0.1);
                color: #667eea;
                border-radius: 12px;
                font-size: 0.8rem;
                font-weight: 500;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>AI Fact-Checker Pro</h1>
                <p>Advanced real-time fact-checking with AI-powered analysis</p>
            </div>
            
            <form id="factCheckForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="content">Enter content to fact-check:</label>
                    <textarea 
                        id="content" 
                        name="content" 
                        required 
                        placeholder="Paste the text, claim, or statement you want to verify..."
                    ></textarea>
                </div>
                
                <div class="form-group">
                    <label for="image">Upload image (optional):</label>
                    <div class="file-input-wrapper">
                        <input type="file" id="image" name="image" accept="image/*" class="file-input">
                        <label for="image" class="file-input-label" id="fileLabel">
                            📸 Choose image file or drag and drop
                        </label>
                    </div>
                </div>
                
                <button type="submit" class="submit-btn" id="submitBtn">
                    Analyze & Fact-Check
                </button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing content and searching for reliable sources...</p>
            </div>
            
            <div class="result" id="result"></div>
        </div>
        
        <script>
            const form = document.getElementById('factCheckForm');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const submitBtn = document.getElementById('submitBtn');
            const fileInput = document.getElementById('image');
            const fileLabel = document.getElementById('fileLabel');
            
            fileInput.addEventListener('change', function() {
                if (this.files && this.files[0]) {
                    fileLabel.textContent = `📸 ${this.files[0].name}`;
                    fileLabel.classList.add('has-file');
                } else {
                    fileLabel.textContent = '📸 Choose image file or drag and drop';
                    fileLabel.classList.remove('has-file');
                }
            });
            
            form.addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(form);
                
                submitBtn.disabled = true;
                loading.style.display = 'block';
                result.style.display = 'none';
                
                try {
                    const response = await fetch('/fact-check', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        displayResult(data);
                    } else {
                        displayError(data.detail || 'An error occurred');
                    }
                } catch (error) {
                    displayError('Network error occurred');
                } finally {
                    submitBtn.disabled = false;
                    loading.style.display = 'none';
                }
            });
            
            function displayResult(data) {
                const verdict = data.verdict;
                const confidence = data.confidence;
                
                result.className = `result ${verdict}`;
                result.innerHTML = `
                    <div class="verdict-header">
                        <span class="verdict-badge ${verdict}">${verdict.replace('_', ' ')}</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%"></div>
                        </div>
                        <span class="confidence-text">${confidence}%</span>
                    </div>
                    
                    <div class="result-section">
                        <h3>Summary</h3>
                        <p>${data.summary}</p>
                    </div>
                    
                    ${data.key_findings && data.key_findings.length > 0 ? `
                        <div class="result-section">
                            <h3>Key Findings</h3>
                            <ul>
                                ${data.key_findings.map(finding => `<li>${finding}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    
                    ${data.evidence_for && data.evidence_for.length > 0 ? `
                        <div class="result-section">
                            <h3>Supporting Evidence</h3>
                            <ul>
                                ${data.evidence_for.map(evidence => `<li>${evidence}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    
                    ${data.evidence_against && data.evidence_against.length > 0 ? `
                        <div class="result-section">
                            <h3>Contradicting Evidence</h3>
                            <ul>
                                ${data.evidence_against.map(evidence => `<li>${evidence}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    
                    ${data.context ? `
                        <div class="result-section">
                            <h3>Context</h3>
                            <p>${data.context}</p>
                        </div>
                    ` : ''}
                    
                    ${data.recommendations ? `
                        <div class="result-section">
                            <h3>Recommendations</h3>
                            <p>${data.recommendations}</p>
                        </div>
                    ` : ''}
                    
                    ${data.sources && data.sources.length > 0 ? `
                        <div class="result-section">
                            <h3>Sources Analyzed</h3>
                            <div class="sources">
                                ${data.sources.slice(0, 5).map(source => `
                                    <div class="source-item">
                                        <div class="source-title">${source.title}</div>
                                        <div class="source-domain">${source.domain}</div>
                                        <div class="source-snippet">${source.snippet}</div>
                                        <span class="source-credibility">Credibility: ${(source.credibility_score * 100).toFixed(0)}%</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}
                    
                    <div class="result-section">
                        <h3>Analysis Details</h3>
                        <p>
                            <strong>Processing Time:</strong> ${data.analysis_duration}s |
                            <strong>Sources Analyzed:</strong> ${data.search_metadata?.total_results || 0} |
                            <strong>Fact-Check Sources:</strong> ${data.search_metadata?.fact_check_sources || 0}
                        </p>
                    </div>
                `;
                
                result.style.display = 'block';
            }
            
            function displayError(message) {
                result.className = 'result false';
                result.innerHTML = `
                    <div class="verdict-header">
                        <span class="verdict-badge false">Error</span>
                    </div>
                    <div class="result-section">
                        <h3>Error</h3>
                        <p>${message}</p>
                    </div>
                `;
                result.style.display = 'block';
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
        
        result = await fact_checker.comprehensive_fact_check(content, image_obj)
        
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