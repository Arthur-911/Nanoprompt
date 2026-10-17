"""
Prompt Compression Engine.
Provides:
1. FastHeuristicCompressor: 0ms offline compression via regex & NLP rules (zero latency, zero API cost).
2. AICompressor: LLM-powered deep semantic prompt restructuring (via Google GenAI or compatible API).
"""
import re
import os
from typing import Optional, Tuple

class FastHeuristicCompressor:
    """
    Sub-millisecond offline prompt compressor.
    Strips conversational pleasantries, passive voice, meta-fluff,
    and converts requests into dense imperative directives.
    """

    # Longer phrases MUST precede shorter substrings in regex alternations
    TRANSFORMATIONS = [
        # Verb phrase contractions & imperative conversions
        (r'\bdo\s+a\s+(pdf\s+)?scan(\s+(for|of)\s+(me|us))?\b', r'Scan \1'),
        (r'\bdo\s+an?\s+analysis\s+of(\s+(for|to)\s+(me|us))?\b', 'Analyze'),
        (r'\bdo\s+a\s+summary\s+of(\s+(for|to)\s+(me|us))?\b', 'Summarize'),
        (r'\bdo\s+a\s+search\s+for\b', 'Search for'),
        (r'\bgive\s+me\s+an?\s+explanation\s+of\b', 'Explain'),
        (r'\bgive\s+me\s+a\s+list\s+of\b', 'List'),
        (r'\btell\s+me\s+what\s+is\s+inside\s+(it|this|that)\b', 'extract its contents'),
        (r'\btell\s+me\s+what\s+is\s+inside\b', 'Extract contents of'),
        (r'\btake\s+a\s+look\s+at\b', 'Inspect'),
        (r'\bcome\s+up\s+with\b', 'Generate'),
        (r'\bfigure\s+out\b', 'Determine'),
        (r'\bmake\s+sure\s+(that\s+)?\b', 'Ensure '),
        (r'\bwrite\s+(some\s+)?code\s+(to|for)\b', 'Implement'),
        (r'\bfind\s+out\s+(if|whether)\b', 'Verify \1'),
    ]

    FILLER_PATTERNS = [
        # Long polite signoffs / greetings first
        r'\bthanks\s+in\s+advance\b[,!.]*',
        r'\bi\s+would\s+appreciate\s+it\b[,!.]*',
        r'\bat\s+your\s+earliest\s+convenience\b[,!.]*',
        r'\bas\s+soon\s+as\s+possible\b[,!.]*',
        r'\bthank\s+you(\s+so\s+much)?\b[,!.]*',
        r'\bmany\s+thanks\b[,!.]*',
        r'\bbest\s+regards\b[,!.]*',
        r'\b(thanks|cheers)\b[,!.]*',
        r'\bgood\s+(morning|afternoon|evening|day)\b[,!.]*',
        r'\b(hello|hi|hey)\b[,!.]*',
        
        # Polite hedges & conversational fluff
        r'\bcould\s+you\s+(please\s+)?(possibly\s+)?\b',
        r'\bwould\s+you\s+(mind\s+)?(please\s+)?\b',
        r'\bcan\s+you\s+(please\s+)?\b',
        r'\bi\s+was\s+wondering\s+if\s+you\s+could\b',
        r'\bi\s+(just\s+)?(want|need|would\s+like)\s+(you\s+to|to)\b',
        r'\bi\s+(am\s+)?looking\s+(for|to)\b',
        r'\bdo\s+me\s+a\s+favor\s+and\b',
        r'\bhelp\s+me\s+(to\s+)?\b',
        r'\b(please|kindly|just|simply)\b',
        
        # Verbal fillers
        r'\b(basically|actually|honestly|literally|pretty\s+much|sort\s+of|kind\s+of|in\s+my\s+opinion)\b',
        
        # Redundant pronoun targets & time meta
        r'\b(for\s+me|for\s+us|to\s+me|to\s+us)\b',
        r'\basap\b',
    ]

    @classmethod
    def compress(cls, text: str) -> str:
        if not text or not text.strip():
            return ""

        result = text.strip()

        # Step 1: Apply transformations to condense common verbose patterns
        for pattern, replacement in cls.TRANSFORMATIONS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Step 2: Strip polite filler & conversational hedges
        for pattern in cls.FILLER_PATTERNS:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        # Step 3: Clean up redundant punctuation (e.g. multiple commas, question marks)
        result = re.sub(r'[,;]\s*[,;]+', ',', result)
        result = re.sub(r'^[,\s;.]+', '', result)  # Leading punctuation
        result = re.sub(r'\?+', '?', result)
        result = re.sub(r'!+', '!', result)

        # Step 4: Fix spaces before punctuation (e.g. "article ?" -> "article?")
        result = re.sub(r'\s+([?.!,])', r'\1', result)

        # Step 5: Clean up whitespace and empty lines
        result = re.sub(r'[ \t]+', ' ', result)
        result = re.sub(r'\n\s*\n', '\n', result)
        result = result.strip()

        # Step 6: Capitalize first letter of output
        if result:
            result = result[0].upper() + result[1:]

        # Uppercase common acronyms if lowercase (pdf -> PDF, api -> API, sql -> SQL)
        result = re.sub(r'\b(pdf|api|sql|json|csv|html|css|url|ui|ux)\b', lambda m: m.group(1).upper(), result, flags=re.IGNORECASE)

        return result


class AICompressor:
    """
    LLM-powered prompt compressor using Google GenAI SDK.
    Takes complex, multi-sentence inputs and reformulates them into
    token-minimized, unambiguous technical prompts.
    """

    SYSTEM_INSTRUCTION = (
        "You are an expert prompt compression engine. Your mission is to convert wordy, conversational human instructions "
        "into the absolute most token-efficient, concise, and high-impact AI prompt possible.\n"
        "Rules:\n"
        "1. Remove ALL conversational filler, polite greetings, hedges, and redundant words.\n"
        "2. Convert passive sentences into direct imperative commands.\n"
        "3. Preserve ALL core constraints, variables, entities, code names, and intended outputs.\n"
        "4. Use bullet points or shorthand formatting only if it reduces token count.\n"
        "5. Output ONLY the compressed prompt. Do NOT wrap in quotes. Do NOT add preamble or explanations."
    )

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    def is_available(self) -> bool:
        return self.client is not None

    def compress(self, text: str, model: str = "gemini-2.5-flash") -> Tuple[str, Optional[str]]:
        """
        Compresses text using Gemini.
        Returns (compressed_text, error_message).
        """
        if not self.client:
            return "", "AI API key not configured or client initialization failed."

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=f"Compress this user request into a minimal token prompt:\n\n{text}",
                config={
                    "system_instruction": self.SYSTEM_INSTRUCTION,
                    "temperature": 0.1,  # Low temperature for deterministic, concise output
                }
            )
            compressed = response.text.strip() if response.text else ""
            return compressed, None
        except Exception as e:
            return "", str(e)
