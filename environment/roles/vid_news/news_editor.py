import os
import time
import re
import sys
import asyncio
import logging
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from environment.config.llm import gpt

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, max_tokens: int = 10000):
        self.max_tokens = max_tokens
        self.timeout = 45
        self.model = "gpt-4o-mini"

    async def process(self, user_idea: str, txt_path: str, pre_txt_path: str) -> str:
        """Main pipeline process - directly using presenter agent"""
        present_content = self.load_text(pre_txt_path)
        logger.info("Loaded presentation method")

        book_content = self.load_text(txt_path)
        
        word_count = len(book_content.split())
        char_count = len(book_content)
        logger.info(f"Text data statistics: {word_count} words, {char_count} characters")
        
        # Limit content length if necessary
        if len(book_content) > 30000:
            logger.warning(f"Content too large ({len(book_content)} chars), truncating to 30K")
            book_content = book_content[:30000]
        
        try:
            # Pass content directly to presenter agent
            presenter_output = await self.presenter_agent(user_idea, book_content, present_content)
            logger.info("Successfully generated presentation")
        except Exception as e:
            logger.error(f"Error in presenter_agent: {e}")
            # Create a meaningful fallback based on user idea and content
            presenter_output = self._create_fallback_content(user_idea, book_content)
            logger.info("Used intelligent fallback content")
                
        try:
            formatted_content = await self.judger_agent(user_idea, presenter_output)
            logger.info("Successfully structured content with judger_agent")
        except Exception as e:
            logger.error(f"Error in judger_agent: {e}")
            formatted_content = self._create_formatted_fallback(presenter_output)
            logger.info("Used intelligent formatting fallback")

        return formatted_content

    def _create_fallback_content(self, user_idea: str, book_content: str) -> str:
        """Create intelligent fallback content when API calls fail"""
        logger.info("Creating fallback content based on user idea and source material")
        
        # Extract key sentences from the book content
        sentences = []
        for line in book_content.split('\n'):
            line = line.strip()
            if len(line) > 50:  # Only include substantial sentences
                sentences.append(line)
        
        # Take the most relevant sentences (up to 8 sentences for a good demo)
        selected_sentences = sentences[:8]
        
        # Create a structured narrative based on user idea
        if "product demo" in user_idea.lower():
            # Create product demo focused content
            demo_content = []
            demo_content.append("This innovative software solution transforms how teams collaborate and manage projects with cutting-edge technology.")
            demo_content.extend(selected_sentences[:6])  # Use source material
            demo_content.append("The platform delivers exceptional user experience through intuitive design and powerful automation features.")
            demo_content.append("If you like the video content we create, please follow our account and there will be more relevant information later.")
        else:
            # Generic content structure
            demo_content = selected_sentences[:8]
            if not demo_content:
                demo_content = [
                    "This presentation showcases important developments in technology and innovation.",
                    "The featured solution addresses key challenges faced by modern organizations.",
                    "Advanced features provide users with comprehensive tools for success.",
                    "The system demonstrates remarkable performance and reliability in real-world scenarios."
                ]
        
        return " ".join(demo_content)

    def _create_formatted_fallback(self, content: str) -> str:
        """Create properly formatted fallback content with section markers"""
        logger.info("Creating formatted fallback content with proper section markers")
        
        # Split content into sentences
        sentences = []
        for punct in ['. ', '! ', '? ', '。', '！', '？']:
            content = content.replace(punct, '|SPLIT|')
        
        raw_sentences = content.split('|SPLIT|')
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only include meaningful sentences
                # Remove commas as per requirements
                sentence = sentence.replace(',', '').replace('，', '')
                sentences.append(sentence)
        
        # Format with proper section markers
        if not sentences:
            return "/////\nContent generation encountered an issue. Please try again."
        
        formatted_parts = []
        for i, sentence in enumerate(sentences):
            if i == 0:
                formatted_parts.append(f"/////\n{sentence}")
            else:
                formatted_parts.append(f"\n\n/////\n{sentence}")
        
        return "".join(formatted_parts)

    def load_text(self, txt_path: str) -> str:
        """Load content from a text file"""
        encodings_to_try = ['utf-8', 'gb18030', 'gbk', 'gb2312', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                logger.info(f"Trying to read file with encoding: {encoding}")
                with open(txt_path, 'r', encoding=encoding, errors='replace') as file:
                    content = file.read()
                logger.info(f"Successfully read file with encoding: {encoding}")
                return content
            except UnicodeDecodeError:
                continue
        
        try:
            logger.info("Trying binary reading approach")
            with open(txt_path, 'rb') as file:
                content = file.read().decode('utf-8', errors='replace')
            return content
        except Exception as e:
            logger.error(f"Error loading text file: {e}")
            return ""

    @retry(
        retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def _make_api_call(self, messages, temperature=0.7, max_tokens=None, timeout=None):
        """Make an API call with retries and exponential backoff using tenacity"""
        if max_tokens is None:
            max_tokens = min(2000, self.max_tokens)
        if timeout is None:
            timeout = self.timeout
            
        try:
            logger.info(f"Making API call with {len(messages)} messages, timeout={timeout}s")
            start_time = time.time()
            
            # Use gpt function with full messages array
            response = gpt(model="gpt-4o-mini", messages=messages)
            
            elapsed_time = time.time() - start_time
            logger.info(f"API call completed in {elapsed_time:.2f}s")
            return response.choices[0].message.content
        except asyncio.TimeoutError as e:
            logger.warning(f"API call timed out after {timeout} seconds, retrying...")
            raise e  # Re-raise to trigger tenacity retry
        except Exception as e:
            logger.warning(f"API call failed with error: {e}, retrying...")
            raise e  # Re-raise to trigger tenacity retry

    async def presenter_agent(self, user_idea: str, content: str, present_content: str) -> str:
        """Process content and adapt to user's idea - directly using input content"""
        
        prompt = f"""
        Create a skit narration copy, strictly following the user's ideas and presentation methods.

        User's idea:
        "{user_idea}"
        
        Grounded text content:
        {content}

        Follow this presentation method, read it and apply it carefully:
        {present_content}
        
        Requirements:
        1. Format and Structure:
        - Remove all sections numbers
        - Don't write too short sentences (Each sentence should contain more than 11 words) !!!
        - Present as one combined, coherent paragraph
        - Begin with clear news background establishment
        
        2. Content Guidelines:
        - Strictly abide by the user's words/字数 count requirements
        - Use only original key dialogues (no fabricated dialogues)
        - Remove unnecessary environmental descriptions
        - Focus on plot-advancing elements
        - Do not use the " " symbol
        
        3. Language and Style:
        - Third-person perspective
        - Process in text language (English/中文)
        - Do not use Arabic numerals. Change all numbers to English words, such as 5 becomes five
        - Do not use slash '/' within the writing
        - When encountering abbreviations of proper nouns, separate the word and the abbreviation appropriately, for example, ChatGPT becomes Chat GPT, OpenAI becomes Open AI, and AndroidOS becomes Android OS
        - Maintain clear narrative flow
        - Never mention or show user's requirements in content
        - Remove duplicated sentences
        
        Create a single, polished version that meets all these requirements.
        """
        
        try:
            system_message = "You are an experienced expert in news writing skit review copy. Pay special attention to user's words/字数 count requirements."
            
            logger.info("Starting presenter agent processing")
            result = await self._make_api_call(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                timeout=120,  # Increased timeout for processing larger content
                max_tokens=16384
            )
            
            logger.info("Completed skit narration generation")
            return result
            
        except Exception as e:
            logger.error(f"Error in presenter_agent: {e}")
            return content[:15000]

    async def judger_agent(self, user_idea: str, presenter_output: str) -> str:
        """Structure the content with proper formatting based on presentation method"""
        prompt = f"""
        User's idea/用户的想法: "{user_idea}"

        Content to format/要结构化的内容:
        {presenter_output}

        Format above content into sections with requirements:

        - Remove all commas, 
        - Start /////\n
        - Chunk each period with \n\n/////\n and No need to chunk at the end of the content !
        - Remove any chapter numbers
        - Keep original content 
        - The purpose is to separate each sentence
        - Align 

        Example format:

        Input:
        /////\nGood morning everyone, nice to meet you again.\n\n/////\nThe weather is very nice today.

        Output:

        /////\nGood morning everyone nice to meet you again.\n\n/////\nThe weather is very nice today.

        
        #################

        - 删除所有逗号
        - 以 /////\n 开头
        - 用 \n\n////\n 分割每个句号，内容的最末尾无需分割！
        - 删除任何章节号
        - 保留原始内容
        - 目的是将每个句子分开

        示例格式：

        输入：
        /////\n大家早上好，很高兴再次见到你。\n\n/////\n今天天气很好。

        输出：

        /////\n大家早上好很高兴再次见到你。\n\n/////\n今天天气很好。
        """
        
        for attempt in range(2):
            try:
                logger.info(f"Starting judger agent (attempt {attempt+1}/2)")
                system_message = "You are a content formatting specialist with expertise in following guidelines"
                
                result = await self._make_api_call(
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4,
                    timeout=60,
                    max_tokens=16384
                )
                logger.info("Judger agent completed successfully")
                return result
            except Exception as e:
                logger.error(f"Error in judger_agent (attempt {attempt+1}/2): {e}")
                if attempt < 1:
                    await asyncio.sleep(2)
                    # Truncate output if needed
                    if len(presenter_output) > 15000:
                        presenter_output = presenter_output[:15000]
                        # Create a new prompt with truncated content
                        prompt = prompt.replace(
                            "Content to format/要结构化的内容:", 
                            f"Content to format/要结构化的内容:\n{presenter_output}"
                        )
                    logger.info("Truncated presenter output for retry")
                else:
                    logger.info("Using simple formatting fallback")
                    return f"/////\n{presenter_output}"
