from fastapi import UploadFile
import io
import PyPDF2

import os
from typing import Optional, Union
import google.generativeai as genai
from PIL import Image
import io
import requests
import logging
import json
import base64
from pdf2image import convert_from_bytes
import asyncio

from pathlib import Path
from dotenv import load_dotenv
import aiohttp

from enum import Enum

from modules.logging_config import configure_logging

logger = logging.getLogger(__name__)


def extract_text_sync(file_path: str) -> str:
    """
    Synchronous version of extract_text for backward compatibility.
    Only supports PyPDF2 for PDFs and direct reading for text files.
    
    Args:
        file_path: Path to the file to extract text from
        
    Returns:
        Extracted text as string
        
    Raises:
        Exception: If text extraction fails
    """
    try:
        configure_logging()
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() == '.pdf':
            # For PDFs, use PyPDF2
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        else:
            # For text files, read directly
            return file_path.read_text(encoding='utf-8', errors='ignore')
            
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        raise Exception(f"Failed to extract text from {file_path}: {str(e)}")


async def extract_text(file_path: str, use_ocr: bool = False) -> str:
    """
    Extract text from a file using appropriate method based on file type.
    
    Args:
        file_path: Path to the file to extract text from
        use_ocr: Whether to use OCR for PDF processing (default: False)
        
    Returns:
        Extracted text as string
        
    Raises:
        Exception: If text extraction fails
        
    Note:
        This function is async when use_ocr=True, sync otherwise
    """
    try:
        configure_logging()
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if use_ocr and file_path.suffix.lower() == '.pdf':
            # Use OCR for PDF processing
            try:
                ocr = GeminiOCR()
                with open(file_path, 'rb') as f:
                    pdf_bytes = io.BytesIO(f.read())
                    # Run OCR in the background
                    import asyncio
                    task = asyncio.create_task(ocr.process_pdf(pdf_bytes))
                    text = await task
                    return text
            except Exception as ocr_error:
                logger.warning(f"OCR failed for {file_path}, falling back to PyPDF2: {str(ocr_error)}")
                # Fall back to PyPDF2 if OCR fails
                pass
        
        if file_path.suffix.lower() == '.pdf':
            # For PDFs, use PyPDF2
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        else:
            # For text files, read directly
            return file_path.read_text(encoding='utf-8', errors='ignore')
            
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        raise Exception(f"Failed to extract text from {file_path}: {str(e)}")

class DocumentHandler: 
    async def extract_text_from_document(self, document: UploadFile):
        """
        Extract text from a document

        Args:
            document: The document to extract text from

        Returns:
            The text from the document
        """
        try:
            if document.content_type == "application/pdf":
                pdf_file = io.BytesIO(await document.read())
                try:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
                finally:
                    pdf_file.close()
            else:
                # Handle other file types (txt, md, etc.)
                content = await document.read()
                return content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Error extracting text from document {document.filename}: {str(e)}")
            raise Exception(f"Failed to extract text from {document.filename}: {str(e)}")


class GeminiOCR:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        '''
        Initialize OCR Helper with Vertex AI configuration.

        Args:
            api_key (str): API key for the model.
            model_name (str): Name of the model to use (currently only supports gemini models). Defaults to "gemini-2.0-flash".
        '''
        load_dotenv()
        configure_logging()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        logger.info("Using OCR")
            
        self.model = genai.GenerativeModel(model_name)

    def _convert_to_jpeg_sync(self, image: Image.Image) -> Image.Image:
        """Convert image to JPEG format with white background.
        
        Args:
            image (Image.Image): Input PIL Image
            
        Returns:
            Image.Image: Converted JPEG image
        """
        # Convert to RGB if needed
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            # Create a white background image
            background = Image.new('RGB', image.size, (255, 255, 255))
            # Paste the image on the background
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        return image

    async def _process_image_async(
            self, 
            image_input: Union[str, bytes, io.BytesIO, Image.Image], 
            prompt: Optional[str] = None, 
            page_number: Optional[int] = 0,
            retry: int = 3
        ) -> str:
        RATELIMIT = 500  # Refer to https://ai.google.dev/gemini-api/docs/rate-limits
        current_time = asyncio.get_event_loop().time()
        if not hasattr(self, '_last_call_time'):
            self._last_call_time = 0
        if not hasattr(self, '_call_count'):
            self._call_count = 0
        
        # Reset counter if a minute has passed
        if current_time - self._last_call_time >= 60:
            self._call_count = 0
            self._last_call_time = current_time
        
        # Check rate limit
        if self._call_count >= RATELIMIT:
            wait_time = 60 - (current_time - self._last_call_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self._call_count = 0
                self._last_call_time = asyncio.get_event_loop().time()
        
        self._call_count += 1
        """Process image using Vertex AI Gemini Vision model.
        
        Args:
            image_input (Union[str, bytes, io.BytesIO]): Input image as:
                - str: Path to local file or URL
                - bytes: Raw image bytes
                - BytesIO: Image bytes stream
                - Image.Image: PIL Image object
            prompt (str, optional): Custom prompt for the model. Defaults to basic OCR instruction.
                default prompt: "Please extract and return all text from this image. Format it properly and maintain the original structure. Return in markdown format without any other text and markdown tags."
            page_number (int, optional): Page number of the image. Defaults to 0.
            retry (int, optional): Number of times to retry the request if it fails. Defaults to 10.

        Returns:
            str: Extracted text from the image
            
        Raises:
            ValueError: If image input type is invalid
            Exception: If there's an error processing the image
        """
        image = None
        img_byte_arr = None
        
        try:
            if retry <= 0:
                raise Exception("Failed to process image after maximum retries")
                
            retry -= 1
            logger.debug(f"Processing image {page_number}: (retry {10 - retry} of {10})")
            
            # Handle different input types
            if isinstance(image_input, str):
                # Handle URL or file path
                if image_input.startswith(('http://', 'https://')):
                    # Use aiohttp for async HTTP requests, fallback to requests if not available
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(image_input) as response:
                                if response.status != 200:
                                    raise ValueError(f"Failed to fetch image from URL: {response.status}")
                                content = await response.read()
                                image = Image.open(io.BytesIO(content))
                    except ImportError:
                        # Fallback to synchronous requests if aiohttp not available
                        response = requests.get(image_input)
                        if response.status_code != 200:
                            raise ValueError(f"Failed to fetch image from URL: {response.status_code}")
                        image = Image.open(io.BytesIO(response.content))
                else:
                    if not os.path.exists(image_input):
                        raise ValueError(f"Image file not found: {image_input}")
                    # Use asyncio.to_thread for file I/O operations
                    image = await asyncio.to_thread(Image.open, image_input)
            elif isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, io.BytesIO):
                image = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                raise ValueError("Invalid image input type. Must be string path/URL, bytes, or BytesIO.")
            
            # Add image size check and resizing
            MAX_SIZE = (1654, 2340)  # Maximum dimensions
            if image.size[0] > MAX_SIZE[0] or image.size[1] > MAX_SIZE[1]:
                image.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
            
            # Convert to JPEG format
            image = self._convert_to_jpeg_sync(image)
                
            # Convert image to bytes and encode as base64 with lower quality
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=85)
            img_bytes = img_byte_arr.getvalue()
            
            # Default prompt if none provided
            if not prompt:
                prompt = "Anda akan diberikan sebuah gambar halaman Undang-Undang Negara Republik Indonesia. Tugas anda adalah menuliskan kembali isi (tanpa header) dari halaman tersebut tanpa mengubah kata-kata yang ada."

            # Create the image data for Gemini
            image_data = {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_bytes).decode('utf-8')
            }
            
            # Generate response from Gemini - use asyncio.to_thread for the synchronous API call
            response = await asyncio.to_thread(
                self.model.generate_content,
                contents=[prompt, image_data],
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 2048,
                }
            )

            if not response.text:
                error_msg = {
                    "error": "No text in response",
                    "response_data": str(response)
                }
                raise Exception(f"Model returned no text. Details: {json.dumps(error_msg, indent=2)}")
                
            return response.text
            
        except Exception as e:
            logger.error(f"Error processing image with Gemini: {str(e)}")
            if retry > 0:
                return await self._process_image_async(image_input, prompt, page_number, retry)
            raise
        finally:
            # Clean up resources
            if img_byte_arr:
                img_byte_arr.close()
            # Note: Don't close image if it's from external input to avoid closing user's image

    async def process_pdf(
            self, 
            pdf_file: io.BytesIO, 
            prompt: Optional[str] = None,
            batch_size: int = 8,
        ) -> str:
        """
        Process a PDF file using OCR with ThreadPoolExecutor.

        Args:
            pdf_file: PDF file to process
            prompt: Prompt to use for the OCR
            batch_size: Number of images to process concurrently (default: 8) - deprecated, using ThreadPoolExecutor

        Returns:
            str: Extracted text from the PDF (concatenated from all pages)

        Raises:
            Exception: If there's an error during PDF processing, with detailed error message
        """
        try:
            # Convert PDF to images with optimized settings
            logger.debug("Converting PDF to images")
            

            # Convert PDF to images with optimized settings
            logger.debug("Converting PDF to images")
            
            # Get the bytes from BytesIO object
            pdf_bytes = pdf_file.getvalue()
            
            images = await asyncio.to_thread(
                convert_from_bytes,
                pdf_bytes,
                dpi=200,  # Lower DPI - still readable but uses less memory
                fmt='jpeg',
                size=(1654, 2340)  # Limit max dimensions (A4 at 200 DPI)
            )
            
            # Process images concurrently using asyncio.gather
            tasks = []
            for i, image in enumerate(images):
                task = self._process_image_async(image, prompt, page_number=i)
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            text_parts = [""] * len(images)  # Pre-allocate list
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing image {i + 1}: {str(result)}")
                    text_parts[i] = f"Error processing image {i + 1}: {str(result)}"
                else:
                    text_parts[i] = result
                
            # Clean up images after all processing is complete
            if 'images' in locals():
                for img in images:
                    try:
                        img.close()
                    except Exception as e:
                        logger.debug(f"Error closing image: {e}")

            # Concatenate all text parts into a single string
            full_text = "\n\n".join(text_parts)
            return full_text
            
        except Exception as e:
            error_msg = f"Error processing PDF: {str(e)}"
            logger.error(error_msg)
            
            # Clean up images
            if 'images' in locals():
                for img in images:
                    img.close()

            raise Exception(error_msg)
        finally:
            # Ensure PDF file is properly handled
            if hasattr(pdf_file, 'seek'):
                pdf_file.seek(0)  # Reset position for potential reuse


class ChunkingMethod(Enum):
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"

class Chunks:
    def __init__(self, chunking_method: ChunkingMethod = ChunkingMethod.RECURSIVE):
        configure_logging()
        self.chunking_method = chunking_method
        
    def clean_text(self, text: str) -> str:
        """
        Clean text by normalizing whitespace and removing excessive newlines.

        Args:
            text: The text to clean

        Returns:
            Cleaned text as a string
        """
        # Replace multiple spaces/tabs with a single space, strip leading/trailing whitespace,
        # and collapse multiple newlines to a single newline.
        import re
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Chunk text based on the chunking method

        Args:
            text: The text to chunk
            chunk_size: The size of each chunk
            chunk_overlap: The overlap between chunks   

        Returns:
            The list of chunks

        Raises:
            ValueError: If the chunking method is invalid
        """
        # validate chunking method
        if self.chunking_method not in [ChunkingMethod.RECURSIVE, ChunkingMethod.SENTENCE, ChunkingMethod.PARAGRAPH]:
            raise ValueError("Invalid chunking method")
        
        # chunk text
        if self.chunking_method == ChunkingMethod.RECURSIVE:
            chunks = self._chunk_text_recursive(text, chunk_size, chunk_overlap)
        elif self.chunking_method == ChunkingMethod.SENTENCE:
            chunks = self._chunk_text_sentence(text, chunk_size, chunk_overlap)
        elif self.chunking_method == ChunkingMethod.PARAGRAPH:
            chunks = self._chunk_text_paragraph(text, chunk_size, chunk_overlap)
        else:
            raise ValueError("Invalid chunking method")

        return [self.clean_text(chunk) for chunk in chunks]

    def _chunk_text_recursive(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """ 
        Chunk text recursively by splitting on spaces and joining chunks back together
        while respecting word boundaries and maintaining overlap between chunks.
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Number of characters to overlap between chunks
            
        Returns:
            list[str]: List of text chunks
        """
        # Handle empty or small texts
        if not text or len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Get the chunk end position
            end = start + chunk_size
            
            # If we're at the end of text, add the final piece
            if end >= text_length:
                chunks.append(text[start:])
                break
                
            # Find the last space within the chunk to avoid splitting words
            last_space = text.rfind(' ', start, end)
            
            if last_space == -1:  # No space found, force split at chunk_size
                chunks.append(text[start:end])
                start = end
            else:
                # Split at the last space
                chunks.append(text[start:last_space])
                start = last_space + 1 - chunk_overlap  # Move back by overlap amount
                
            # Ensure we don't go backwards in the text
            start = max(0, start)

        return chunks

    def _chunk_text_sentence(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Chunk text by splitting on sentence boundaries and combining sentences into chunks
        while maintaining the specified chunk size and overlap.
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Number of characters to overlap between chunks
            
        Returns:
            list[str]: List of text chunks
        """
        import re
        configure_logging()

        # Handle empty or small texts
        if not text or len(text) <= chunk_size:
            return [text]

        # Split text into sentences using regex
        # This pattern matches sentence endings with period, question mark, or exclamation mark
        # followed by spaces or newlines
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If a single sentence is longer than chunk_size, use recursive chunking
            if sentence_length > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                # Recursively chunk the long sentence
                sentence_chunks = self._chunk_text_recursive(sentence, chunk_size, chunk_overlap)
                chunks.extend(sentence_chunks)
                continue
                
            # If adding this sentence would exceed chunk_size
            if current_length + sentence_length + 1 > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Keep some sentences for overlap
                    overlap_size = 0
                    overlap_chunk = []
                    for s in reversed(current_chunk):
                        if overlap_size + len(s) + 1 <= chunk_overlap:
                            overlap_chunk.insert(0, s)
                            overlap_size += len(s) + 1
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_length = overlap_size
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
            
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def _chunk_text_paragraph(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Chunk text by splitting on paragraph boundaries and combining paragraphs into chunks
        while maintaining the specified chunk size and overlap.
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Number of characters to overlap between chunks
            
        Returns:
            list[str]: List of text chunks
        """
        # Handle empty or small texts
        if not text or len(text) <= chunk_size:
            return [text]

        # Split text into paragraphs using double newlines
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            # If a single paragraph is longer than chunk_size, use sentence chunking
            if paragraph_length > chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                # Use sentence chunking for long paragraphs
                paragraph_chunks = self._chunk_text_sentence(paragraph, chunk_size, chunk_overlap)
                chunks.extend(paragraph_chunks)
                continue
                
            # If adding this paragraph would exceed chunk_size
            if current_length + paragraph_length + 2 > chunk_size:  # +2 for '\n\n'
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    # Keep some paragraphs for overlap
                    overlap_size = 0
                    overlap_chunk = []
                    for p in reversed(current_chunk):
                        if overlap_size + len(p) + 2 <= chunk_overlap:
                            overlap_chunk.insert(0, p)
                            overlap_size += len(p) + 2
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_length = overlap_size
            
            current_chunk.append(paragraph)
            current_length += paragraph_length + 2  # +2 for '\n\n'
            
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks
    
