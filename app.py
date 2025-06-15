import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
from dotenv import load_dotenv
from gtts import gTTS
import re
import os
import io
import time
from openai import OpenAI

# --- 1. ConfigManager Class ---
class ConfigManager:
    """
    Manages environment variables and initializes API clients.
    Adheres to Single Responsibility Principle (SRP) for configuration.
    """
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.serpapi_api_key = os.getenv("SERPAPI_API_KEY")
        
        # Initialize client instances to None, to be set later
        self.llm = None
        self.search_tool = None
        self.openai_client = None
        
        self._initialize_api_clients()

    def _initialize_api_clients(self):
        """Initializes LangChain LLM, OpenAI Whisper client, and SerpAPI tool."""
        if self.openai_api_key:
            try:
                # Initialize ChatOpenAI for text generation
                self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, openai_api_key=self.openai_api_key)
                # Initialize raw OpenAI client for Whisper (speech-to-text)
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            except Exception as e:
                st.error(f"Error initializing LLM or OpenAI client. Details: {e}")
        else:
            st.warning("OPENAI_API_KEY not found in environment variables. LLM and Whisper features will be disabled.")

        if self.serpapi_api_key:
            # Initialize SerpAPI for internet search, if key is available
            self.search_tool = Tool(
                name="Search", 
                func=SerpAPIWrapper(serpapi_api_key=self.serpapi_api_key).run,
                description="useful for when you need to answer questions about current events or things you don't have in your knowledge base"
            )
        else:
            st.warning("SERPAPI_API_KEY not found. Internet search for quiz generation will be disabled.")

    def get_llm(self):
        """Returns the initialized LangChain LLM instance."""
        return self.llm

    def get_search_tool(self):
        """Returns the initialized SerpAPI search tool instance."""
        return self.search_tool

    def get_openai_client(self):
        """Returns the initialized raw OpenAI client instance."""
        return self.openai_client

    def is_llm_available(self) -> bool:
        """Checks if the LLM is successfully initialized."""
        return self.llm is not None

    def is_openai_client_available(self) -> bool:
        """Checks if the OpenAI Whisper client is successfully initialized."""
        return self.openai_client is not None

# --- 2. AudioManager Class ---
class AudioManager:
    """
    Handles text-to-speech (TTS) and speech-to-text (STT) functionalities.
    Adheres to SRP for audio processing.
    """
    def __init__(self, openai_client: OpenAI = None):
        # Dependencies are injected (OpenAI client for Whisper)
        self.openai_client = openai_client

    def text_to_speech(self, text: str, lang: str = 'en') -> io.BytesIO | None:
        """
        Converts text to speech using gTTS and returns a BytesIO object.
        Args:
            text (str): The text to convert to speech.
            lang (str): The language for speech generation (e.g., 'en' for English).
        Returns:
            io.BytesIO | None: A BytesIO object containing the MP3 audio data, or None if an error occurs.
        """
        fp = io.BytesIO()
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.write_to_fp(fp)
            fp.seek(0) # Rewind the buffer to the beginning
        except Exception as e:
            st.error(f"Error generating speech: {e}")
            return None
        return fp

    def transcribe_audio_with_whisper(self, audio_bytes: bytes) -> str | None:
        """
        Transcribes audio bytes using OpenAI's Whisper API.
        Args:
            audio_bytes (bytes): Raw audio data in bytes.
        Returns:
            str | None: The transcribed text, or None if transcription fails.
        """
        if not self.openai_client:
            st.error("OpenAI client not initialized. Cannot transcribe audio.")
            return None

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.mp3" # Whisper API expects a file-like object with a name

        try:
            with st.spinner("Transcribing audio with Whisper..."):
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return transcript
        except Exception as e:
            st.error(f"Error transcribing audio with Whisper: {e}")
            return None

# --- 3. QuizContentProcessor Class ---
class QuizContentProcessor:
    """
    Processes different types of content (e.g., PDF) for quiz generation.
    Adheres to SRP for content processing.
    """
    def process_pdf(self, uploaded_file: io.BytesIO) -> str | None:
        """
        Processes an uploaded PDF file, extracts its text content, and returns it.
        Args:
            uploaded_file (io.BytesIO): The uploaded PDF file object.
        Returns:
            str | None: The extracted text content, or None if processing fails.
        """
        if not uploaded_file:
            return None

        with st.spinner("Processing PDF..."):
            try:
                # Save the uploaded file temporarily to disk for PyPDFLoader
                temp_file_path = f"temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load, split, and extract text from the PDF
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
                chunks = text_splitter.split_documents(docs)
                content = " ".join([chunk.page_content for chunk in chunks])
                
                os.remove(temp_file_path) # Clean up the temporary file
                st.success("PDF processed!")
                return content
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                return None

# --- 4. QuestionGenerator Class ---
class QuestionGenerator:
    """
    Responsible for generating quiz questions using the LLM and parsing its output.
    Encapsulates LLM interaction logic.
    """
    def __init__(self, llm: ChatOpenAI, search_tool: Tool = None):
        self.llm = llm
        self.search_tool = search_tool

    # Define DIFFICULTY_GUIDELINES as a class attribute
    DIFFICULTY_GUIDELINES = {
        "Easy": "Questions should test direct recall of information. Use simple vocabulary. The correct answer should be obvious from direct textual evidence.",
        "Medium": "Questions may require basic comprehension or simple inference. Options should be plausible distractors, requiring careful reading. Use moderate vocabulary.",
        "Hard": "Questions should require deeper analysis, synthesis of information, or complex inference. Options might be very close, requiring subtle distinctions or understanding of implications. Use advanced vocabulary and concepts."
    }

    def _parse_llm_response(self, raw_text: str) -> list[dict]:
        """
        Parses the raw text response from the LLM into a list of question dictionaries.
        Includes robust regex matching for question structure.
        """
        questions = []
        if not raw_text or not isinstance(raw_text, str):
            if not isinstance(raw_text, str) and raw_text is not None:
                st.warning(f"LLM response was not a string: {type(raw_text)}. Cannot parse.")
            return questions

        # Split the raw text into individual question blocks
        question_blocks = re.split(r'(?i)\n*(?:Question|Q)\s*\d*:\s*', raw_text.strip())
        
        for block_content in question_blocks:
            if not block_content.strip():
                continue

            current_question = {}
            
            # Regex to find the question text
            q_match = re.match(r'(.*?)\n(?=[A-D][\)\.:]\s*|Correct Option:|Explanation:)', block_content, re.DOTALL | re.IGNORECASE)
            if q_match:
                current_question["question"] = q_match.group(1).strip()
                remaining_block = block_content[len(q_match.group(1)):].strip() 
            else:
                # Fallback regex if options/correct option not immediately after question
                q_match_fallback = re.match(r'(.*?)(?=\nCorrect Option:|\nExplanation:|$)', block_content, re.DOTALL | re.IGNORECASE)
                if q_match_fallback:
                    current_question["question"] = q_match_fallback.group(1).strip()
                    remaining_block = block_content[len(q_match_fallback.group(1)):].strip()
                else:
                    st.warning(f"Could not find question text for a block. Skipping. Block content: {block_content[:100]}...")
                    continue

            # Extract options (A, B, C, D)
            options = {}
            option_matches = re.findall(
                r'([A-Da-d])(?:[\)\.:]|\s*):\s*(.*?)(?=\n\s*(?:[A-Da-d][\)\.:]|\s*Correct Option:|\s*Explanation:)|$)',
                remaining_block, re.IGNORECASE | re.DOTALL
            )
            for opt_letter, opt_text in option_matches:
                options[opt_letter.upper()] = opt_text.strip()
            
            current_question["options"] = options
            
            # Extract correct option letter
            correct_match = re.search(r'Correct Option:\s*([A-Da-d])', block_content, re.IGNORECASE)
            if correct_match:
                current_question["correct_option"] = correct_match.group(1).upper()
            else:
                st.warning(f"Could not find Correct Option for: {current_question.get('question', 'N/A')}. Skipping.")
                continue

            # Extract explanation
            explanation_match = re.search(r'Explanation:\s*(.*?)(?=\n*(?:Question:|Q:|I cannot generate questions on this topic|Agent stopped|This query is outside my current knowledge base|$))', block_content, re.DOTALL | re.IGNORECASE)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
                # Clean up any leftover 'Question:' headers that might be in the explanation
                explanation = re.sub(r'(?i)\n*Question:\s*.*', '', explanation).strip()
                current_question["explanation"] = explanation
            else:
                st.warning(f"Could not find Explanation for: {current_question.get('question', 'N/A')}. Skipping.")
                continue

            # Validate the parsed question before adding
            if all(key in current_question for key in ["question", "options", "correct_option", "explanation"]) \
               and len(current_question["options"]) >= 2 \
               and current_question["correct_option"] in current_question["options"]:
                questions.append(current_question)
            else:
                st.warning(f"Skipping malformed question block due to missing key or invalid options: {block_content[:200]}...")

        return questions

    def generate_mcqs(self, content: str, num_questions: int, context_source_type: str, difficulty: str) -> list[dict] | None:
        """
        Generates multiple-choice questions using the configured LLM.
        Args:
            content (str): The topic or document content for question generation.
            num_questions (int): The number of questions to generate.
            context_source_type (str): "topic" or "document".
            difficulty (str): The desired difficulty level (e.g., "Easy", "Medium", "Hard").
        Returns:
            list[dict] | None: A list of parsed question dictionaries, or None if generation fails.
        """
        if not self.llm:
            st.error("LLM not initialized. Cannot generate questions.")
            return None

        # Access DIFFICULTY_GUIDELINES as a class attribute
        difficulty_instruction = f"Generate questions. {self.DIFFICULTY_GUIDELINES[difficulty]}"


        if context_source_type == "topic" and self.search_tool:
            # Use LangChain Agent for topic-based questions with search capabilities
            task_description_for_agent = (
                f"You are an expert quiz question generator. Your task is to generate exactly {num_questions} multiple-choice question(s) on the topic: \"{content}\".\n"
                f"{difficulty_instruction}\n"
                "If the topic involves future, speculative, or highly current events not yet widely published, use your 'Search' tool to find accurate and verifiable information. "
                "After using any necessary tools, or if no tools are needed, provide your final response containing the quiz questions.\n"
                "If verifiable information for future or speculative events is genuinely unavailable even after searching, your final response should state clearly (after 'Final Answer:') that you cannot generate questions on that specific aspect due to lack of confirmed data.\n"
                "For each question, provide 4 options (A, B, C, D), indicate the single correct option, and give a brief explanation.\n\n"
                "When you have gathered all necessary information and are ready to provide the quiz, format your response as follows:\n"
                "Final Answer: [Your complete quiz content or statement about inability to generate questions starts here, immediately after 'Final Answer: ']\n"
                "Question: [The question text]\n"
                "A: [Option A text]\n"
                "B: [Option B text]\n"
                "C: [Option C text]\n"
                "D: [Option D text]\n"
                "Correct Option: [Single letter A, B, C, or D]\n"
                "Explanation: [Brief explanation for the correct answer]\n\n" 
                "# Repeat for other questions, ensuring a blank line separates each complete question block.\n"
                "\n" 
                "**Strictly ensure the quiz content (or your statement of inability) directly follows the 'Final Answer: ' prefix.**\n"
                "Do not include any other text before 'Final Answer:' in your final turn. "
                "Do not use the 'Question:' format in your thought process, only in the 'Final Answer:' block."
            )

            agent_llm_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert quiz question generator. You have access to a 'Search' tool for current events or information not in your knowledge base. Strictly follow the user's instructions regarding the number of questions and the output format for the final answer. Your final response should be only the quiz questions or a statement of inability, formatted as requested by the user."),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            agent = create_openai_tools_agent(self.llm, [self.search_tool], agent_llm_prompt)
            agent_executor = AgentExecutor(agent=agent, tools=[self.search_tool], verbose=True, max_iterations=7, handle_parsing_errors=True)

            try:
                response_dict = agent_executor.invoke({"input": task_description_for_agent})
                raw_mcq_text = response_dict.get("output", "")
                
                quiz_content_match = re.search(r'(?i)(?:Question|Q):\s*.*', raw_mcq_text, re.DOTALL)
                
                if quiz_content_match:
                    return self._parse_llm_response(quiz_content_match.group(0))
                elif "Agent stopped due to iteration limit or time limit" in raw_mcq_text or \
                     "I genuinely do not know how to proceed" in raw_mcq_text or \
                     "I cannot generate questions on this topic" in raw_mcq_text:
                    st.error("The AI agent encountered an issue and could not generate questions based on the topic. This might be due to lack of verifiable information, a very complex/ambiguous query, or an internal agent error that it couldn't resolve.")
                    st.text_area("Agent's internal (failure) output:", raw_mcq_text, height=150)
                    return None
                else:
                    st.error("The AI agent generated an unexpected response format. Could not extract questions.")
                    st.text_area("Agent's raw output (unexpected format):", raw_mcq_text, height=150)
                    return None

            except Exception as e:
                st.error(f"Error during agent execution: {e}")
                st.text_area("Agent execution error details:", str(e), height=150)
                # Attempt to parse even from an exception string if it contains question data
                quiz_content_match_from_exception = re.search(r'(?i)(?:Question|Q):\s*.*', str(e), re.DOTALL)
                if quiz_content_match_from_exception:
                    st.warning("Attempting to parse questions found within the agent's error message. This might indicate an underlying agent issue, but questions were recoverable.")
                    return self._parse_llm_response(quiz_content_match_from_exception.group(0))
                return None
        else:
            # Direct LLM call for document-based or topic-based without search
            if context_source_type == "document":
                instruction_text = (
                    f"Generate exactly {num_questions} multiple-choice question(s) based *only* on the following document content. "
                    f"{difficulty_instruction}\n"
                    "Ensure all information, including options and the correct answer, is directly verifiable from this text. "
                    "If the document is too short or lacks sufficient detail for the requested number of questions, generate as many as possible."
                )
            else: # Topic-based without search tool
                st.warning("Search tool not available. Generating questions based on LLM's existing knowledge, which may be limited for very recent or future events.")
                instruction_text = (
                    f"Generate exactly {num_questions} multiple-choice question(s) on the topic: \"{content}\". "
                    f"{difficulty_instruction}\n"
                    "Prioritize information that is well-established and widely known. "
                    "If the topic pertains to future events or very recent information that might be beyond your knowledge cutoff, "
                    "and you cannot provide a factually accurate question with verifiable options and a correct answer, "
                    "state clearly that you cannot generate questions on that specific aspect due to knowledge limitations or lack of confirmed data."
                )

            prompt_template_str = f"""
            You are a helpful quiz generation assistant.
            {instruction_text}
            
            **Output Format (Strictly follow this for EACH question block, no extra text or markdown):**
            Question: [The question text]
            A: [Option A text]\nB: [Option B text]\nC: [Option C text]\nD: [Option D text]
            Correct Option: [Single letter A, B, C, or D]
            Explanation: [Brief explanation for the correct answer]
            
            Ensure a blank line separates each complete question block if multiple questions are generated.
            Your final output should ONLY contain the formatted questions. Do not include any introductory or concluding sentences outside this format.
            
            ---
            Content/Topic: {content}
            Number of questions: {num_questions}
            """
            prompt = ChatPromptTemplate.from_template(prompt_template_str)
            chain = prompt | self.llm
            try:
                response = chain.invoke({"content": content, "num_questions": num_questions})
                return self._parse_llm_response(response.content)
            except Exception as e:
                st.error(f"Error calling LLM directly: {e}")
                return None

    def match_voice_answer_to_option(self, transcribed_text: str, q_data: dict) -> str | None:
        """
        Attempts to match a transcribed voice answer to one of the quiz options using LLM first,
        then falling back to regex/text matching.
        Args:
            transcribed_text (str): The user's transcribed spoken answer.
            q_data (dict): The current question data, including options.
        Returns:
            str | None: The matched option letter (A, B, C, D) or None if no match.
        """
        transcribed_lower = transcribed_text.lower().strip()
        user_selected_letter = None

        # Prepare option keys for validation
        option_keys = sorted(q_data['options'].keys())

        # Attempt 1: Use LLM for sophisticated matching
        if self.llm:
            options_dict = q_data['options']
            options_str = "\n".join([f"{key}: {options_dict[key]}" for key in option_keys])

            llm_match_prompt_text = f"""
            You are an assistant designed to identify a user's chosen option from a multiple-choice question based on their spoken (transcribed) answer.
            
            Question: {q_data['question']}

            Options:
            {options_str}

            User's Spoken Answer: "{transcribed_text}"

            Based on the user's spoken answer, which option (A, B, C, or D) did they choose?
            Respond ONLY with the single letter of the option (A, B, C, or D) that best matches the user's answer.
            If you are unsure or cannot identify a clear option from the provided options, respond with 'NONE'.
            Do not include any other text, explanations, or punctuation.
            """
            
            llm_match_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a precise answer matching assistant. Your sole task is to determine the correct option letter based on user's input. Respond ONLY with A, B, C, D, or NONE."),
                ("human", "{input}")
            ])
            llm_chain = llm_match_prompt | self.llm
            
            try:
                with st.spinner("Analyzing voice input with AI..."):
                    llm_response = llm_chain.invoke({"input": llm_match_prompt_text}).content.strip().upper()
                
                if llm_response in option_keys:
                    user_selected_letter = llm_response
                elif llm_response == "NONE":
                    st.warning("AI could not confidently identify your chosen option from your speech.")
                else:
                    st.warning(f"AI returned an unexpected response for voice matching: '{llm_response}'. Could not identify option.")

            except Exception as e:
                st.error(f"Error calling LLM for voice matching: {e}")
        else:
            st.warning("LLM not available for advanced voice matching. Falling back to simpler text matching.")

        # Fallback to simplified regex and text matching if LLM didn't set a letter or was not available
        if user_selected_letter is None:
            # Normalize the option texts for robust comparison (remove punctuation for better matching)
            normalized_options_text_only = {
                key: re.sub(r'[^\w\s]', '', q_data['options'][key]).lower().strip()
                for key in option_keys
            }
            
            # Attempt 1: Look for single letter at start or after common phrases
            letter_pattern = re.compile(
                r'(?:^|\b(?:option|answer\s*is|my\s*answer\s*is|i\s*choose|i\s*pick|it\s*is|that\s*is|select|the\s*letter|letter)\s*)'
                r'([a-d])(?:\b|\.|$|\s.*)'
            )
            match = letter_pattern.search(transcribed_lower)
            if match:
                possible_letter = match.group(1).upper()
                if possible_letter in option_keys:
                    user_selected_letter = possible_letter
            
            # Attempt 2: Check for full option text contained in transcription
            if user_selected_letter is None:
                for key in option_keys:
                    normalized_option_text = normalized_options_text_only[key]
                    if re.search(r'\b' + re.escape(normalized_option_text) + r'\b', transcribed_lower):
                        user_selected_letter = key
                        break
                    if len(transcribed_lower) > 1 and \
                       transcribed_lower not in ["the", "a", "an", "it", "is", "of", "and", "or", "but", "in", "for", "to"] and \
                       re.search(r'\b' + re.escape(transcribed_lower) + r'\b', normalized_option_text):
                        user_selected_letter = key
                        break
        return user_selected_letter


# --- 5. QuizManager Class ---
class QuizManager:
    """
    Manages the state and flow of the quiz using Streamlit's session state.
    Provides a clean API for quiz state manipulation, abstracting Streamlit's internals.
    """
    def __init__(self):
        # Initialize all necessary session state variables upon instantiation
        # This ensures all required keys exist from the start
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initializes or resets all Streamlit session state variables for the quiz."""
        st.session_state.setdefault('quiz_started', False)
        st.session_state.setdefault('questions', [])
        st.session_state.setdefault('current_question_index', 0)
        st.session_state.setdefault('score', 0)
        st.session_state.setdefault('user_transcribed_answer', "")
        st.session_state.setdefault('user_selected_radio_answer', None)
        st.session_state.setdefault('feedback_given_for_current_q', False)
        st.session_state.setdefault('answered_correctly_indices', set())
        st.session_state.setdefault('quiz_context', "")
        st.session_state.setdefault('question_audio_played_for_index', -1)
        st.session_state.setdefault('feedback_audio_played_for_index', -1)
        st.session_state.setdefault('last_recorded_audio', None)
        st.session_state.setdefault('answer_method_chosen', None) # 'radio', 'voice', or None

    def start_quiz(self, questions: list[dict], quiz_context: str = ""):
        """Sets up the quiz with new questions and resets state."""
        st.session_state.questions = questions
        st.session_state.quiz_started = True
        st.session_state.current_question_index = 0
        st.session_state.score = 0
        st.session_state.user_transcribed_answer = ""
        st.session_state.user_selected_radio_answer = None
        st.session_state.feedback_given_for_current_q = False
        st.session_state.answered_correctly_indices = set()
        st.session_state.question_audio_played_for_index = -1
        st.session_state.feedback_audio_played_for_index = -1
        st.session_state.last_recorded_audio = None
        st.session_state.answer_method_chosen = None
        st.session_state.quiz_context = quiz_context
        st.rerun()

    def next_question(self):
        """Advances to the next question and resets relevant per-question state."""
        st.session_state.current_question_index += 1
        st.session_state.feedback_given_for_current_q = False
        st.session_state.user_transcribed_answer = ""
        st.session_state.user_selected_radio_answer = None
        st.session_state.question_audio_played_for_index = -1
        st.session_state.feedback_audio_played_for_index = -1
        st.session_state.last_recorded_audio = None
        st.session_state.answer_method_chosen = None
        st.rerun()

    def restart_quiz(self):
        """Resets the entire quiz state to allow starting a new quiz."""
        st.session_state.quiz_started = False
        st.session_state.questions = []
        st.session_state.quiz_context = ""
        self._initialize_session_state() # Fully re-initialize all to default
        st.rerun()

    # Properties to access session state variables with clear names
    @property
    def is_quiz_started(self) -> bool:
        return st.session_state.quiz_started

    @property
    def questions(self) -> list[dict]:
        return st.session_state.questions

    @property
    def current_question_index(self) -> int:
        return st.session_state.current_question_index

    @property
    def current_question(self) -> dict | None:
        """Returns the data for the current question."""
        if self.is_quiz_started and self.questions and self.current_question_index < len(self.questions):
            return self.questions[self.current_question_index]
        return None

    @property
    def score(self) -> int:
        return st.session_state.score

    @score.setter
    def score(self, value: int):
        st.session_state.score = value

    @property
    def user_transcribed_answer(self) -> str:
        return st.session_state.user_transcribed_answer

    @user_transcribed_answer.setter
    def user_transcribed_answer(self, value: str):
        st.session_state.user_transcribed_answer = value

    @property
    def user_selected_radio_answer(self) -> str | None:
        return st.session_state.user_selected_radio_answer

    @user_selected_radio_answer.setter
    def user_selected_radio_answer(self, value: str | None):
        st.session_state.user_selected_radio_answer = value

    @property
    def feedback_given_for_current_q(self) -> bool:
        return st.session_state.feedback_given_for_current_q

    @feedback_given_for_current_q.setter
    def feedback_given_for_current_q(self, value: bool):
        st.session_state.feedback_given_for_current_q = value

    @property
    def answered_correctly_indices(self) -> set:
        return st.session_state.answered_correctly_indices
    
    def add_correct_answer_index(self, index: int):
        """Adds the index of a correctly answered question to the set."""
        st.session_state.answered_correctly_indices.add(index)

    @property
    def question_audio_played_for_index(self) -> int:
        return st.session_state.question_audio_played_for_index

    @question_audio_played_for_index.setter
    def question_audio_played_for_index(self, value: int):
        st.session_state.question_audio_played_for_index = value

    @property
    def feedback_audio_played_for_index(self) -> int:
        return st.session_state.feedback_audio_played_for_index

    @feedback_audio_played_for_index.setter
    def feedback_audio_played_for_index(self, value: int):
        st.session_state.feedback_audio_played_for_index = value

    @property
    def last_recorded_audio(self) -> bytes | None:
        return st.session_state.last_recorded_audio

    @last_recorded_audio.setter
    def last_recorded_audio(self, value: bytes | None):
        st.session_state.last_recorded_audio = value

    @property
    def answer_method_chosen(self) -> str | None:
        return st.session_state.answer_method_chosen

    @answer_method_chosen.setter
    def answer_method_chosen(self, value: str | None):
        st.session_state.answer_method_chosen = value

    def get_quiz_status_text(self) -> str:
        """Returns formatted text for the scoreboard."""
        current_q_display = self.current_question_index + 1 if self.is_quiz_started and self.questions else 0
        total_q_display = len(self.questions) if self.questions else 0
        return f"Score: {self.score} / {total_q_display}\nQuestion: {min(current_q_display, total_q_display)} / {total_q_display}"

# --- 6. QuizApp Class (Main Application Orchestrator) ---
class QuizApp:
    """
    The main class that orchestrates the Streamlit quiz application.
    It manages the UI and coordinates interactions between other service classes.
    """
    def __init__(self):
        st.set_page_config(page_title="🗣️ Interactive Voice Quiz Generator", layout="wide")
        st.title("📚 Interactive Quiz Generator")

        # Initialize core service classes
        self.config = ConfigManager() # Handles API keys and client initialization
        self.audio_manager = AudioManager(self.config.get_openai_client()) # Handles TTS/STT
        self.content_processor = QuizContentProcessor() # Handles PDF processing
        self.question_generator = QuestionGenerator(self.config.get_llm(), self.config.get_search_tool()) # Handles LLM calls
        self.quiz_manager = QuizManager() # Manages all quiz state in st.session_state
        
        # Display initial API availability warnings if any
        self._check_api_availability()

    def _check_api_availability(self):
        """Displays user-facing warnings if necessary API keys/clients are not available."""
        if not self.config.openai_api_key:
            st.error("🔴 OpenAI API key not found. Please set the `OPENAI_API_KEY` environment variable to use this app.")
        elif not self.config.is_llm_available():
            st.error("LLM initialization failed. Please check your API key validity or OpenAI service status.")
        elif not self.config.is_openai_client_available():
             st.error("OpenAI client for Whisper failed to initialize. Please check your API key validity or OpenAI service status.")

    def run(self):
        """Main method to run the Streamlit application layout."""
        self._render_sidebar()
        self._render_main_quiz_area()

    def _render_sidebar(self):
        """Renders the sidebar with quiz controls and scoreboard."""
        with st.sidebar:
            st.header("⚙️ Quiz Controls")
            if not self.quiz_manager.is_quiz_started:
                self._render_quiz_generation_controls()
            self._render_scoreboard()

    def _render_quiz_generation_controls(self):
        """Renders the UI for generating a new quiz (topic/PDF, num questions, difficulty)."""
        input_type = st.radio("Choose content source:", ("Topic", "Upload PDF"), key="input_type_sidebar")
        content_input = ""
        context_source_type = "topic"

        if input_type == "Topic":
            content_input = st.text_input("Enter the quiz topic:", key="topic_input_sidebar")
            context_source_type = "topic"
        else: # Upload PDF
            uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], key="pdf_uploader_sidebar")
            if uploaded_file:
                content_input = self.content_processor.process_pdf(uploaded_file)
                if content_input: # Only set context type if PDF processing was successful
                    context_source_type = "document"

        num_q_options = [1, 3, 5, 7, 10]
        num_questions = st.selectbox("Number of questions:", num_q_options, index=1, key="num_questions_select_sidebar")
        
        difficulty_options = ["Easy", "Medium", "Hard"]
        selected_difficulty = st.selectbox("Difficulty Level:", difficulty_options, index=1, key="difficulty_select_sidebar")

        if st.button("✨ Generate Quiz & Start", key="start_button_sidebar", disabled=not self.config.is_llm_available() or not content_input):
            if self.config.is_llm_available() and content_input:
                spinner_message = f"Generating {num_questions} {selected_difficulty.lower()} questions from your {context_source_type}..."
                with st.spinner(spinner_message):
                    # Call the question generator to get questions
                    questions = self.question_generator.generate_mcqs(content_input, num_questions, context_source_type, selected_difficulty)
                    if questions:
                        if len(questions) != num_questions:
                            st.warning(f"LLM generated {len(questions)} questions, but {num_questions} were requested. Proceeding with generated questions.")
                            st.text_area("LLM Raw Output (for debugging):", questions, height=150) # Show raw output for debugging
                        self.quiz_manager.start_quiz(questions, content_input)
                    else:
                        st.error("Failed to generate or parse questions. Please try again with a different topic or document.")
            elif not self.config.is_llm_available():
                st.error("LLM is not available. Please check your OPENAI_API_KEY.")
            elif not content_input:
                st.warning("Please provide content (topic or PDF).")

    def _render_scoreboard(self):
        """Renders the current score and question progress."""
        st.header("Scoreboard")
        st.write(self.quiz_manager.get_quiz_status_text())

    def _render_main_quiz_area(self):
        """Renders the main area where the quiz questions and interactions happen."""
        if self.quiz_manager.is_quiz_started and self.quiz_manager.questions:
            if self.quiz_manager.current_question_index >= len(self.quiz_manager.questions):
                self._render_quiz_finished_screen()
            else:
                self._render_current_question()
        elif not self.quiz_manager.is_quiz_started and self.config.is_llm_available():
            st.info("☝️ Provide content (topic or PDF) and number of questions in the sidebar to start the quiz!")

    def _render_quiz_finished_screen(self):
        """Displays the quiz completion screen with the final score."""
        st.balloons()
        st.header("🎉 Quiz Finished! 🎉")
        st.subheader(f"Your final score: {self.quiz_manager.score} out of {len(self.quiz_manager.questions)}")
        
        if st.button("Restart Quiz with New Topic", key="restart_quiz_main"):
            self.quiz_manager.restart_quiz()

    def _render_current_question(self):
        """Renders the current quiz question, plays audio, and handles user input."""
        q_data = self.quiz_manager.current_question
        if not q_data: # Safety check if current_question is somehow None
            st.error("Could not load current question data. Please restart the quiz.")
            return

        st.subheader(f"Question {self.quiz_manager.current_question_index + 1}")
        st.markdown(f"### {q_data['question']}")

        option_keys = sorted(q_data['options'].keys()) 
        display_options = [f"{key}: {q_data['options'][key]}" for key in option_keys]

        # Only play question audio if it hasn't been played for the current question
        if self.quiz_manager.question_audio_played_for_index != self.quiz_manager.current_question_index:
            question_audio_text = f"Question {self.quiz_manager.current_question_index + 1}. {q_data['question']}. "
            for i, opt_text in enumerate(display_options):
                question_audio_text += f"Option {option_keys[i]} is {q_data['options'][option_keys[i]]}. "
            
            audio_io = self.audio_manager.text_to_speech(question_audio_text)
            if audio_io:
                audio_bytes_val = audio_io.getvalue()
                # Using a unique key with time.time() to force Streamlit to re-render and autoplay
                st.audio(audio_bytes_val, format="audio/mp3", autoplay=True) 
                self.quiz_manager.question_audio_played_for_index = self.quiz_manager.current_question_index
            else:
                st.warning("Audio generation failed for the question.")

        st.markdown("---")
        
        if not self.quiz_manager.feedback_given_for_current_q:
            self._render_answer_input_methods(q_data, option_keys, display_options)
        else:
            self._render_feedback_and_next_button(q_data, option_keys)

    def _render_answer_input_methods(self, q_data: dict, option_keys: list, display_options: list):
        """Renders the radio button and voice input sections."""
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Choose an Option:")
            selected_radio = st.radio(
                "Your answer:",
                options=display_options,
                index=None,
                key=f"q_radio_{self.quiz_manager.current_question_index}",
                disabled=(self.quiz_manager.answer_method_chosen == 'voice' and self.quiz_manager.user_transcribed_answer != "")
            )
            
            # Update state if a radio option is newly selected
            if selected_radio and self.quiz_manager.user_selected_radio_answer != selected_radio:
                self.quiz_manager.user_selected_radio_answer = selected_radio
                self.quiz_manager.answer_method_chosen = 'radio'
                self.quiz_manager.user_transcribed_answer = "" # Clear voice input if radio is selected
                self.quiz_manager.last_recorded_audio = None
                st.rerun() # Rerun to update state and disable voice input

            submit_text_button = st.button("Submit Text Answer", key=f"submit_text_btn_{self.quiz_manager.current_question_index}", 
                                           disabled=not self.quiz_manager.user_selected_radio_answer or self.quiz_manager.answer_method_chosen == 'voice')
            
            if submit_text_button:
                if self.quiz_manager.user_selected_radio_answer:
                    self.quiz_manager.feedback_given_for_current_q = True
                    st.rerun()
                else:
                    st.warning("Please select an answer from the options.")

        with col2:
            st.markdown("#### Or Speak Your Answer:")
            st.write("Say the letter (e.g., 'A') or the full option text.")
            
            # Streamlit's native audio input widget
            audio_file_object = st.audio_input(
                "Record or Upload Voice Message",
                key=f"audio_input_{self.quiz_manager.current_question_index}",
                disabled=(self.quiz_manager.answer_method_chosen == 'radio' and self.quiz_manager.user_selected_radio_answer is not None)
            )

            if audio_file_object is not None:
                audio_bytes_from_input = audio_file_object.getvalue()
                
                # Only transcribe if new audio is detected
                if audio_bytes_from_input and audio_bytes_from_input != self.quiz_manager.last_recorded_audio:
                    self.quiz_manager.last_recorded_audio = audio_bytes_from_input # Store to detect changes
                    st.info("Audio recorded/uploaded. Transcribing...")
                    transcribed_text = self.audio_manager.transcribe_audio_with_whisper(audio_bytes_from_input)
                    
                    if transcribed_text:
                        self.quiz_manager.user_transcribed_answer = transcribed_text
                        self.quiz_manager.answer_method_chosen = 'voice'
                        self.quiz_manager.user_selected_radio_answer = None # Clear radio selection if voice is used
                        st.success(f"You said: \"{transcribed_text}\"")
                        st.rerun() # Re-run to update UI/disable radio options
                    else:
                        st.error("Failed to transcribe audio.")
                        self.quiz_manager.user_transcribed_answer = ""
                        self.quiz_manager.answer_method_chosen = None # Reset if transcription fails
            
            # Display transcribed answer and submission button if voice input is active
            if self.quiz_manager.user_transcribed_answer:
                st.text_input("Transcribed Answer:", value=self.quiz_manager.user_transcribed_answer, disabled=True, key=f"transcribed_input_{self.quiz_manager.current_question_index}")
                
                submit_voice_button = st.button("Confirm & Submit Voice Answer", key=f"submit_voice_btn_{self.quiz_manager.current_question_index}",
                                                disabled=self.quiz_manager.answer_method_chosen != 'voice')
                
                if submit_voice_button:
                    self.quiz_manager.feedback_given_for_current_q = True
                    st.rerun()

    def _render_feedback_and_next_button(self, q_data: dict, option_keys: list):
        """Renders the feedback (correct/incorrect, explanation) and the "Next Question" button."""
        user_input_method = self.quiz_manager.answer_method_chosen
        user_answer_text = ""
        user_selected_letter = None
        
        if user_input_method == 'radio':
            chosen_display_option = self.quiz_manager.user_selected_radio_answer
            user_answer_text = f"You chose: **{chosen_display_option}**"
            user_selected_letter = chosen_display_option[0] if chosen_display_option else None
        elif user_input_method == 'voice':
            chosen_display_option_text = self.quiz_manager.user_transcribed_answer
            user_answer_text = f"You said: **{chosen_display_option_text}**"
            
            # Use QuestionGenerator's method for robust voice answer matching
            user_selected_letter = self.question_generator.match_voice_answer_to_option(chosen_display_option_text, q_data)
        
        st.markdown(user_answer_text)
        
        correct_option_letter = q_data['correct_option']
        feedback_text_short = ""

        if user_selected_letter == correct_option_letter:
            st.success("Correct! 🎉")
            feedback_text_short = "Correct!"
            if self.quiz_manager.current_question_index not in self.quiz_manager.answered_correctly_indices:
                self.quiz_manager.score += 1 # Increment score
                self.quiz_manager.add_correct_answer_index(self.quiz_manager.current_question_index)
        else:
            st.error("Incorrect. 😟")
            # Provide more specific feedback if voice input failed to match
            if user_input_method == 'voice' and not user_selected_letter:
                feedback_text_short = f"Incorrect. Your spoken answer could not be clearly matched to an option. The correct answer was {correct_option_letter}: {q_data['options'][correct_option_letter]}."
            else:
                feedback_text_short = f"Incorrect. The correct answer was {correct_option_letter}: {q_data['options'][correct_option_letter]}."
        
        st.info(f"**Correct Answer:** {correct_option_letter}: {q_data['options'][correct_option_letter]}")
        st.markdown(f"**Explanation:** {q_data['explanation']}")

        full_feedback_audio_text = f"{feedback_text_short} {q_data['explanation']}"
        
        # Play feedback audio only once per question after feedback is given
        if self.quiz_manager.feedback_audio_played_for_index != self.quiz_manager.current_question_index:
            # We use a button for playback so the user can choose to listen
            if st.button("🔊 Play Feedback", key=f"play_feedback_{self.quiz_manager.current_question_index}_{time.time()}"):
                audio_io = self.audio_manager.text_to_speech(full_feedback_audio_text)
                if audio_io:
                    audio_bytes_val = audio_io.getvalue()
                    # Using a unique key with time.time() to force Streamlit to re-render and autoplay
                    st.audio(audio_bytes_val, format="audio/mp3", autoplay=True) 
                    self.quiz_manager.feedback_audio_played_for_index = self.quiz_manager.current_question_index
                else:
                    st.warning("Audio generation failed for the feedback.")
        else:
             # If feedback audio was already played, just show a disabled button
             st.button("🔊 Play Feedback", key=f"play_feedback_disabled_{self.quiz_manager.current_question_index}", disabled=True, help="Audio already played for this feedback.")


        if st.button("Next Question", key=f"next_btn_{self.quiz_manager.current_question_index}"):
            self.quiz_manager.next_question()

# --- Main Entry Point ---
if __name__ == "__main__":
    app = QuizApp()
    app.run()
