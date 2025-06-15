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
from openai import OpenAI # Import OpenAI client for Whisper API

# --- Configuration ---
load_dotenv()
llm = None
search_tool = None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Initialize OpenAI client for Whisper
openai_client = None
if OPENAI_API_KEY:
    try:
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3, openai_api_key=OPENAI_API_KEY)
        openai_client = OpenAI(api_key=OPENAI_API_KEY) # Initialize OpenAI client
    except Exception as e:
        st.error(f"Error initializing LLM or OpenAI client. Details: {e}")
    
    if SERPAPI_API_KEY:
        search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
        search_tool = Tool(name="Search", func=search.run, description="useful for when you need to answer questions about current events or things you don't have in your knowledge base")
    else:
        st.warning("SERPAPI_API_KEY not found. Internet search for quiz generation will be disabled.")
else:
    st.warning("OPENAI_API_KEY not found in environment variables. LLM and Whisper features will be disabled.")

# --- Helper Functions ---
def text_to_speech(text, lang='en'):
    """Converts text to speech and returns a BytesIO object."""
    fp = io.BytesIO()
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(fp)
        fp.seek(0)
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None
    return fp

def transcribe_audio_with_whisper(audio_bytes: bytes):
    """Transcribes audio bytes using OpenAI's Whisper API."""
    if not openai_client:
        st.error("OpenAI client not initialized. Cannot transcribe audio.")
        return None

    # Create a BytesIO object from the audio bytes
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.mp3" # Whisper API needs a file-like object with a name

    try:
        with st.spinner("Transcribing audio with Whisper..."):
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return transcript
    except Exception as e:
        st.error(f"Error transcribing audio with Whisper: {e}")
        return None

# Modified to accept difficulty
def generate_mcqs_via_llm(content: str, num_questions: int = 3, context_source_type: str = "topic", difficulty: str = "Medium"):
    """
    Generates MCQs using the LLM based on the provided topic and number of questions.
    """
    if not llm:
        st.error("LLM not initialized. Cannot generate questions.")
        return None

    # Add difficulty instruction to the prompt
    # In QuestionGenerator class
    DIFFICULTY_GUIDELINES = {
        "Easy": "Questions should test direct recall of information. Use simple vocabulary. The correct answer should be obvious from direct textual evidence.",
        "Medium": "Questions may require basic comprehension or simple inference. Options should be plausible distractors, requiring careful reading. Use moderate vocabulary.",
        "Hard": "Questions should require deeper analysis, synthesis of information, or complex inference. Options might be very close, requiring subtle distinctions or understanding of implications. Use advanced vocabulary and concepts."
    }

    # Then in generate_mcqs:
    difficulty_instruction = f"Generate questions. {DIFFICULTY_GUIDELINES[difficulty]}"

    if context_source_type == "topic" and search_tool:
        task_description_for_agent = (
            f"You are an expert quiz question generator. Your task is to generate exactly {num_questions} multiple-choice question(s) on the topic: \"{content}\".\n"
            f"{difficulty_instruction}\n" # Added difficulty instruction
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
        agent = create_openai_tools_agent(llm, [search_tool], agent_llm_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[search_tool], verbose=True, max_iterations=7, handle_parsing_errors=True)

        try:
            st.info("Agent is thinking and potentially searching for information...")
            response_dict = agent_executor.invoke({"input": task_description_for_agent})
            
            raw_output = response_dict.get("output", "")
            
            quiz_content_match = re.search(r'(?i)(?:Question|Q):\s*.*', raw_output, re.DOTALL)
            
            if quiz_content_match:
                return quiz_content_match.group(0)
            elif "Agent stopped due to iteration limit or time limit" in raw_output or \
                 "I genuinely do not know how to proceed" in raw_output or \
                 "I cannot generate questions on this topic" in raw_output:
                st.error("The AI agent encountered an issue and could not generate questions based on the topic. This might be due to lack of verifiable information, a very complex/ambiguous query, or an internal agent error that it couldn't resolve.")
                st.text_area("Agent's internal (failure) output:", raw_output, height=150)
                return None
            else:
                st.error("The AI agent generated an unexpected response format. Could not extract questions.")
                st.text_area("Agent's raw output (unexpected format):", raw_output, height=150)
                return None

        except Exception as e:
            st.error(f"Error during agent execution: {e}")
            st.text_area("Agent execution error details:", str(e), height=150)
            quiz_content_match_from_exception = re.search(r'(?i)(?:Question|Q):\s*.*', str(e), re.DOTALL)
            if quiz_content_match_from_exception:
                st.warning("Attempting to parse questions found within the agent's error message. This might indicate an underlying agent issue, but questions were recoverable.")
                return quiz_content_match_from_exception.group(0)
            return None
    else:
        if context_source_type == "document":
            instruction_text = (
                f"Generate exactly {num_questions} multiple-choice question(s) based *only* on the following document content. "
                f"{difficulty_instruction}\n" # Added difficulty instruction
                "Ensure all information, including options and the correct answer, is directly verifiable from this text. "
                "If the document is too short or lacks sufficient detail for the requested number of questions, generate as many as possible."
            )
        else:
            st.warning("Search tool not available. Generating questions based on LLM's existing knowledge, which may be limited for very recent or future events.")
            instruction_text = (
                f"Generate exactly {num_questions} multiple-choice question(s) on the topic: \"{content}\". "
                f"{difficulty_instruction}\n" # Added difficulty instruction
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
        chain = prompt | llm
        try:
            response = chain.invoke({"content": content, "num_questions": num_questions})
            return response.content
        except Exception as e:
            st.error(f"Error calling LLM directly: {e}")
            return None

def parse_llm_response(raw_text: str):
    """
    Parses the raw text response from LLM into a list of question dictionaries.
    """
    questions = []
    if not raw_text or not isinstance(raw_text, str):
        if not isinstance(raw_text, str) and raw_text is not None:
            st.warning(f"LLM response was not a string: {type(raw_text)}. Cannot parse.")
        return questions

    question_blocks = re.split(r'(?i)\n*(?:Question|Q)\s*\d*:\s*', raw_text.strip())
    
    for block_content in question_blocks:
        if not block_content.strip():
            continue

        current_question = {}
        
        q_match = re.match(r'(.*?)\n(?=[A-D][\)\.:]\s*|Correct Option:|Explanation:)', block_content, re.DOTALL | re.IGNORECASE)
        if q_match:
            current_question["question"] = q_match.group(1).strip()
            remaining_block = block_content[len(q_match.group(1)):].strip() 
        else:
            q_match_fallback = re.match(r'(.*?)(?=\nCorrect Option:|\nExplanation:|$)', block_content, re.DOTALL | re.IGNORECASE)
            if q_match_fallback:
                current_question["question"] = q_match_fallback.group(1).strip()
                remaining_block = block_content[len(q_match_fallback.group(1)):].strip()
            else:
                st.warning(f"Could not find question text for a block. Skipping. Block content: {block_content[:100]}...")
                continue

        options = {}
        option_matches = re.findall(
            r'([A-Da-d])(?:[\)\.:]|\s*):\s*(.*?)(?=\n\s*(?:[A-Da-d][\)\.:]|\s*Correct Option:|\s*Explanation:)|$)',
            remaining_block, re.IGNORECASE | re.DOTALL
        )
        for opt_letter, opt_text in option_matches:
            options[opt_letter.upper()] = opt_text.strip()
        
        current_question["options"] = options
        
        correct_match = re.search(r'Correct Option:\s*([A-Da-d])', block_content, re.IGNORECASE)
        if correct_match:
            current_question["correct_option"] = correct_match.group(1).upper()
        else:
            st.warning(f"Could not find Correct Option for: {current_question.get('question', 'N/A')}. Skipping.")
            continue

        explanation_match = re.search(r'Explanation:\s*(.*?)(?=\n*(?:Question:|Q:|I cannot generate questions on this topic|Agent stopped|This query is outside my current knowledge base|$))', block_content, re.DOTALL | re.IGNORECASE)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
            explanation = re.sub(r'(?i)\n*Question:\s*.*', '', explanation).strip()
            current_question["explanation"] = explanation
        else:
            st.warning(f"Could not find Explanation for: {current_question.get('question', 'N/A')}. Skipping.")
            continue

        if all(key in current_question for key in ["question", "options", "correct_option", "explanation"]) \
           and len(current_question["options"]) >= 2 \
           and current_question["correct_option"] in current_question["options"]:
            questions.append(current_question)
        else:
            st.warning(f"Skipping malformed question block due to missing key or invalid options: {block_content[:200]}...")

    return questions

# --- Streamlit App UI ---
st.set_page_config(page_title="🗣️ Interactive Voice Quiz Generator", layout="wide")
st.title("📚 Interactive Quiz Generator")

# Initialize session state variables
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'user_transcribed_answer' not in st.session_state: # Stores Whisper's transcription
    st.session_state.user_transcribed_answer = ""
if 'user_selected_radio_answer' not in st.session_state: # Stores user's radio choice
    st.session_state.user_selected_radio_answer = None
if 'feedback_given_for_current_q' not in st.session_state:
    st.session_state.feedback_given_for_current_q = False
if 'answered_correctly_indices' not in st.session_state:
    st.session_state.answered_correctly_indices = set()
if 'quiz_context' not in st.session_state:
    st.session_state.quiz_context = ""
if 'question_audio_played_for_index' not in st.session_state:
    st.session_state.question_audio_played_for_index = -1
if 'last_recorded_audio' not in st.session_state: # To store audio bytes from recorder
    st.session_state.last_recorded_audio = None
if 'answer_method_chosen' not in st.session_state: # 'radio', 'voice', or None
    st.session_state.answer_method_chosen = None
# New state for feedback audio
if 'feedback_audio_played_for_index' not in st.session_state:
    st.session_state.feedback_audio_played_for_index = -1


# Sidebar for controls and score
with st.sidebar:
    st.header("⚙️ Quiz Controls")
    if not st.session_state.quiz_started:
        input_type = st.radio("Choose content source:", ("Topic", "Upload PDF"), key="input_type_sidebar")
        
        content_input = ""
        context_source_type = "topic"

        if input_type == "Topic":
            content_input = st.text_input("Enter the quiz topic:", key="topic_input_sidebar")
            context_source_type = "topic"
        else: # Upload PDF
            uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], key="pdf_uploader_sidebar")
            if uploaded_file:
                with st.spinner("Processing PDF..."):
                    try:
                        temp_file_path = f"temp_{uploaded_file.name}"
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        loader = PyPDFLoader(temp_file_path)
                        docs = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                        chunks = text_splitter.split_documents(docs)
                        content_input = " ".join([chunk.page_content for chunk in chunks])
                        st.session_state.quiz_context = content_input
                        os.remove(temp_file_path)
                        st.success("PDF processed!")
                        context_source_type = "document"
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
                        content_input = ""

        num_q_options = [1, 3, 5, 7, 10]
        num_questions = st.selectbox("Number of questions:", num_q_options, index=1, key="num_questions_select_sidebar")
        
        # --- New Difficulty Level Selector ---
        difficulty_options = ["Easy", "Medium", "Hard"]
        selected_difficulty = st.selectbox("Difficulty Level:", difficulty_options, index=1, key="difficulty_select_sidebar")
        # --- End New Difficulty Level Selector ---

        if st.button("✨ Generate Quiz & Start", key="start_button_sidebar", disabled=not llm or not content_input):
            if llm and content_input:
                spinner_message = f"Generating {num_questions} {selected_difficulty.lower()} questions from your {context_source_type}..."
                with st.spinner(spinner_message):
                    # Pass selected_difficulty to the function
                    raw_mcq_text = generate_mcqs_via_llm(content_input, num_questions, context_source_type, selected_difficulty)
                    
                    if raw_mcq_text:
                        parsed_questions = parse_llm_response(raw_mcq_text)
                        
                        if parsed_questions:
                            if len(parsed_questions) != num_questions:
                                st.warning(f"LLM generated {len(parsed_questions)} questions, but {num_questions} were requested. Proceeding with generated questions.")
                                st.text_area("LLM Raw Output (for debugging):", raw_mcq_text, height=150)
                            
                            st.session_state.questions = parsed_questions
                            st.session_state.quiz_started = True
                            st.session_state.current_question_index = 0
                            st.session_state.score = 0
                            st.session_state.user_transcribed_answer = ""
                            st.session_state.user_selected_radio_answer = None
                            st.session_state.feedback_given_for_current_q = False
                            st.session_state.answered_correctly_indices = set()
                            st.session_state.question_audio_played_for_index = -1 # Reset for new quiz
                            st.session_state.feedback_audio_played_for_index = -1 # Reset for new quiz
                            st.session_state.last_recorded_audio = None
                            st.session_state.answer_method_chosen = None # Reset
                            if context_source_type == "topic":
                                st.session_state.quiz_context = content_input
                            st.success(f"Generated {len(parsed_questions)} questions. Ready to start!")
                            st.rerun()
                        else:
                            st.error("Could not parse any questions from the LLM response. The format might be incorrect or the content was insufficient.")
                            st.text_area("LLM Raw Output (for debugging):", raw_mcq_text, height=250)
                    else:
                        st.error("Failed to get a response from the LLM. Check API key, network, or LLM status, or try a different topic.")
            elif not llm:
                st.error("LLM is not available. Please check your OPENAI_API_KEY.")
            elif not content_input:
                st.warning("Please provide content (topic or PDF).")

    st.header("Scoreboard")
    current_q_display = st.session_state.current_question_index + 1 if st.session_state.quiz_started and st.session_state.questions else 0
    total_q_display = len(st.session_state.questions) if st.session_state.questions else 0
    st.write(f"Score: {st.session_state.score} / {total_q_display}")
    if st.session_state.quiz_started and total_q_display > 0:
        st.write(f"Question: {min(current_q_display, total_q_display)} / {total_q_display}")


# Main Quiz Area
if not OPENAI_API_KEY:
    st.error("🔴 OpenAI API key not found. Please set the `OPENAI_API_KEY` environment variable to use this app.")
elif not llm and OPENAI_API_KEY:
    st.error("LLM initialization failed. Please check your API key validity or OpenAI service status.")
elif not openai_client and OPENAI_API_KEY:
     st.error("OpenAI client for Whisper failed to initialize. Please check your API key validity or OpenAI service status.")


if st.session_state.quiz_started and st.session_state.questions:
    if st.session_state.current_question_index >= len(st.session_state.questions):
        st.balloons()
        st.header("🎉 Quiz Finished! 🎉")
        st.subheader(f"Your final score: {st.session_state.score} out of {len(st.session_state.questions)}")
        
        if st.button("Restart Quiz with New Topic", key="restart_quiz_main"):
            st.session_state.quiz_started = False
            st.session_state.questions = []
            st.session_state.quiz_context = ""
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
            st.rerun()
    else:
        q_data = st.session_state.questions[st.session_state.current_question_index]
        
        st.subheader(f"Question {st.session_state.current_question_index + 1}")
        st.markdown(f"### {q_data['question']}")

        option_keys = sorted(q_data['options'].keys()) 
        display_options = [f"{key}: {q_data['options'][key]}" for key in option_keys]

        # Only play question audio if it hasn't been played for the current question
        if st.session_state.question_audio_played_for_index != st.session_state.current_question_index:
            question_audio_text = f"Question {st.session_state.current_question_index + 1}. {q_data['question']}. "
            for i, opt_text in enumerate(display_options):
                question_audio_text += f"Option {option_keys[i]} is {q_data['options'][option_keys[i]]}. "
            
            audio_io = text_to_speech(question_audio_text)
            if audio_io:
                audio_bytes_val = audio_io.getvalue()
                st.audio(audio_bytes_val, format="audio/mp3", autoplay=True) 
                st.session_state.question_audio_played_for_index = st.session_state.current_question_index
            else:
                st.warning("Audio generation failed for the question.")

        st.markdown("---")
        
        # --- Display Options and Voice Input simultaneously ---
        if not st.session_state.feedback_given_for_current_q:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Choose an Option:")
                # Radio button for answer selection
                # Disable if voice input has been recorded
                selected_radio = st.radio(
                    "Your answer:",
                    options=display_options,
                    index=None,
                    key=f"q_radio_{st.session_state.current_question_index}",
                    disabled=(st.session_state.answer_method_chosen == 'voice' and st.session_state.user_transcribed_answer != "")
                )
                
                # If a radio option is selected, set the chosen method
                if selected_radio and st.session_state.user_selected_radio_answer != selected_radio:
                    st.session_state.user_selected_radio_answer = selected_radio
                    st.session_state.answer_method_chosen = 'radio'
                    # Clear voice input if radio is selected after voice
                    st.session_state.user_transcribed_answer = ""
                    st.session_state.last_recorded_audio = None
                    # No rerun here, let the submit button trigger rerun for feedback

                submit_text_button = st.button("Submit Text Answer", key=f"submit_text_btn_{st.session_state.current_question_index}", 
                                               disabled=not st.session_state.user_selected_radio_answer or st.session_state.answer_method_chosen == 'voice')
                
                if submit_text_button:
                    if st.session_state.user_selected_radio_answer:
                        st.session_state.feedback_given_for_current_q = True
                        st.rerun()
                    else:
                        st.warning("Please select an answer from the options.")

            with col2:
                st.markdown("#### Or Speak Your Answer:")
                st.write("Say the letter (e.g., 'A') or the full option text.")
                
                audio_file_object = st.audio_input(
                    "Record or Upload Voice Message",
                    key=f"audio_input_{st.session_state.current_question_index}",
                    disabled=(st.session_state.answer_method_chosen == 'radio' and st.session_state.user_selected_radio_answer is not None)
                )

                if audio_file_object is not None:
                    audio_bytes_from_input = audio_file_object.getvalue()
                    
                    if audio_bytes_from_input and audio_bytes_from_input != st.session_state.last_recorded_audio:
                        st.session_state.last_recorded_audio = audio_bytes_from_input
                        st.info("Audio recorded/uploaded. Transcribing...")
                        transcribed_text = transcribe_audio_with_whisper(audio_bytes_from_input)
                        
                        if transcribed_text:
                            st.session_state.user_transcribed_answer = transcribed_text
                            st.session_state.answer_method_chosen = 'voice'
                            st.session_state.user_selected_radio_answer = None
                            st.success(f"You said: \"{transcribed_text}\"")
                            st.rerun() # Rerun to update state/disable radio input
                        else:
                            st.error("Failed to transcribe audio.")
                            st.session_state.user_transcribed_answer = ""
                            st.session_state.answer_method_chosen = None
                
                if st.session_state.user_transcribed_answer:
                    st.text_input("Transcribed Answer:", value=st.session_state.user_transcribed_answer, disabled=True, key=f"transcribed_input_{st.session_state.current_question_index}")
                    
                    submit_voice_button = st.button("Confirm & Submit Voice Answer", key=f"submit_voice_btn_{st.session_state.current_question_index}",
                                                    disabled=st.session_state.answer_method_chosen != 'voice')
                    
                    if submit_voice_button:
                        st.session_state.feedback_given_for_current_q = True
                        st.rerun()
        else: # Feedback has been given, show the chosen answer, correctness, and explanation.
            user_input_method = st.session_state.answer_method_chosen
            user_answer_text = ""
            user_selected_letter = None
            
            if user_input_method == 'radio':
                chosen_display_option = st.session_state.user_selected_radio_answer
                user_answer_text = f"You chose: **{chosen_display_option}**"
                user_selected_letter = chosen_display_option[0] if chosen_display_option else None
            elif user_input_method == 'voice':
                chosen_display_option_text = st.session_state.user_transcribed_answer
                user_answer_text = f"You said: **{chosen_display_option_text}**"
                
                # --- Improved Voice Matching Logic - REVISED with LLM ---
                transcribed_lower = chosen_display_option_text.lower().strip()
                user_selected_letter = None # Reset for this block

                # Only attempt LLM matching if LLM is initialized
                if llm:
                    question_text = q_data['question']
                    options_dict = q_data['options']
                    
                    # Format options for the LLM prompt
                    options_str = "\n".join([f"{key}: {options_dict[key]}" for key in sorted(options_dict.keys())])

                    llm_match_prompt_text = f"""
                    You are an assistant designed to identify a user's chosen option from a multiple-choice question based on their spoken (transcribed) answer.
                    
                    Question: {question_text}

                    Options:
                    {options_str}

                    User's Spoken Answer: "{chosen_display_option_text}"

                    Based on the user's spoken answer, which option (A, B, C, or D) did they choose?
                    Respond ONLY with the single letter of the option (A, B, C, or D) that best matches the user's answer.
                    If you are unsure or cannot identify a clear option from the provided options, respond with 'NONE'.
                    Do not include any other text, explanations, or punctuation.
                    """
                    
                    # Create a prompt template for the LLM call
                    llm_match_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a precise answer matching assistant. Your sole task is to determine the correct option letter based on user's input. Respond ONLY with A, B, C, D, or NONE."),
                        ("human", "{input}") # The entire formatted prompt text goes into 'input'
                    ])

                    # Create a chain to invoke the LLM
                    llm_chain = llm_match_prompt | llm
                    
                    try:
                        with st.spinner("Analyzing voice input with AI..."):
                            # Invoke the LLM with the structured input
                            llm_response = llm_chain.invoke({"input": llm_match_prompt_text}).content.strip().upper()
                        
                        # Check if the LLM's response is a valid option letter
                        if llm_response in option_keys:
                            user_selected_letter = llm_response
                        elif llm_response == "NONE":
                            st.warning("AI could not confidently identify your chosen option from your speech.")
                            # user_selected_letter remains None, allowing fallback
                        else:
                            st.warning(f"AI returned an unexpected response for voice matching: '{llm_response}'. Could not identify option.")
                            # user_selected_letter remains None, allowing fallback

                    except Exception as e:
                        st.error(f"Error calling LLM for voice matching: {e}")
                        # user_selected_letter remains None, allowing fallback
                else:
                    st.warning("LLM not available for advanced voice matching. Falling back to simpler text matching.")


                # Fallback to simplified regex and text matching if LLM didn't set a letter or was not available
                if user_selected_letter is None:
                    # Normalized options (already done above, but re-init if LLM path wasn't taken)
                    normalized_options_text_only = {
                        key: re.sub(r'[^\w\s]', '', q_data['options'][key]).lower().strip()
                        for key in option_keys
                    }
                    
                    # Attempt 1: Look for single letter at start or after common phrases
                    letter_pattern = re.compile(
                        r'(?:^|\b(?:option|answer\s*is|my\s*answer\s*is|i\s*choose|i\s*pick|it\s*is|that\s*is|select|the\s*letter|letter)\s*)'
                        r'([a-d])(?:\b|\.|$|\s.*)' # Captures 'a', 'b', 'c', or 'd', allowing for more text after
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
                            
                            # Check if the normalized option text (as a whole word/phrase) is present in the transcription
                            if re.search(r'\b' + re.escape(normalized_option_text) + r'\b', transcribed_lower):
                                user_selected_letter = key
                                break

                            # Check if the transcription (as a whole word/phrase) is present in the normalized option text
                            # Added minimum length check to avoid matching very common single words
                            if len(transcribed_lower) > 1 and \
                               transcribed_lower not in ["the", "a", "an", "it", "is", "of", "and", "or", "but", "in", "for", "to"] and \
                               re.search(r'\b' + re.escape(transcribed_lower) + r'\b', normalized_option_text):
                                user_selected_letter = key
                                break
                # --- End Improved Voice Matching Logic ---
            
            st.markdown(user_answer_text)
            
            correct_option_letter = q_data['correct_option']
            feedback_text_short = ""

            if user_selected_letter == correct_option_letter:
                st.success("Correct! 🎉")
                feedback_text_short = "Correct!"
                if st.session_state.current_question_index not in st.session_state.answered_correctly_indices:
                    st.session_state.score += 1
                    st.session_state.answered_correctly_indices.add(st.session_state.current_question_index)
            else:
                st.error("Incorrect. 😟")
                # If voice input and couldn't match, provide more context
                if user_input_method == 'voice' and not user_selected_letter:
                    feedback_text_short = f"Incorrect. Your spoken answer could not be clearly matched to an option. The correct answer was {correct_option_letter}: {q_data['options'][correct_option_letter]}."
                else:
                    feedback_text_short = f"Incorrect. The correct answer was {correct_option_letter}: {q_data['options'][correct_option_letter]}."
            
            st.info(f"**Correct Answer:** {correct_option_letter}: {q_data['options'][correct_option_letter]}")
            st.markdown(f"**Explanation:** {q_data['explanation']}")

            full_feedback_audio_text = f"{feedback_text_short} {q_data['explanation']}"
            
            # Only play feedback audio if it hasn't been played for the current question after feedback is given
            if st.session_state.feedback_audio_played_for_index != st.session_state.current_question_index:
                if st.button("🔊 Play Feedback", key=f"play_feedback_{st.session_state.current_question_index}_{time.time()}"):
                    audio_io = text_to_speech(full_feedback_audio_text)
                    if audio_io:
                        audio_bytes_val = audio_io.getvalue()
                        st.audio(audio_bytes_val, format="audio/mp3", autoplay=True) 
                        st.session_state.feedback_audio_played_for_index = st.session_state.current_question_index
                    else:
                        st.warning("Audio generation failed for the feedback.")
            else:
                 # If feedback audio was already played, just show a disabled button or remove it
                 st.button("🔊 Play Feedback", key=f"play_feedback_disabled_{st.session_state.current_question_index}", disabled=True, help="Audio already played for this feedback.")


            if st.button("Next Question", key=f"next_btn_{st.session_state.current_question_index}"):
                st.session_state.current_question_index += 1
                st.session_state.feedback_given_for_current_q = False
                st.session_state.user_transcribed_answer = ""
                st.session_state.user_selected_radio_answer = None
                st.session_state.question_audio_played_for_index = -1 # Reset for next question
                st.session_state.feedback_audio_played_for_index = -1 # Reset for next question
                st.session_state.last_recorded_audio = None
                st.session_state.answer_method_chosen = None # Reset for next question
                st.rerun()
elif not st.session_state.quiz_started and llm:
    st.info("☝️ Provide content (topic or PDF) and number of questions in the sidebar to start the quiz!")