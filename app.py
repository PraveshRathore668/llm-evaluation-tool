import streamlit as st
import random
import time
import pandas as pd
import wikipedia
import re
import statistics
import itertools
from typing import Optional # Import Optional for type hinting

# Libraries for computation-based metrics (assumption: these are installed)
# pip install rouge-score nltk scikit-learn sentence-transformers
try:
    from rouge_score import rouge_scorer
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from sklearn.metrics import accuracy_score, f1_score
    from sentence_transformers import SentenceTransformer, util
    # Download NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    _sentence_transformer_model = None # Lazy load model
except ImportError:
    st.warning("Some computation-based metric libraries are not installed. Please install them using: `pip install rouge-score nltk scikit-learn sentence-transformers`")
    rouge_scorer = None
    nltk = None
    sentence_bleu = None
    SmoothingFunction = None
    accuracy_score = None
    f1_score = None
    SentenceTransformer = None
    util = None

st.set_page_config(layout="wide", page_title="LLM Multi-Round Evaluation Tool")
st.title("🎯 LLM Multi-Round Evaluation Tool")

NUM_ROUNDS = 5

PREDEFINED_LLMS = [
    {"label": "MedPaLM2_Google_Medical", "model_id": "models/gemini-1.5-pro-latest"},
    {"label": "GPT4_OpenAI_Medical", "model_id": "gpt-4-turbo"},
    {"label": "Claude3_Anthropic_Medical", "model_id": "claude-3-opus-20240229"},
    {"label": "Llama3_Meta_Medical", "model_id": "meta-llama/Llama-3-70b-instruct"},
    {"label": "Gemini_Google_Medical", "model_id": "models/gemini-1.5-flash-latest"},
    {"label": "GPT4_OpenAI_Market", "model_id": "gpt-4"},
    {"label": "Claude3_Anthropic_Market", "model_id": "claude-3-sonnet-20240229"},
    {"label": "GeminiPro_Google_Market", "model_id": "models/gemini-pro"},
    {"label": "CommandRPlus_Cohere_Market", "model_id": "command-r-plus"},
    {"label": "Mixtral_MistralAI_Market", "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1"}
]

FIXED_SCORING_RUBRIC = """Score 1-5 (5 is best):
5 (Excellent): Clearly and accurately lists 3-4 key preventative measures with concise, easy-to-understand explanations for each. Advice is practical and widely applicable.
4 (Good): Lists 2-3 key measures accurately with good explanations, or 3-4 measures with slightly less clarity or a minor omission. Language is generally clear.
3 (Fair): Lists 2 key measures, or more but with some explanations being vague, slightly inaccurate, or missing. Information is mostly correct but could be more actionable or complete.
2 (Poor): Lists only 1 key measure, or the measures listed are not primary prevention strategies, or explanations are unclear/incorrect.
1 (Very Poor): Information is largely inaccurate, irrelevant, unhelpful, or provides potentially harmful advice.
"""

if 'llm_configs_master_list' not in st.session_state:
    st.session_state.llm_configs_master_list = PREDEFINED_LLMS

if 'selected_llms_status' not in st.session_state:
    st.session_state.selected_llms_status = {llm['label']: True for llm in st.session_state.llm_configs_master_list}

if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"

if 'evaluation_mode' not in st.session_state:
    st.session_state.evaluation_mode = "Pointwise"

# New state variable for selected computation-based metrics
if 'selected_computation_metrics' not in st.session_state:
    st.session_state.selected_computation_metrics = []

if 'test_cases' not in st.session_state:
    st.session_state.test_cases = []

if 'test_case_counter' not in st.session_state:
    st.session_state.test_case_counter = 0

if 'initial_responses' not in st.session_state:
    st.session_state.initial_responses = {}

if 'evaluations' not in st.session_state:
    st.session_state.evaluations = {}

# New state variable for computation-based metric results
if 'computation_results' not in st.session_state:
    st.session_state.computation_results = {}


def get_active_llms():
    active_llms = []
    if 'llm_configs_master_list' in st.session_state and 'selected_llms_status' in st.session_state:
        for llm_config in st.session_state.llm_configs_master_list:
            if st.session_state.selected_llms_status.get(llm_config['label'], False):
                active_llms.append(llm_config)
    return active_llms

def generate_wiki_query_from_prompt(prompt_text: str, max_words: int = 5) -> str:
    if not prompt_text:
        return ""
    words = prompt_text.split()
    query_words = [re.sub(r'\W+', '', word) for word in words[:max_words] if re.sub(r'\W+', '', word)]
    query = " ".join(query_words)
    return query.strip()

def fetch_wikipedia_criteria(query: str, lang: str = "en") -> str:
    if not query:
        return "No search query derived for ideal answer criteria."
    try:
        wikipedia.set_lang(lang[:2].lower())
        summary = wikipedia.summary(query, sentences=7)
        if not summary:
            return f"Wikipedia summary for '{query}' in language '{lang}' was empty. Page might exist but have no summary text or the query was too broad/specific."
        return summary
    except wikipedia.exceptions.PageError:
        return f"Wikipedia page not found for derived query: '{query}' in language '{lang}'."
    except wikipedia.exceptions.DisambiguationError as e:
        if e.options:
            first_option_query = e.options[0]
            st.info(f"Ambiguous query '{query}'. Trying first option: '{first_option_query}'...")
            return fetch_wikipedia_criteria(first_option_query, lang)
        return f"Ambiguous query '{query}' in language '{lang}'. No clear options found. Suggestions (first 3): {e.options[:3]}..."
    except Exception as e:
        return f"Error fetching Wikipedia data for derived query '{query}' in language '{lang}': {str(e)}"

# Identify Query Type
def identify_query_type(prompt: str) -> str:
    prompt_lower = prompt.lower()

    if any(keyword in prompt_lower for keyword in ["translate", "translation", "in hindi", "in english", "भाषांतर", "अनुवाद"]):
        return "Translation"
    if any(keyword in prompt_lower for keyword in ["hello", "hi", "how are you", "chat", "talk to me", "conversation", "can you help"]):
        return "Chat"
    if any(keyword in prompt_lower for keyword in ["what is", "how to", "explain", "describe", "define", "why is", "who is", "when is", "tell me about"]):
        return "Q&A"
    if "?" in prompt_lower:
        return "Q&A"

    return "General/Uncategorized"

with st.sidebar:
    st.header("⚙️ Setup Your Evaluation")
    st.info(f"Each test case will be run {NUM_ROUNDS} times.")

    st.subheader("Select Language")
    st.session_state.selected_language = st.radio(
        "Choose evaluation language (for prompts & Wikipedia search):",
        ("English", "Hindi"),
        key="language_select",
        horizontal=True,
    )
    st.caption(f"Current language: **{st.session_state.selected_language}**")

    st.divider()
    st.subheader("Select LLMs for Evaluation")
    if st.session_state.llm_configs_master_list:
        for llm in st.session_state.llm_configs_master_list:
            st.session_state.selected_llms_status[llm['label']] = st.checkbox(
                f"{llm['label']} (`{llm['model_id']}`)",
                value=st.session_state.selected_llms_status.get(llm['label'], True),
                key=f"select_llm_{llm['label']}"
            )
    else:
        st.info("No LLMs are currently listed.")

    st.divider()
    st.subheader("Choose Judge Model Evaluation Mode")
    st.session_state.evaluation_mode = st.radio(
        "Select how the judge model will evaluate responses:",
        ("Pointwise", "Pairwise"),
        key="evaluation_mode_select",
        help="Pointwise: Judge scores one output (0-5). Pairwise: Judge compares two outputs and picks a winner."
    )
    st.caption(f"Current evaluation mode: **{st.session_state.evaluation_mode}**")

    st.divider()
    st.subheader("Computation-Based Metrics")
    st.caption("Select automated metrics to compare LLM responses against reference criteria.")
    available_metrics = {
        "Exact Match": "Exact Match",
        "ROUGE (L, 1, 2)": "ROUGE",
        "BLEU": "BLEU",
        "Cosine Similarity": "Cosine Similarity",
        # "Accuracy (needs explicit labels)": "Accuracy", # Deferring these for now
        # "F1-score (needs explicit labels)": "F1-score", # Deferring these for now
        # "Tool Match (needs tool labels)": "Tool Match" # Deferring these for now
    }
    selected_metrics_list = []
    for metric_label, metric_key in available_metrics.items():
        if st.checkbox(metric_label, key=f"comp_metric_{metric_key}", value=(metric_key in st.session_state.selected_computation_metrics)):
            selected_metrics_list.append(metric_key)
    st.session_state.selected_computation_metrics = selected_metrics_list
    st.caption(f"Selected: {', '.join(st.session_state.selected_computation_metrics) if st.session_state.selected_computation_metrics else 'None'}")


    st.divider()
    st.subheader("Define Your Test Cases")
    st.caption(f"Prompts should be in {st.session_state.selected_language}. Test Case ID is auto-generated. Ideal answer criteria will be auto-fetched from Wikipedia using the first few words of your prompt. Scoring rubric is fixed.")
    with st.form("test_case_form", clear_on_submit=True):
        tc_prompt = st.text_area("Full Prompt for LLM", height=150, placeholder=f"e.g., What are the common symptoms of a heart attack and how should one respond in an emergency? (in {st.session_state.selected_language})")

        add_tc_button = st.form_submit_button("Add Test Case")

        if add_tc_button:
            if tc_prompt:
                st.session_state.test_case_counter += 1
                generated_tc_id = f"TC{st.session_state.test_case_counter}"

                auto_derived_wiki_query = generate_wiki_query_from_prompt(tc_prompt)

                fetched_criteria = "No useful information fetched from Wikipedia."
                if auto_derived_wiki_query:
                    with st.spinner(f"Fetching Wikipedia summary for '{auto_derived_wiki_query}'..."):
                        wiki_lang_code = "hi" if st.session_state.selected_language == "Hindi" else "en"
                        fetched_criteria = fetch_wikipedia_criteria(auto_derived_wiki_query, lang=wiki_lang_code)

                query_type = identify_query_type(tc_prompt)

                st.session_state.test_cases.append({
                    "id": generated_tc_id,
                    "prompt": tc_prompt,
                    "wiki_query": auto_derived_wiki_query,
                    "ideal_answer_criteria": fetched_criteria, # This will serve as reference for computation metrics
                    "scoring_rubric": FIXED_SCORING_RUBRIC,
                    "language": st.session_state.selected_language,
                    "query_type": query_type
                })
                st.success(f"Test Case '{generated_tc_id}' ({st.session_state.selected_language}, Type: {query_type}) added!")
                st.info(f"Auto-derived Wikipedia search query: '{auto_derived_wiki_query}'.")
                if "Error" in fetched_criteria or "not found" in fetched_criteria or "No search query" in fetched_criteria or "was empty" in fetched_criteria:
                    st.warning(f"Wikipedia fetch result: {fetched_criteria}")
            else:
                st.warning("The 'Prompt for LLM' field is required.")

    if st.session_state.test_cases:
        st.markdown("##### Current Test Cases:")
        for i, tc in enumerate(st.session_state.test_cases):
            lang_display = f" ({tc.get('language', 'N/A')})"
            type_display = f" (Type: {tc.get('query_type', 'N/A')})"
            display_name_snippet = tc['prompt'][:40] + "..." if len(tc['prompt']) > 40 else tc['prompt']
            with st.expander(f"{tc['id']}: '{display_name_snippet}'{lang_display}{type_display}"):
                st.markdown(f"**Full Prompt:**\n```\n{tc['prompt']}\n```")
                st.markdown(f"**Detected Query Type:** `{tc.get('query_type', 'N/A')}`")
                st.markdown(f"**Auto-derived Wikipedia Search Query (from prompt):** `{tc.get('wiki_query', 'N/A')}`")
                st.markdown(f"**Fetched Ideal Answer/Criteria (from Wikipedia):**")
                st.info(tc.get('ideal_answer_criteria', 'Not fetched or N/A'))
                st.markdown(f"**Fixed Scoring Rubric:**\n```\n{tc['scoring_rubric']}\n```")
                if st.button(f"Remove##{tc['id']}", key=f"remove_tc_{tc['id']}", help="Remove this Test Case"):
                    st.session_state.test_cases.pop(i)
                    st.rerun()
    else:
        st.caption("Add some test cases to get started.")


def call_llm_api(model_id: str, model_label: str, prompt_text: str, language: str, is_evaluation: bool = False, round_num: int = 0) -> str:
    st.write(f"📞 Round {round_num}, Calling {model_label} (ID: {model_id}) for lang: {language}... (SIMULATED)")
    time.sleep(random.uniform(0.2, 0.5))

    if is_evaluation:
        if st.session_state.evaluation_mode == "Pointwise":
            simulated_score = str(random.randint(1, 5))
            st.write(f"✔️ R{round_num}, Simulated pointwise score from {model_label}: {simulated_score}")
            return simulated_score
        else: # Pairwise simulation
            simulated_winner_options = [model_label, "Other_LLM_Simulated", "Tie"] # This is just illustrative
            simulated_winner = random.choice(simulated_winner_options)
            st.write(f"✔️ R{round_num}, Simulated pairwise outcome from {model_label}: {simulated_winner}")
            return simulated_winner
    else:
        if language == "Hindi":
            simulated_response = (
                f"राउंड {round_num}: यह **{model_label}** से एक **नकली** विस्तृत प्रतिक्रिया है "
                f"'{prompt_text[:60]}...' प्रॉम्प्ट के लिए **हिंदी** में।\n"
                f"विविधता के लिए यह प्रतिक्रिया थोड़ी अलग हो सकती है।"
            )
        else:
            simulated_response = (
                f"Round {round_num}: This is a **simulated** detailed response from **{model_label}** "
                f"for the prompt: '{prompt_text[:60]}...' in **English**.\n"
                f"This response might differ slightly for variety in round {round_num}."
            )
        st.write(f"✔️ R{round_num}, Simulated response generated by {model_label} in {language}.")
        return simulated_response

def generate_response_for_test_case_st(
    llm_label: str,
    model_id: str,
    test_case_prompt: str,
    language: str,
    round_num: int
) -> tuple[str, str, str, int]:
    response_text = call_llm_api(model_id, llm_label, test_case_prompt, language, is_evaluation=False, round_num=round_num)
    return llm_label, test_case_prompt, response_text, round_num

def evaluate_response_st(
    evaluator_llm_label: str,
    evaluator_model_id: str,
    test_case_obj: dict,
    target_llm_label_A: str,
    response_to_evaluate_A: str,
    target_llm_label_B: str = None,
    response_to_evaluate_B: str = None,
    round_num: int = 0,
    evaluation_mode: str = "Pointwise"
):
    if evaluation_mode == "Pointwise":
        evaluation_full_prompt = f"""You are an expert evaluator. This is for Round {round_num}.
The original prompt was in {test_case_obj.get('language', 'the specified language')}.
The response you are evaluating should also be in {test_case_obj.get('language', 'the specified language')}.
The detected query type for this prompt is: {test_case_obj.get('query_type', 'N/A')}.

Original Test Case Prompt:
{test_case_obj['prompt']}

Reference Information / Key Criteria (from Wikipedia, auto-queried for: '{test_case_obj.get('wiki_query', 'N/A')}'):
--- BEGIN REFERENCE INFO ---
{test_case_obj['ideal_answer_criteria']}
--- END REFERENCE INFO ---

Scoring Rubric:
{test_case_obj['scoring_rubric']}

Response to Evaluate (from LLM: {target_llm_label_A}, for Round {round_num}):
--- BEGIN RESPONSE ---
{response_to_evaluate_A}
--- END RESPONSE ---

Based on all the above, assess the 'Response to Evaluate' against the 'Reference Information / Key Criteria' using the 'Scoring Rubric'.
Consider accuracy, completeness, relevance, and adherence to the language.
Output ONLY a single integer score from 1 to 5 (e.g., 4).
"""
        score_text = call_llm_api(evaluator_model_id, evaluator_llm_label, evaluation_full_prompt, "English", is_evaluation=True, round_num=round_num)

        parsed_score = 0
        try:
            extracted_digits = "".join(filter(str.isdigit, score_text))
            if extracted_digits:
                first_digit_char = next((char for char in extracted_digits if '1' <= char <= '5'), None)
                if first_digit_char:
                    parsed_score = int(first_digit_char)
        except ValueError:
            pass
        return evaluator_llm_label, test_case_obj['prompt'], target_llm_label_A, parsed_score, round_num

    elif evaluation_mode == "Pairwise":
        if target_llm_label_B is None or response_to_evaluate_B is None:
            st.error("Error: Pairwise evaluation requires two responses for comparison.")
            return evaluator_llm_label, test_case_obj['prompt'], "PAIRWISE_ERROR", "ERROR", "ERROR", round_num

        evaluation_full_prompt = f"""You are an expert evaluator. This is for Round {round_num}.
The original prompt was in {test_case_obj.get('language', 'the specified language')}.
You need to compare two responses provided below for the same prompt.
The detected query type for this prompt is: {test_case_obj.get('query_type', 'N/A')}.

Original Test Case Prompt:
{test_case_obj['prompt']}

Reference Information / Key Criteria (from Wikipedia, auto-queried for: '{test_case_obj.get('wiki_query', 'N/A')}'):
--- BEGIN REFERENCE INFO ---
{test_case_obj['ideal_answer_criteria']}
--- END REFERENCE INFO ---

Response A (from LLM: {target_llm_label_A}, for Round {round_num}):
--- BEGIN RESPONSE A ---
{response_to_evaluate_A}
--- END RESPONSE A ---

Response B (from LLM: {target_llm_label_B}, for Round {round_num}):
--- BEGIN RESPONSE B ---
{response_to_evaluate_B}
--- END RESPONSE B ---

Based on the original prompt, reference information, and the quality of Response A and Response B, which response is better?
Consider accuracy, completeness, relevance, and adherence to the language.
Output ONLY one of the following: "{target_llm_label_A}", "{target_llm_label_B}", or "Tie".
"""
        simulated_winner = call_llm_api(evaluator_model_id, evaluator_llm_label, evaluation_full_prompt, "English", is_evaluation=True, round_num=round_num)
        if simulated_winner not in [target_llm_label_A, target_llm_label_B, "Tie"]:
            simulated_winner = random.choice([target_llm_label_A, target_llm_label_B, "Tie"])

        return evaluator_llm_label, test_case_obj['prompt'], target_llm_label_A, target_llm_label_B, simulated_winner, round_num

# --- Computation-Based Metric Functions ---
def calculate_exact_match(reference: str, candidate: str) -> float:
    """Calculates Exact Match score."""
    return 1.0 if reference.strip().lower() == candidate.strip().lower() else 0.0

def calculate_rouge(reference: str, candidate: str) -> dict:
    """Calculates ROUGE-L, ROUGE-1, ROUGE-2 scores."""
    if not rouge_scorer:
        return {"rouge-1": {"fmeasure": 0.0}, "rouge-2": {"fmeasure": 0.0}, "rouge-l": {"fmeasure": 0.0}}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge-1_f": round(scores["rouge1"].fmeasure, 4),
        "rouge-2_f": round(scores["rouge2"].fmeasure, 4),
        "rouge-l_f": round(scores["rougeL"].fmeasure, 4)
    }

def calculate_bleu(reference: str, candidate: str) -> float:
    """Calculates BLEU score."""
    if not nltk or not sentence_bleu:
        return 0.0
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    # Smoothing function 4 is commonly used for short sentences
    smoothie = SmoothingFunction().method4
    score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
    return round(score, 4)

def get_sentence_transformer_model():
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        try:
            # Using a general-purpose model suitable for various languages
            _sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Failed to load SentenceTransformer model: {e}")
            _sentence_transformer_model = None
    return _sentence_transformer_model

def calculate_cosine_similarity(reference: str, candidate: str) -> float:
    """Calculates Cosine Similarity using SentenceTransformers."""
    model = get_sentence_transformer_model()
    if not model or not util:
        return 0.0
    if not reference or not candidate: # Handle empty strings
        return 0.0

    try:
        embeddings1 = model.encode(reference, convert_to_tensor=True)
        embeddings2 = model.encode(candidate, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        return round(cosine_scores.item(), 4)
    except Exception as e:
        st.error(f"Error during cosine similarity calculation: {e}")
        return 0.0

def run_computation_metrics(llm_response: str, ideal_answer: str, selected_metrics: list) -> dict:
    """Runs selected computation-based metrics."""
    results = {}
    if "Exact Match" in selected_metrics:
        results["Exact Match"] = calculate_exact_match(ideal_answer, llm_response)
    if "ROUGE" in selected_metrics:
        rouge_scores = calculate_rouge(ideal_answer, llm_response)
        results.update(rouge_scores)
    if "BLEU" in selected_metrics:
        results["BLEU"] = calculate_bleu(ideal_answer, llm_response)
    if "Cosine Similarity" in selected_metrics:
        results["Cosine Similarity"] = calculate_cosine_similarity(ideal_answer, llm_response)
    # Add other metrics here if implemented
    return results

# --- End Computation-Based Metric Functions ---


st.divider()

active_llms_for_display = get_active_llms()

if not active_llms_for_display:
    st.warning("👈 Please select at least one LLM in the sidebar to proceed with evaluation.")
elif not st.session_state.test_cases:
    st.info("👈 Add at least one Test Case in the sidebar to begin.")
else:
    st.header("Phase 1: Get LLM Responses")
    st.caption(f"Click to get responses from selected LLMs for each test case, repeated for {NUM_ROUNDS} rounds.")

    if st.button(f"Get All LLM Responses ({NUM_ROUNDS} Rounds)", key="run_phase1_button"):
        active_llms_for_run = get_active_llms()
        if not active_llms_for_run:
            st.warning("No LLMs selected. Please select LLMs in the sidebar.")
        else:
            st.session_state.initial_responses = {tc["id"]: {f"Round{r+1}": {} for r in range(NUM_ROUNDS)} for tc in st.session_state.test_cases}
            st.session_state.evaluations = {} # Clear judge evaluations when re-running responses
            st.session_state.computation_results = {} # Clear computation results

            with st.spinner(f"Getting responses ({NUM_ROUNDS} rounds each) from {len(active_llms_for_run)} LLM(s) for {len(st.session_state.test_cases)} test case(s)..."):
                prompt_to_id_map = {tc["prompt"]: tc["id"] for tc in st.session_state.test_cases}

                for r_idx in range(NUM_ROUNDS):
                    round_num_str = f"Round{r_idx + 1}"
                    st.write(f"--- Starting {round_num_str} for Initial Responses ---")
                    for tc in st.session_state.test_cases:
                        current_tc_lang = tc.get('language', st.session_state.selected_language)
                        st.write(f"Processing Test Case: {tc['id']} ({round_num_str}, Language: {current_tc_lang}, Type: {tc.get('query_type', 'N/A')})")
                        for llm_config in active_llms_for_run:
                            try:
                                llm_label, tc_prompt, response_text, _ = generate_response_for_test_case_st(
                                    llm_label=llm_config["label"],
                                    model_id=llm_config["model_id"],
                                    test_case_prompt=tc["prompt"],
                                    language=current_tc_lang,
                                    round_num=r_idx + 1
                                )
                                tc_id_from_map = prompt_to_id_map.get(tc_prompt)
                                if tc_id_from_map:
                                    st.session_state.initial_responses[tc_id_from_map][round_num_str][llm_label] = response_text
                                else:
                                    st.error(f"Mapping error for prompt: {tc_prompt[:30]}...")
                            except Exception as e:
                                st.error(f"Error getting response from {llm_config['label']} for TC {tc['id']} ({round_num_str}): {e}")
                st.success(f"Finished getting all initial responses for {NUM_ROUNDS} rounds!")
                st.rerun()

    if st.session_state.initial_responses:
        st.subheader("📜 Review Initial LLM Responses (Multi-Round)")
        if st.checkbox("Show raw initial_responses data (for debugging)", key="debug_initial_responses"):
            st.json(st.session_state.initial_responses)

        active_llms_labels_for_display = [llm['label'] for llm in get_active_llms()]
        for tc_obj in st.session_state.test_cases:
            tc_id = tc_obj["id"]
            tc_lang = tc_obj.get('language', 'N/A')
            query_type_display = tc_obj.get('query_type', 'N/A')
            wiki_query_display = tc_obj.get('wiki_query', 'N/A')
            display_name_snippet = tc_obj['prompt'][:40] + "..." if len(tc_obj['prompt']) > 40 else tc_obj['prompt']

            if tc_id in st.session_state.initial_responses:
                with st.expander(f"Responses for: {tc_id} ('{display_name_snippet}') ({tc_lang}, Type: {query_type_display}) - {NUM_ROUNDS} Rounds", expanded=False):
                    st.markdown(f"**Full Prompt for {tc_id} (Language: {tc_lang}, Type: {query_type_display}):**")
                    st.code(tc_obj['prompt'], language=None)
                    st.markdown(f"**Reference Criteria (auto-derived from Wikipedia for query: `{wiki_query_display}`):**")
                    st.info(tc_obj.get('ideal_answer_criteria', 'N/A'))
                    st.markdown(f"**Fixed Scoring Rubric:**\n```\n{tc_obj['scoring_rubric']}\n```")
                    st.markdown("---")

                    for r_idx in range(NUM_ROUNDS):
                        round_num_str = f"Round{r_idx + 1}"
                        if round_num_str in st.session_state.initial_responses[tc_id]:
                            st.markdown(f"##### {round_num_str} Responses:")
                            responses_this_round = st.session_state.initial_responses[tc_id][round_num_str]
                            responses_to_show_this_round = {
                                llm_label: response
                                for llm_label, response in responses_this_round.items()
                                if llm_label in active_llms_labels_for_display
                            }
                            if responses_to_show_this_round:
                                for llm_label, response_text in responses_to_show_this_round.items():
                                    st.markdown(f"**LLM: {llm_label}**")
                                    st.code(response_text, language=None)
                                    st.markdown("---")
                            else:
                                st.caption(f"No responses from selected LLMs for {round_num_str}.")
                        else:
                            st.caption(f"No data for {round_num_str}.")
    st.divider()

    if st.session_state.initial_responses:
        st.header("Phase 2: LLMs Evaluate Each Other (Multi-Round)")
        st.caption(f"Click to have selected LLMs score each other's responses for each of the {NUM_ROUNDS} rounds using **{st.session_state.evaluation_mode}** mode.")
        if st.button(f"Start Peer Evaluation ({NUM_ROUNDS} Rounds) in {st.session_state.evaluation_mode} Mode", key="run_phase2_button"):
            active_llms_for_run = get_active_llms()
            if not active_llms_for_run:
                st.warning("Please select at least one LLM in the sidebar for evaluation.")
            elif st.session_state.evaluation_mode == "Pairwise" and len(active_llms_for_run) < 2:
                st.warning("For Pairwise evaluation, please select at least two LLMs in the sidebar.")
            else:
                if st.session_state.evaluation_mode == "Pointwise":
                    st.session_state.evaluations = {tc["id"]: {f"Round{r+1}": {} for r in range(NUM_ROUNDS)} for tc in st.session_state.test_cases}
                elif st.session_state.evaluation_mode == "Pairwise":
                    st.session_state.evaluations = {tc["id"]: {f"Round{r+1}": {} for r in range(NUM_ROUNDS)} for tc in st.session_state.test_cases}


                with st.spinner(f"Selected LLMs are evaluating responses ({NUM_ROUNDS} rounds) in {st.session_state.evaluation_mode} mode..."):
                    prompt_to_id_map_eval = {tc["prompt"]: tc["id"] for tc in st.session_state.test_cases}

                    for r_idx in range(NUM_ROUNDS):
                        round_num_str = f"Round{r_idx + 1}"
                        st.write(f"--- Starting Peer Evaluation for {round_num_str} in {st.session_state.evaluation_mode} Mode ---")
                        for tc_obj in st.session_state.test_cases:
                            tc_id = tc_obj["id"]
                            st.write(f"Evaluating for TC: {tc_id}, {round_num_str} (Ref: '{tc_obj.get('wiki_query','N/A')}', Type: {tc_obj.get('query_type', 'N/A')})")

                            if tc_id not in st.session_state.initial_responses or \
                               round_num_str not in st.session_state.initial_responses[tc_id] or \
                               not st.session_state.initial_responses[tc_id][round_num_str]:
                                st.info(f"No initial responses for {tc_id}, {round_num_str} to evaluate.")
                                continue

                            responses_for_tc_active_this_round = {
                                llm_label: resp
                                for llm_label, resp in st.session_state.initial_responses[tc_id][round_num_str].items()
                                if llm_label in [llm['label'] for llm in active_llms_for_run]
                            }

                            if not responses_for_tc_active_this_round:
                                st.info(f"No initial responses from selected LLMs for {tc_id}, {round_num_str} to evaluate.")
                                continue

                            st.session_state.evaluations[tc_id].setdefault(round_num_str, {})

                            if st.session_state.evaluation_mode == "Pointwise":
                                for llm_being_evaluated_label, response_text in responses_for_tc_active_this_round.items():
                                    st.session_state.evaluations[tc_id][round_num_str].setdefault(llm_being_evaluated_label, {})
                                    if response_text is None or "Error:" in str(response_text) or "SIMULATED FALLBACK" in response_text :
                                        st.info(f"Skipping eval of {llm_being_evaluated_label} for {tc_id}, {round_num_str} (bad response).")
                                        continue

                                    for evaluator_config in active_llms_for_run:
                                        try:
                                            evaluator_llm_label_res, tc_prompt_key_from_eval, evaluated_llm_label_from_eval, score, _ = evaluate_response_st(
                                                evaluator_config["label"], evaluator_config["model_id"],
                                                tc_obj,
                                                llm_being_evaluated_label, response_text,
                                                round_num=r_idx + 1,
                                                evaluation_mode="Pointwise"
                                            )
                                            tc_id_for_eval = prompt_to_id_map_eval.get(tc_prompt_key_from_eval)
                                            if tc_id_for_eval:
                                                st.session_state.evaluations[tc_id_for_eval][round_num_str].setdefault(evaluated_llm_label_from_eval, {})[evaluator_llm_label_res] = score
                                            else:
                                                st.error(f"Eval mapping error for prompt: {tc_prompt_key_from_eval[:30]}...")
                                        except Exception as e:
                                            st.error(f"Error during pointwise eval by {evaluator_config['label']} for {llm_being_evaluated_label} on TC {tc_id}, {round_num_str}: {e}")

                            elif st.session_state.evaluation_mode == "Pairwise":
                                llm_labels_to_compare = list(responses_for_tc_active_this_round.keys())
                                for llm_label_A, llm_label_B in itertools.combinations(llm_labels_to_compare, 2):
                                    response_A = responses_for_tc_active_this_round.get(llm_label_A)
                                    response_B = responses_for_tc_active_this_round.get(llm_label_B)

                                    if response_A is None or response_B is None or "Error:" in str(response_A) or "Error:" in str(response_B):
                                        st.info(f"Skipping pairwise eval for {llm_label_A} vs {llm_label_B} (bad response).")
                                        continue

                                    comparison_key = f"{llm_label_A}_vs_{llm_label_B}"
                                    st.session_state.evaluations[tc_id][round_num_str].setdefault(comparison_key, {})

                                    for evaluator_config in active_llms_for_run:
                                        try:
                                            evaluator_llm_label_res, _, _, _, winner_label, _ = evaluate_response_st(
                                                evaluator_config["label"], evaluator_config["model_id"],
                                                tc_obj,
                                                llm_label_A, response_A,
                                                llm_label_B, response_B,
                                                round_num=r_idx + 1,
                                                evaluation_mode="Pairwise"
                                            )
                                            st.session_state.evaluations[tc_id][round_num_str][comparison_key][evaluator_llm_label_res] = winner_label
                                        except Exception as e:
                                            st.error(f"Error during pairwise eval by {evaluator_config['label']} for {llm_label_A} vs {llm_label_B} on TC {tc_id}, {round_num_str}: {e}")

                st.success(f"Evaluation for {NUM_ROUNDS} rounds finished in {st.session_state.evaluation_mode} mode!")
                st.rerun()

    if st.session_state.evaluations:
        st.subheader(f"📊 Review Evaluation Results ({st.session_state.evaluation_mode} Mode)")
        if st.checkbox("Show raw evaluation data (for debugging)", key="debug_eval_data"):
            st.json(st.session_state.evaluations)

        active_llms_for_matrix_stats = get_active_llms()
        active_llm_labels_for_matrix_stats = [llm['label'] for llm in active_llms_for_matrix_stats]

        if active_llm_labels_for_matrix_stats:
            for tc_obj in st.session_state.test_cases:
                tc_id = tc_obj["id"]
                tc_lang = tc_obj.get('language', 'N/A')
                query_type_display = tc_obj.get('query_type', 'N/A')
                wiki_query_display = tc_obj.get('wiki_query', 'N/A')
                display_name_snippet = tc_obj['prompt'][:40] + "..." if len(tc_obj['prompt']) > 40 else tc_obj['prompt']

                if tc_id in st.session_state.evaluations and st.session_state.evaluations[tc_id]:
                    with st.expander(f"Results for: {tc_id} - '{display_name_snippet}' ({tc_lang}, Type: {query_type_display}) (Ref Query: '{wiki_query_display}') - {st.session_state.evaluation_mode} Mode", expanded=False):
                        st.markdown(f"**Prompt for {tc_id} (Language: {tc_lang}, Type: {query_type_display}):**")
                        st.code(tc_obj['prompt'], language=None)
                        st.markdown(f"**Reference Criteria (auto-derived from Wikipedia for query: `{wiki_query_display}`):**")
                        st.info(tc_obj.get('ideal_answer_criteria', 'N/A'))
                        st.markdown(f"**Fixed Scoring Rubric:**\n```\n{tc_obj['scoring_rubric']}\n```")
                        st.markdown("---")

                        if st.session_state.evaluation_mode == "Pointwise":
                            st.markdown("##### Pointwise Score Matrices (Per Round)")
                            for r_idx in range(NUM_ROUNDS):
                                round_num_str = f"Round{r_idx + 1}"
                                st.markdown(f"###### Score Matrix for {round_num_str}")
                                if round_num_str in st.session_state.evaluations[tc_id]:
                                    round_eval_data = st.session_state.evaluations[tc_id][round_num_str]

                                    filtered_round_eval_data = {
                                        eval_llm: {
                                            evaluator: score
                                            for evaluator, score in scores.items()
                                            if evaluator in active_llm_labels_for_matrix_stats
                                        }
                                        for eval_llm, scores in round_eval_data.items()
                                        if eval_llm in active_llm_labels_for_matrix_stats
                                    }

                                    if filtered_round_eval_data:
                                        matrix_display_data_round = {}
                                        for llm_evaluated_label in active_llm_labels_for_matrix_stats:
                                            row_scores = {
                                                eval_llm_label_col: filtered_round_eval_data.get(llm_evaluated_label, {}).get(eval_llm_label_col, "-")
                                                for eval_llm_label_col in active_llm_labels_for_matrix_stats
                                            }
                                            matrix_display_data_round[llm_evaluated_label] = row_scores

                                        try:
                                            if matrix_display_data_round:
                                                df_round = pd.DataFrame.from_dict(matrix_display_data_round, orient='index', columns=active_llm_labels_for_matrix_stats)
                                                st.dataframe(df_round)
                                            else:
                                                st.caption(f"No scores from active LLMs for {round_num_str} to display.")
                                        except Exception as e:
                                            st.error(f"Error creating DataFrame for TC {tc_id}, {round_num_str}: {e}")
                                            st.json(matrix_display_data_round)
                                    else:
                                        st.caption(f"No evaluation data from selected LLMs for {round_num_str}.")
                                else:
                                    st.caption(f"No evaluation data for {round_num_str}.")
                                st.markdown("---")

                            st.markdown("##### Pointwise Score Statistics Across Rounds (Average Peer Score per Round)")
                            llm_round_avg_scores_data = {}

                            for llm_evaluated_label in active_llm_labels_for_matrix_stats:
                                llm_round_avg_scores_data[llm_evaluated_label] = []
                                for r_idx in range(NUM_ROUNDS):
                                    round_num_str = f"Round{r_idx + 1}"
                                    if round_num_str in st.session_state.evaluations[tc_id] and \
                                       llm_evaluated_label in st.session_state.evaluations[tc_id][round_num_str]:

                                        scores_from_peers_this_round = st.session_state.evaluations[tc_id][round_num_str][llm_evaluated_label]
                                        valid_scores_this_round = [
                                            s for evaluator, s in scores_from_peers_this_round.items()
                                            if evaluator in active_llm_labels_for_matrix_stats and evaluator != llm_evaluated_label and isinstance(s, int) and 1 <= s <= 5
                                        ]
                                        if valid_scores_this_round:
                                            avg_score_this_round = sum(valid_scores_this_round) / len(valid_scores_this_round)
                                            llm_round_avg_scores_data[llm_evaluated_label].append(round(avg_score_this_round,2))
                                        else:
                                            llm_round_avg_scores_data[llm_evaluated_label].append(None)
                                    else:
                                        llm_round_avg_scores_data[llm_evaluated_label].append(None)

                            stats_display_list = []
                            for llm_label, round_avgs in llm_round_avg_scores_data.items():
                                valid_round_avgs = [s for s in round_avgs if s is not None]
                                if valid_round_avgs:
                                    mean_score = round(statistics.mean(valid_round_avgs), 2)
                                    min_score = round(min(valid_round_avgs), 2)
                                    max_score = round(max(valid_round_avgs), 2)
                                    std_dev = round(statistics.stdev(valid_round_avgs), 2) if len(valid_round_avgs) > 1 else 0.00
                                    stats_display_list.append({
                                        "LLM": llm_label,
                                        "Mean of Round Avgs": mean_score,
                                        "Min Round Avg": min_score,
                                        "Max Round Avg": max_score,
                                        "StdDev of Round Avgs": std_dev,
                                        "Round Avg Scores": ", ".join(map(str, [f"{s:.2f}" if s is not None else "N/A" for s in round_avgs]))
                                    })

                            if stats_display_list:
                                stats_df = pd.DataFrame(stats_display_list)
                                st.dataframe(stats_df.set_index("LLM"))
                            else:
                                st.info("No pointwise evaluation statistics to display for active LLMs on this test case.")

                        elif st.session_state.evaluation_mode == "Pairwise":
                            st.markdown("##### Pairwise Evaluation Results (Per Round)")
                            for r_idx in range(NUM_ROUNDS):
                                round_num_str = f"Round{r_idx + 1}"
                                st.markdown(f"###### Comparison Results for {round_num_str}")
                                if round_num_str in st.session_state.evaluations[tc_id]:
                                    round_eval_data = st.session_state.evaluations[tc_id][round_num_str]
                                    pairwise_results_this_round = {
                                        comp_key: {
                                            evaluator: winner
                                            for evaluator, winner in results.items()
                                            if evaluator in active_llm_labels_for_matrix_stats
                                        }
                                        for comp_key, results in round_eval_data.items()
                                        if "_vs_" in comp_key
                                    }

                                    if pairwise_results_this_round:
                                        pairwise_df_data = []
                                        for comp_key, eval_results in pairwise_results_this_round.items():
                                            llm_a, llm_b = comp_key.split("_vs_")
                                            row = {"Comparison": f"{llm_a} vs {llm_b}"}
                                            for evaluator_label in active_llm_labels_for_matrix_stats:
                                                row[evaluator_label] = eval_results.get(evaluator_label, "-")
                                            pairwise_df_data.append(row)

                                        if pairwise_df_data:
                                            df_pairwise = pd.DataFrame(pairwise_df_data)
                                            df_pairwise = df_pairwise.set_index("Comparison")
                                            st.dataframe(df_pairwise)
                                        else:
                                            st.caption(f"No pairwise comparison data from selected LLMs for {round_num_str}.")
                                    else:
                                        st.caption(f"No pairwise evaluation data for {round_num_str}.")
                                else:
                                    st.caption(f"No evaluation data for {round_num_str}.")
                                st.markdown("---")

                            st.markdown("##### Pairwise Win/Loss/Tie Summary Across Rounds")
                            pairwise_summary_data = {}
                            for llm_label in active_llm_labels_for_matrix_stats:
                                pairwise_summary_data[llm_label] = {"Wins": 0, "Losses": 0, "Ties": 0, "Total_Comparisons_Evaluated": 0}

                            for r_idx in range(NUM_ROUNDS):
                                round_num_str = f"Round{r_idx + 1}"
                                if round_num_str in st.session_state.evaluations[tc_id]:
                                    round_eval_data = st.session_state.evaluations[tc_id][round_num_str]
                                    for comp_key, eval_results_by_judge in round_eval_data.items():
                                        if "_vs_" in comp_key:
                                            llm_a, llm_b = comp_key.split("_vs_")
                                            for evaluator_label, winner_label in eval_results_by_judge.items():
                                                if evaluator_label in active_llm_labels_for_matrix_stats:
                                                    if winner_label == llm_a:
                                                        pairwise_summary_data[llm_a]["Wins"] += 1
                                                        pairwise_summary_data[llm_b]["Losses"] += 1
                                                    elif winner_label == llm_b:
                                                        pairwise_summary_data[llm_b]["Wins"] += 1
                                                        pairwise_summary_data[llm_a]["Losses"] += 1
                                                    elif winner_label == "Tie":
                                                        pairwise_summary_data[llm_a]["Ties"] += 1
                                                        pairwise_summary_data[llm_b]["Ties"] += 1
                                                    pairwise_summary_data[llm_a]["Total_Comparisons_Evaluated"] += 1
                                                    pairwise_summary_data[llm_b]["Total_Comparisons_Evaluated"] += 1

                            summary_display_list = []
                            for llm_label, data in pairwise_summary_data.items():
                                if data["Total_Comparisons_Evaluated"] > 0:
                                    win_rate = f"{data['Wins'] / data['Total_Comparisons_Evaluated'] * 100:.2f}%"
                                    loss_rate = f"{data['Losses'] / data['Total_Comparisons_Evaluated'] * 100:.2f}%"
                                    tie_rate = f"{data['Ties'] / data['Total_Comparisons_Evaluated'] * 100:.2f}%"
                                    summary_display_list.append({
                                        "LLM": llm_label,
                                        "Wins": data["Wins"],
                                        "Losses": data["Losses"],
                                        "Ties": data["Ties"],
                                        "Win Rate": win_rate,
                                        "Loss Rate": loss_rate,
                                        "Tie Rate": tie_rate,
                                        "Total Comparisons Evaluated": data["Total_Comparisons_Evaluated"]
                                    })
                            if summary_display_list:
                                sorted_pairwise_summary = sorted(
                                    summary_display_list,
                                    key=lambda item: float(item['Win Rate'].strip('%')),
                                    reverse=True
                                )
                                summary_df = pd.DataFrame(sorted_pairwise_summary)
                                st.dataframe(summary_df.set_index("LLM"))
                            else:
                                st.info("No final pairwise summary to display.")
        else:
            st.info("No LLMs selected to display evaluation results.")

    st.divider()

    if st.session_state.initial_responses and st.session_state.selected_computation_metrics:
        st.header("Phase 3: Computation-Based Metrics Evaluation")
        st.caption(f"Click to compute selected metrics based on LLM responses and Wikipedia reference criteria.")

        if st.button("Run Computation-Based Metrics", key="run_computation_metrics_button"):
            active_llms_for_comp = get_active_llms()
            if not active_llms_for_comp:
                st.warning("Please select at least one LLM in the sidebar.")
            elif not st.session_state.selected_computation_metrics:
                st.warning("Please select at least one Computation-Based Metric in the sidebar.")
            else:
                st.session_state.computation_results = {tc["id"]: {f"Round{r+1}": {} for r in range(NUM_ROUNDS)} for tc in st.session_state.test_cases}

                with st.spinner("Calculating computation-based metrics... This might take a moment for embedding models."):
                    for r_idx in range(NUM_ROUNDS):
                        round_num_str = f"Round{r_idx + 1}"
                        st.write(f"--- Calculating Computation Metrics for {round_num_str} ---")
                        for tc_obj in st.session_state.test_cases:
                            tc_id = tc_obj["id"]
                            ideal_answer_text = tc_obj['ideal_answer_criteria']

                            if not ideal_answer_text or "No useful information" in ideal_answer_text:
                                st.warning(f"Skipping computation metrics for TC {tc_id} ({round_num_str}): No valid ideal answer/criteria found. Please ensure Wikipedia fetching was successful.")
                                continue

                            if tc_id not in st.session_state.initial_responses or \
                               round_num_str not in st.session_state.initial_responses[tc_id] or \
                               not st.session_state.initial_responses[tc_id][round_num_str]:
                                st.info(f"No initial responses for {tc_id}, {round_num_str} to run computation metrics.")
                                continue

                            responses_for_tc_active_this_round = {
                                llm_label: resp
                                for llm_label, resp in st.session_state.initial_responses[tc_id][round_num_str].items()
                                if llm_label in [llm['label'] for llm in active_llms_for_comp]
                            }

                            if not responses_for_tc_active_this_round:
                                st.info(f"No active LLM responses for {tc_id}, {round_num_str} to calculate metrics.")
                                continue

                            st.session_state.computation_results[tc_id].setdefault(round_num_str, {})

                            for llm_label, response_text in responses_for_tc_active_this_round.items():
                                if response_text is None or "Error:" in str(response_text) or "SIMULATED FALLBACK" in response_text:
                                    st.info(f"Skipping computation metrics for {llm_label} on TC {tc_id}, {round_num_str} (bad response).")
                                    continue
                                try:
                                    metric_scores = run_computation_metrics(response_text, ideal_answer_text, st.session_state.selected_computation_metrics)
                                    st.session_state.computation_results[tc_id][round_num_str][llm_label] = metric_scores
                                    st.write(f"✔️ Calculated metrics for {llm_label} on TC {tc_id}, {round_num_str}.")
                                except Exception as e:
                                    st.error(f"Error calculating metrics for {llm_label} on TC {tc_id}, {round_num_str}: {e}")
                st.success("Computation-based metrics calculated!")
                st.rerun()

        if st.session_state.computation_results:
            st.subheader("📈 Computation-Based Metric Results")
            if st.checkbox("Show raw computation_results data (for debugging)", key="debug_comp_results"):
                st.json(st.session_state.computation_results)

            active_llms_for_comp_display = get_active_llms()
            active_llm_labels_for_comp_display = [llm['label'] for llm in active_llms_for_comp_display]

            for tc_obj in st.session_state.test_cases:
                tc_id = tc_obj["id"]
                tc_lang = tc_obj.get('language', 'N/A')
                query_type_display = tc_obj.get('query_type', 'N/A')
                wiki_query_display = tc_obj.get('wiki_query', 'N/A')
                display_name_snippet = tc_obj['prompt'][:40] + "..." if len(tc_obj['prompt']) > 40 else tc_obj['prompt']

                if tc_id in st.session_state.computation_results and st.session_state.computation_results[tc_id]:
                    with st.expander(f"Computation Metrics for: {tc_id} - '{display_name_snippet}' ({tc_lang}, Type: {query_type_display})", expanded=False):
                        st.markdown(f"**Prompt for {tc_id} (Language: {tc_lang}, Type: {query_type_display}):**")
                        st.code(tc_obj['prompt'], language=None)
                        st.markdown(f"**Reference Criteria (auto-derived from Wikipedia for query: `{wiki_query_display}`):**")
                        st.info(tc_obj.get('ideal_answer_criteria', 'N/A'))
                        st.markdown("---")

                        all_metrics_to_display = st.session_state.selected_computation_metrics.copy()
                        # Add ROUGE sub-metrics if ROUGE is selected
                        if "ROUGE" in all_metrics_to_display:
                            all_metrics_to_display.remove("ROUGE")
                            all_metrics_to_display.extend(["rouge-1_f", "rouge-2_f", "rouge-l_f"])

                        st.markdown("##### Metric Scores Per Round")
                        for r_idx in range(NUM_ROUNDS):
                            round_num_str = f"Round{r_idx + 1}"
                            st.markdown(f"###### {round_num_str} Metric Scores:")
                            if round_num_str in st.session_state.computation_results[tc_id]:
                                round_comp_data = st.session_state.computation_results[tc_id][round_num_str]
                                display_data_round = []
                                for llm_label in active_llm_labels_for_comp_display:
                                    if llm_label in round_comp_data:
                                        row = {"LLM": llm_label}
                                        for metric in all_metrics_to_display:
                                            row[metric] = round_comp_data[llm_label].get(metric, "N/A")
                                        display_data_round.append(row)

                                if display_data_round:
                                    df_comp_round = pd.DataFrame(display_data_round).set_index("LLM")
                                    st.dataframe(df_comp_round)
                                else:
                                    st.caption(f"No computation metric data for active LLMs in {round_num_str}.")
                            else:
                                st.caption(f"No computation metric data for {round_num_str}.")
                            st.markdown("---")

                        st.markdown("##### Average Metric Scores Across All Rounds")
                        avg_metric_scores = {}
                        for llm_label in active_llm_labels_for_comp_display:
                            llm_metric_data = {metric: [] for metric in all_metrics_to_display}
                            for r_idx in range(NUM_ROUNDS):
                                round_num_str = f"Round{r_idx + 1}"
                                if round_num_str in st.session_state.computation_results[tc_id] and \
                                   llm_label in st.session_state.computation_results[tc_id][round_num_str]:
                                    for metric, score in st.session_state.computation_results[tc_id][round_num_str][llm_label].items():
                                        if isinstance(score, (int, float)):
                                            llm_metric_data[metric].append(score)

                            row_avg_data = {"LLM": llm_label}
                            for metric, scores_list in llm_metric_data.items():
                                if scores_list:
                                    row_avg_data[metric] = round(statistics.mean(scores_list), 4)
                                else:
                                    row_avg_data[metric] = "N/A"
                            avg_metric_scores[llm_label] = row_avg_data

                        if avg_metric_scores:
                            df_avg_metrics = pd.DataFrame(list(avg_metric_scores.values())).set_index("LLM")
                            st.dataframe(df_avg_metrics)
                        else:
                            st.info("No average computation metric scores to display for active LLMs on this test case.")
                else:
                    st.info(f"No computation metric results available for TC: {tc_id}.")
    st.divider()

    if st.session_state.evaluations or st.session_state.computation_results:
        st.header("Phase 4: Final Summary")
        st.caption("Overall performance based on selected evaluation methods.")

        active_llms_for_summary = get_active_llms()
        active_llm_labels_for_summary = [llm['label'] for llm in active_llms_for_summary]

        if not active_llm_labels_for_summary:
            st.info("No LLMs were selected for evaluation, so no summary can be generated.")
        else:
            if st.session_state.evaluations: # Summary for Model-Based Metrics
                if st.session_state.evaluation_mode == "Pointwise":
                    st.subheader("🏆 LLM Performance Ranking (Model-Based Pointwise - Mean of Round Averages)")
                    final_scores_summary = {}
                    for llm_label_evaluated in active_llm_labels_for_summary:
                        final_scores_summary[llm_label_evaluated] = {"overall_mean_of_means": "N/A", "mean_scores_by_tc": {}}
                        sum_of_tc_mean_scores = 0
                        num_valid_tc_for_llm = 0
 
                        for tc_obj in st.session_state.test_cases:
                            tc_id_str = tc_obj["id"]

                            round_averages_for_this_tc_llm = []
                            if tc_id_str in st.session_state.evaluations:
                                for r_idx in range(NUM_ROUNDS):
                                    round_num_str = f"Round{r_idx + 1}"
                                    if round_num_str in st.session_state.evaluations[tc_id_str] and \
                                       llm_label_evaluated in st.session_state.evaluations[tc_id_str][round_num_str]:

                                        scores_received_map = st.session_state.evaluations[tc_id_str][round_num_str][llm_label_evaluated]
                                        valid_scores_received = [
                                            s for evaluator, s in scores_received_map.items()
                                            if evaluator in active_llm_labels_for_summary and evaluator != llm_label_evaluated and isinstance(s, int) and 1 <= s <= 5
                                        ]
                                        if valid_scores_received:
                                            round_averages_for_this_tc_llm.append(sum(valid_scores_received) / len(valid_scores_received))

                        if round_averages_for_this_tc_llm:
                            mean_score_for_tc = round(statistics.mean(round_averages_for_this_tc_llm), 2)
                            final_scores_summary[llm_label_evaluated]["mean_scores_by_tc"][tc_id_str] = mean_score_for_tc
                            sum_of_tc_mean_scores += mean_score_for_tc
                            num_valid_tc_for_llm += 1
                        else:
                            final_scores_summary[llm_label_evaluated]["mean_scores_by_tc"][tc_id_str] = "N/A"

                    if num_valid_tc_for_llm > 0:
                        overall_mean = sum_of_tc_mean_scores / num_valid_tc_for_llm
                        final_scores_summary[llm_label_evaluated]["overall_mean_of_means"] = round(overall_mean, 2)

                    sorted_llms_summary = sorted(
                        final_scores_summary.items(),
                        key=lambda item: item[1]["overall_mean_of_means"] if isinstance(item[1]["overall_mean_of_means"], (int, float)) else -float('inf'),
                        reverse=True
                    )

                    for llm_label, data in sorted_llms_summary:
                        if llm_label not in active_llm_labels_for_summary:
                            continue

                        model_id_display = next((llm['model_id'] for llm in st.session_state.llm_configs_master_list if llm['label'] == llm_label), "N/A")
                        st.markdown(f"#### **{llm_label}** (Model ID: `{model_id_display}`)")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Overall Mean of Per-Test-Case Mean Scores: {data['overall_mean_of_means']}**")
                        with st.expander("View Mean Scores per Test Case (across rounds)", expanded=False):
                            if data["mean_scores_by_tc"]:
                                for tc_id_str, score in data["mean_scores_by_tc"].items():
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{tc_id_str}: {score}")
                            else:
                                st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;No test case scores available.")
                        st.markdown("---")

                elif st.session_state.evaluation_mode == "Pairwise":
                    st.subheader("🏆 LLM Performance Ranking (Model-Based Pairwise - Total Wins/Losses)")
                    final_pairwise_summary = {}
                    for llm_label in active_llm_labels_for_summary:
                        final_pairwise_summary[llm_label] = {"Wins": 0, "Losses": 0, "Ties": 0, "Total_Comparisons_Evaluated": 0}

                    for tc_obj in st.session_state.test_cases:
                        tc_id_str = tc_obj["id"]
                        if tc_id_str in st.session_state.evaluations:
                            for r_idx in range(NUM_ROUNDS):
                                round_num_str = f"Round{r_idx + 1}"
                                if round_num_str in st.session_state.evaluations[tc_id_str]:
                                    round_eval_data = st.session_state.evaluations[tc_id_str][round_num_str]
                                    for comp_key, eval_results_by_judge in round_eval_data.items():
                                        if "_vs_" in comp_key:
                                            llm_a, llm_b = comp_key.split("_vs_")
                                            for evaluator_label, winner_label in eval_results_by_judge.items():
                                                if evaluator_label in active_llm_labels_for_summary:
                                                    if winner_label == llm_a:
                                                        if llm_a in final_pairwise_summary: final_pairwise_summary[llm_a]["Wins"] += 1
                                                        if llm_b in final_pairwise_summary: final_pairwise_summary[llm_b]["Losses"] += 1
                                                    elif winner_label == llm_b:
                                                        if llm_b in final_pairwise_summary: final_pairwise_summary[llm_b]["Wins"] += 1
                                                        if llm_a in final_pairwise_summary: final_pairwise_summary[llm_a]["Losses"] += 1
                                                    elif winner_label == "Tie":
                                                        if llm_a in final_pairwise_summary: final_pairwise_summary[llm_a]["Ties"] += 1
                                                        if llm_b in final_pairwise_summary: final_pairwise_summary[llm_b]["Ties"] += 1
                                                    if llm_a in final_pairwise_summary: final_pairwise_summary[llm_a]["Total_Comparisons_Evaluated"] += 1
                                                    if llm_b in final_pairwise_summary: final_pairwise_summary[llm_b]["Total_Comparisons_Evaluated"] += 1

                    summary_display_list = []
                    for llm_label, data in final_pairwise_summary.items():
                        if data["Total_Comparisons_Evaluated"] > 0:
                            win_rate = f"{data['Wins'] / data['Total_Comparisons_Evaluated'] * 100:.2f}%"
                            loss_rate = f"{data['Losses'] / data['Total_Comparisons_Evaluated'] * 100:.2f}%"
                            tie_rate = f"{data['Ties'] / data['Total_Comparisons_Evaluated'] * 100:.2f}%"
                            summary_display_list.append({
                                "LLM": llm_label,
                                "Total Wins": data["Wins"],
                                "Total Losses": data["Losses"],
                                "Total Ties": data["Ties"],
                                "Win Rate": win_rate,
                                "Loss Rate": loss_rate,
                                "Tie Rate": tie_rate,
                                "Total Pairwise Comparisons": data["Total_Comparisons_Evaluated"]
                            })
                    if summary_display_list:
                        sorted_pairwise_summary = sorted(
                            summary_display_list,
                            key=lambda item: float(item['Win Rate'].strip('%')),
                            reverse=True
                        )
                        summary_df = pd.DataFrame(sorted_pairwise_list).set_index("LLM")
                        st.dataframe(summary_df)
                    else:
                        st.info("No final pairwise summary to display.")

            if st.session_state.computation_results and st.session_state.selected_computation_metrics:
                st.subheader("📈 Overall Computation-Based Metrics Summary (Average Across All Test Cases & Rounds)")
                overall_comp_summary_data = {llm_label: {metric: [] for metric in st.session_state.selected_computation_metrics} for llm_label in active_llm_labels_for_summary}
                
                # Add ROUGE sub-metrics to the overall summary keys if ROUGE is selected
                all_overall_metrics_to_display = st.session_state.selected_computation_metrics.copy()
                if "ROUGE" in all_overall_metrics_to_display:
                    all_overall_metrics_to_display.remove("ROUGE")
                    all_overall_metrics_to_display.extend(["rouge-1_f", "rouge-2_f", "rouge-l_f"])
                    for llm_label in active_llm_labels_for_summary:
                        overall_comp_summary_data[llm_label].update({m: [] for m in ["rouge-1_f", "rouge-2_f", "rouge-l_f"]})

                for tc_obj in st.session_state.test_cases:
                    tc_id = tc_obj["id"]
                    if tc_id in st.session_state.computation_results:
                        for r_idx in range(NUM_ROUNDS):
                            round_num_str = f"Round{r_idx + 1}"
                            if round_num_str in st.session_state.computation_results[tc_id]:
                                for llm_label, metrics_data in st.session_state.computation_results[tc_id][round_num_str].items():
                                    if llm_label in active_llm_labels_for_summary:
                                        for metric_name, score in metrics_data.items():
                                            if isinstance(score, (int, float)) and metric_name in overall_comp_summary_data[llm_label]:
                                                overall_comp_summary_data[llm_label][metric_name].append(score)

                summary_rows = []
                for llm_label, metric_lists in overall_comp_summary_data.items():
                    row = {"LLM": llm_label}
                    for metric, scores_list in metric_lists.items():
                        if scores_list:
                            row[metric] = round(statistics.mean(scores_list), 4)
                        else:
                            row[metric] = "N/A"
                    summary_rows.append(row)

                if summary_rows:
                    df_overall_comp = pd.DataFrame(summary_rows)
                    st.dataframe(df_overall_comp.set_index("LLM"))
                else:
                    st.info("No overall computation-based metric summary to display.")

    st.caption("End of LLM Evaluation Tool")
