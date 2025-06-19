import streamlit as st
import random
import time
import pandas as pd
import wikipedia
import re
import statistics 

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

if 'test_cases' not in st.session_state:
    st.session_state.test_cases = []

if 'test_case_counter' not in st.session_state:
    st.session_state.test_case_counter = 0

if 'initial_responses' not in st.session_state:
    st.session_state.initial_responses = {}

if 'evaluations' not in st.session_state:
    st.session_state.evaluations = {}


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
                
                st.session_state.test_cases.append({
                    "id": generated_tc_id,
                    "prompt": tc_prompt,
                    "wiki_query": auto_derived_wiki_query,
                    "ideal_answer_criteria": fetched_criteria,
                    "scoring_rubric": FIXED_SCORING_RUBRIC,
                    "language": st.session_state.selected_language
                })
                st.success(f"Test Case '{generated_tc_id}' ({st.session_state.selected_language}) added!")
                st.info(f"Auto-derived Wikipedia search query: '{auto_derived_wiki_query}'.")
                if "Error" in fetched_criteria or "not found" in fetched_criteria or "No search query" in fetched_criteria or "was empty" in fetched_criteria:
                    st.warning(f"Wikipedia fetch result: {fetched_criteria}")
            else:
                st.warning("The 'Prompt for LLM' field is required.")

    if st.session_state.test_cases:
        st.markdown("##### Current Test Cases:")
        for i, tc in enumerate(st.session_state.test_cases):
            lang_display = f" ({tc.get('language', 'N/A')})"
            display_name_snippet = tc['prompt'][:40] + "..." if len(tc['prompt']) > 40 else tc['prompt']
            with st.expander(f"{tc['id']}: '{display_name_snippet}'{lang_display}"):
                st.markdown(f"**Full Prompt:**\n```\n{tc['prompt']}\n```")
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
        simulated_score = str(random.randint(1, 5)) 
        st.write(f"✔️ R{round_num}, Simulated score from {model_label}: {simulated_score}")
        return simulated_score
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
    target_llm_label: str,
    response_to_evaluate: str,
    round_num: int 
) -> tuple[str, str, str, int, int]: 
    
    evaluation_full_prompt = f"""You are an expert evaluator. This is for Round {round_num}.
The original prompt was in {test_case_obj.get('language', 'the specified language')}.
The response you are evaluating should also be in {test_case_obj.get('language', 'the specified language')}.

Original Test Case Prompt:
{test_case_obj['prompt']}

Reference Information / Key Criteria (from Wikipedia, auto-queried for: '{test_case_obj.get('wiki_query', 'N/A')}'):
--- BEGIN REFERENCE INFO ---
{test_case_obj['ideal_answer_criteria']} 
--- END REFERENCE INFO ---

Scoring Rubric:
{test_case_obj['scoring_rubric']}

Response to Evaluate (from LLM: {target_llm_label}, for Round {round_num}):
--- BEGIN RESPONSE ---
{response_to_evaluate}
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

    return evaluator_llm_label, test_case_obj['prompt'], target_llm_label, parsed_score, round_num

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
            st.session_state.evaluations = {} 

            with st.spinner(f"Getting responses ({NUM_ROUNDS} rounds each) from {len(active_llms_for_run)} LLM(s) for {len(st.session_state.test_cases)} test case(s)..."):
                prompt_to_id_map = {tc["prompt"]: tc["id"] for tc in st.session_state.test_cases}

                for r_idx in range(NUM_ROUNDS):
                    round_num_str = f"Round{r_idx + 1}"
                    st.write(f"--- Starting {round_num_str} for Initial Responses ---")
                    for tc in st.session_state.test_cases:
                        current_tc_lang = tc.get('language', st.session_state.selected_language)
                        st.write(f"Processing Test Case: {tc['id']} ({round_num_str}, Language: {current_tc_lang})")
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
            wiki_query_display = tc_obj.get('wiki_query', 'N/A')
            display_name_snippet = tc_obj['prompt'][:40] + "..." if len(tc_obj['prompt']) > 40 else tc_obj['prompt']
            
            if tc_id in st.session_state.initial_responses:
                with st.expander(f"Responses for: {tc_id} ('{display_name_snippet}') ({tc_lang}) - {NUM_ROUNDS} Rounds", expanded=False):
                    st.markdown(f"**Full Prompt for {tc_id} (Language: {tc_lang}):**")
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
        st.caption(f"Click to have selected LLMs score each other's responses for each of the {NUM_ROUNDS} rounds.")
        if st.button(f"Start Peer Evaluation ({NUM_ROUNDS} Rounds)", key="run_phase2_button"):
            active_llms_for_run = get_active_llms()
            if not active_llms_for_run or len(active_llms_for_run) < 1: # Changed to < 1 as self-evaluation is now allowed
                st.warning("Please select at least one LLM in the sidebar for evaluation.")
            else:
                st.session_state.evaluations = {tc["id"]: {f"Round{r+1}": {} for r in range(NUM_ROUNDS)} for tc in st.session_state.test_cases}

                with st.spinner(f"Selected LLMs are evaluating responses ({NUM_ROUNDS} rounds)..."):
                    prompt_to_id_map_eval = {tc["prompt"]: tc["id"] for tc in st.session_state.test_cases}

                    for r_idx in range(NUM_ROUNDS):
                        round_num_str = f"Round{r_idx + 1}"
                        st.write(f"--- Starting Peer Evaluation for {round_num_str} ---")
                        for tc_obj in st.session_state.test_cases:
                            tc_id = tc_obj["id"]
                            st.write(f"Peer evaluating for TC: {tc_id}, {round_num_str} (Ref: '{tc_obj.get('wiki_query','N/A')}')")

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

                            for llm_being_evaluated_label, response_text in responses_for_tc_active_this_round.items():
                                st.session_state.evaluations[tc_id][round_num_str].setdefault(llm_being_evaluated_label, {})
                                if response_text is None or "Error:" in str(response_text) or "SIMULATED FALLBACK" in response_text :
                                    st.info(f"Skipping eval of {llm_being_evaluated_label} for {tc_id}, {round_num_str} (bad response).")
                                    continue

                                for evaluator_config in active_llms_for_run:
                                    # REMOVED: if evaluator_config["label"] == llm_being_evaluated_label: continue
                                    try:
                                        evaluator_llm_label_res, tc_prompt_key_from_eval, evaluated_llm_label_from_eval, score, _ = evaluate_response_st(
                                            evaluator_config["label"], evaluator_config["model_id"],
                                            tc_obj,
                                            llm_being_evaluated_label, response_text,
                                            round_num=r_idx + 1
                                        )
                                        tc_id_for_eval = prompt_to_id_map_eval.get(tc_prompt_key_from_eval)
                                        if tc_id_for_eval:
                                            st.session_state.evaluations[tc_id_for_eval][round_num_str].setdefault(evaluated_llm_label_from_eval, {})[evaluator_llm_label_res] = score
                                        else:
                                            st.error(f"Eval mapping error for prompt: {tc_prompt_key_from_eval[:30]}...")
                                    except Exception as e:
                                        st.error(f"Error during eval by {evaluator_config['label']} for {llm_being_evaluated_label} on TC {tc_id}, {round_num_str}: {e}")
                    st.success(f"Peer evaluation for {NUM_ROUNDS} rounds finished!")
                    st.rerun()

    if st.session_state.evaluations:
        st.subheader("📊 Review Evaluation Scores (Per-Round Matrices & Stats)")
        if st.checkbox("Show raw multi-round evaluation data (for debugging)", key="debug_multi_round_eval_data"):
            st.json(st.session_state.evaluations)

        active_llms_for_matrix_stats = get_active_llms()
        active_llm_labels_for_matrix_stats = [llm['label'] for llm in active_llms_for_matrix_stats]

        if active_llm_labels_for_matrix_stats:
            for tc_obj in st.session_state.test_cases:
                tc_id = tc_obj["id"]
                tc_lang = tc_obj.get('language', 'N/A')
                wiki_query_display = tc_obj.get('wiki_query', 'N/A')
                display_name_snippet = tc_obj['prompt'][:40] + "..." if len(tc_obj['prompt']) > 40 else tc_obj['prompt']

                if tc_id in st.session_state.evaluations and st.session_state.evaluations[tc_id]:
                    with st.expander(f"Scores & Stats for: {tc_id} - '{display_name_snippet}' ({tc_lang}) (Ref Query: '{wiki_query_display}')", expanded=False):
                        st.markdown(f"**Prompt for {tc_id} (Language: {tc_lang}):**")
                        st.code(tc_obj['prompt'], language=None)
                        st.markdown(f"**Reference Criteria (auto-derived from Wikipedia for query: `{wiki_query_display}`):**")
                        st.info(tc_obj.get('ideal_answer_criteria', 'N/A'))
                        st.markdown(f"**Fixed Scoring Rubric:**\n```\n{tc_obj['scoring_rubric']}\n```")
                        st.markdown("---")

                        # Display Per-Round Matrices
                        for r_idx in range(NUM_ROUNDS):
                            round_num_str = f"Round{r_idx + 1}"
                            st.markdown(f"##### Score Matrix for {round_num_str}")
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
                                            # Self-evaluation score will now appear on the diagonal
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
                            st.markdown("---") # Separator between round matrices

                        # Display Score Statistics (Min, Max, Mean, StdDev of average scores per round)
                        st.markdown("##### Score Statistics Across Rounds (Average Peer Score per Round)")
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
                                        # For stats, only consider scores from *other* active LLMs if you want peer-review stats
                                        # Or include self-score if that's desired for this particular statistic
                                        if evaluator in active_llm_labels_for_matrix_stats and evaluator != llm_evaluated_label and isinstance(s, int) and 1 <= s <= 5
                                    ]
                                    if valid_scores_this_round: # Need at least one peer score
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
                            st.info("No evaluation statistics to display for active LLMs on this test case.")
        else:
            st.info("No LLMs selected to display score statistics or matrices.")

    st.divider()

    if st.session_state.evaluations:
        st.header("Phase 3: Final Summary (Based on Mean of Round Averages)")
        st.caption("Overall performance scores based on the mean of average scores achieved by selected LLMs across all rounds and test cases.")

        active_llms_for_summary = get_active_llms()
        active_llm_labels_for_summary = [llm['label'] for llm in active_llms_for_summary]

        if not active_llm_labels_for_summary:
            st.info("No LLMs were selected for evaluation, so no summary can be generated.")
        else:
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
                                    # For final summary, consider if self-scores should be included in this average
                                    # Here, we only average scores given by *other* active LLMs
                                    if evaluator in active_llm_labels_for_summary and evaluator != llm_label_evaluated and isinstance(s, int) and 1 <= s <= 5
                                ]
                                if valid_scores_received:
                                    round_averages_for_this_tc_llm.append(sum(valid_scores_received) / len(valid_scores_received))
                    
                    if round_averages_for_this_tc_llm: # If there were any valid round averages for this TC
                        mean_score_for_tc = round(statistics.mean(round_averages_for_this_tc_llm), 2)
                        final_scores_summary[llm_label_evaluated]["mean_scores_by_tc"][tc_id_str] = mean_score_for_tc
                        sum_of_tc_mean_scores += mean_score_for_tc
                        num_valid_tc_for_llm += 1
                    else:
                        final_scores_summary[llm_label_evaluated]["mean_scores_by_tc"][tc_id_str] = "N/A"

                if num_valid_tc_for_llm > 0:
                    overall_mean = sum_of_tc_mean_scores / num_valid_tc_for_llm
                    final_scores_summary[llm_label_evaluated]["overall_mean_of_means"] = round(overall_mean, 2)

            st.subheader("🏆 LLM Performance Ranking (Selected LLMs - Mean of Round Averages)")
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

st.caption("End of LLM Evaluation Tool")
