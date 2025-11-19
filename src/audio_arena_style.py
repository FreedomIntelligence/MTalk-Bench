import os
import json
import random
import re
import base64
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI
from pydub import AudioSegment
import time
import threading
import itertools
from datasets import load_dataset
import tempfile
import soundfile as sf

# Command line arguments
parser = argparse.ArgumentParser(description="Run hybrid arena evaluation with HF dataset and new JSON data")
parser.add_argument("--eval_type", type=str, required=True,
                    choices=["semantic", "paralinguistic", "ambient"],
                    help="Evaluation type to run")
parser.add_argument("--judge_model", type=str, required=True,
                    choices=[
                        "gpt-4o-audio-preview",
                        "gemini-2.5-pro"
                    ],
                    help="Judge model to use")
parser.add_argument("--new_data_file", type=str, default=None,
                    help="Path to new JSON data file with additional model responses")
args = parser.parse_args()

# Configuration
EVAL_TYPE = args.eval_type
JUDGE_MODEL = args.judge_model
NEW_DATA_FILE = args.new_data_file

# Evaluation type mapping
TYPE_MAPPING = {
    "semantic": "sem", 
    "paralinguistic": "para", 
    "ambient": "amb"
}
TYPE_ABBREV = TYPE_MAPPING[EVAL_TYPE]

NAME_IN_JSON = {
    "semantic": "Semantic information",
    "paralinguistic": "Paralinguistic information",
    "ambient": "Ambient sound"
}[EVAL_TYPE]

# HuggingFace dataset configuration
HF_DATASET_ID = "./MTalk-Bench"

# Output directory
OUTPUT_DIR = f"arena_{JUDGE_MODEL}_hybrid_eval_output_{EVAL_TYPE}"
TEMP_AUDIO_DIR = os.path.join(OUTPUT_DIR, "temp_audio")

# Hint audio directory (local GitHub repo path)
TTS_HINT_DIR = "../asset"

# Multithreading configuration
MAX_WORKERS = 1
THREAD_LOCK = threading.Lock()

# Elo rating parameters
K = 4
SCALE = 400
INIT_RATING = 1000

# API configuration
API_KEY = "API_KEY"
BASE_URL = 'BASE_URL'

# Initialization
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Check if hint sounds exist
required_hints = [
    "dialogue_with_model_a.wav", "dialogue_with_model_b.wav",
    "turn_1.wav", "turn_2.wav", "turn_3.wav",
    "models_response.wav", "silence_1s.wav"
]
for hint in required_hints:
    hint_path = os.path.join(TTS_HINT_DIR, hint)
    if not os.path.exists(hint_path):
        print(f"Warning: Hint audio '{hint}' not found in '{TTS_HINT_DIR}'. Some features may be affected.")

# Data loading and processing

def load_hf_dataset():
    """Load dataset from HuggingFace"""
    print(f"Loading dataset from HuggingFace: {HF_DATASET_ID}")
    try:
        dataset = load_dataset(HF_DATASET_ID)
        print(f"HF dataset loaded successfully, total records: {len(dataset['test'])}")
        return dataset['test']
    except Exception as e:
        print(f"Warning: Cannot load HF dataset {HF_DATASET_ID}: {e}")
        return None

def load_new_json_data():
    """Load new JSON data file"""
    if not NEW_DATA_FILE or not os.path.exists(NEW_DATA_FILE):
        print("No new data file provided or file doesn't exist, using HF data only")
        return None
    
    try:
        with open(NEW_DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"New JSON data loaded successfully: {NEW_DATA_FILE}")
        return data
    except Exception as e:
        print(f"Warning: Cannot load new JSON data {NEW_DATA_FILE}: {e}")
        return None

def save_audio_to_temp(audio_record, prefix="temp_audio"):
    """Save HuggingFace audio record as temporary file"""
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, 
        suffix='.wav', 
        prefix=f"{prefix}_",
        dir=TEMP_AUDIO_DIR
    )
    
    # Write audio data
    sf.write(
        temp_file.name, 
        audio_record['audio']['array'], 
        audio_record['audio']['sampling_rate']
    )
    
    return temp_file.name

def discover_dialogues_hybrid():
    """Discover dialogue structure from hybrid HF dataset and new JSON data"""
    dialogues = {}
    all_models = set()
    
    # 1. First load base data from HF dataset (contains all questions and prompts, may lack model answers)
    hf_dataset = load_hf_dataset()
    
    if not hf_dataset:
        print("Error: Cannot load HF dataset")
        return {}, []
    
    # Filter for specified type data
    filtered_data = hf_dataset.filter(lambda x: x['type'] == TYPE_ABBREV)
    print(f"Filtered {EVAL_TYPE} HF data records: {len(filtered_data)}")
    
    # Group data by dialogue ID
    dialogue_groups = defaultdict(list)
    
    for record in filtered_data:
        dialogue_id = record['number']  # e.g. "1-1", "2-3"
        dialogue_groups[dialogue_id].append(record)
    
    print(f"Discovered {len(dialogue_groups)} dialogues from HF data")
    
    # Build basic dialogue structure (at least contains questions, model answers may be empty)
    for dialogue_id, records in dialogue_groups.items():
        # Separate questions and answers
        questions = {}
        model_responses = defaultdict(dict)
        
        for record in records:
            source = record['source']
            turn = record['turn']
            
            if source == 'question':
                questions[turn] = record
            else:
                # This is model answer
                model_responses[turn][source] = record
                all_models.add(source)
        
        # Validate data integrity: at least need questions
        if len(questions) > 0:
            # Get all turns (including those without model answers)
            all_turns = set(questions.keys())
            # Add turns with model answers
            all_turns.update(model_responses.keys())
            
            available_models = set()
            for turn in all_turns:
                if turn in model_responses:
                    available_models.update(model_responses[turn].keys())
            
            # Build dialogue data structure (keep even without model answers)
            dialogue_data = {
                "turns": {},
                "models": list(available_models),  # May be empty list
                "source": "hf"  # Mark data source
            }
            
            # Build data for each turn
            for turn in sorted(all_turns, key=lambda x: int(x.replace('turn', ''))):
                turn_num = int(turn.replace('turn', ''))
                
                # Ensure this turn has question
                if turn in questions:
                    dialogue_data["turns"][turn_num] = {
                        "user": questions[turn]  # Contains question audio and arena_prompt
                    }
                    
                    # If this turn has model answers, add them
                    if turn in model_responses:
                        dialogue_data["turns"][turn_num].update(model_responses[turn])
                else:
                    # If no question but has model answers, log warning but don't create turn
                    print(f"Warning: Dialogue {dialogue_id} turn {turn} has model answers but missing question, skipping this turn")
            
            # Keep dialogue as long as it has valid turns
            if len(dialogue_data["turns"]) > 0:
                dialogues[dialogue_id] = dialogue_data
    
    print(f"HF data construction complete, {len(dialogues)} dialogues, {len(all_models)} models")
    
    # Statistics for HF data dialogues
    hf_dialogues_with_models = 0
    hf_dialogues_without_models = 0
    for dialogue_data in dialogues.values():
        if len(dialogue_data["models"]) > 0:
            hf_dialogues_with_models += 1
        else:
            hf_dialogues_without_models += 1
    
    print(f"HF data statistics:")
    print(f"  - Dialogues with model answers: {hf_dialogues_with_models}")
    print(f"  - Question-only dialogues: {hf_dialogues_without_models}")
    
    # 2. Load additional model answers from new JSON data (supplement only, no override)
    new_json_data = load_new_json_data()
    
    if new_json_data:
        eval_data = new_json_data.get(NAME_IN_JSON, [])
        print(f"Discovered {len(eval_data)} dialogue groups from new JSON data")
        
        added_models = set()
        updated_dialogues = 0
        dialogues_activated = 0  # Number of dialogues changed from no-model to has-model
        
        for dialogue_group in eval_data:
            if not isinstance(dialogue_group, list) or not dialogue_group:
                continue
            
            # Extract dialogue ID from first turn
            first_turn = dialogue_group[0]
            question_path = first_turn.get("question", "")
            
            # Extract dialogue ID from path
            match = re.search(r'record_(\d+-\d+)', question_path)
            if not match:
                continue
            dialogue_id = match.group(1)
            
            # Check if exists in HF data
            if dialogue_id not in dialogues:
                print(f"Warning: Dialogue {dialogue_id} not in HF data, skipping JSON supplement")
                continue
            
            existing_dialogue = dialogues[dialogue_id]
            dialogue_updated = False
            was_empty = len(existing_dialogue["models"]) == 0
            
            # Add new model answers to existing dialogue
            for turn_idx, turn_data in enumerate(dialogue_group, 1):
                dialogue_dict = turn_data.get("dialogue", {})
                
                # Check if this turn exists in HF data
                if turn_idx not in existing_dialogue["turns"]:
                    print(f"Warning: Dialogue {dialogue_id} turn {turn_idx} not in HF data, skipping")
                    continue
                
                # Add new model answers (only add non-existing models)
                for model_name, model_response_path in dialogue_dict.items():
                    if model_name not in existing_dialogue["turns"][turn_idx]:
                        # Create new model answer record structure
                        new_record = {
                            "type": TYPE_ABBREV,
                            "number": dialogue_id,
                            "turn": f"turn{turn_idx}",
                            "source": model_name,
                            "audio_path": model_response_path,
                            "transcription": "",  # New data may not have transcription
                            "arena_prompt": "",   # Get from question record
                            "criteria_prompt_general": "",
                            "criteria_prompt_specific": "",
                            "source_type": "json"  # Mark as JSON source
                        }
                        
                        # Add to existing dialogue structure
                        existing_dialogue["turns"][turn_idx][model_name] = new_record
                        
                        # Update model list
                        if model_name not in existing_dialogue["models"]:
                            existing_dialogue["models"].append(model_name)
                        
                        all_models.add(model_name)
                        added_models.add(model_name)
                        dialogue_updated = True
                    else:
                        print(f"  Skip: Dialogue {dialogue_id} turn {turn_idx} already has model {model_name}")
            
            if dialogue_updated:
                if was_empty and len(existing_dialogue["models"]) > 0:
                    dialogues_activated += 1
                existing_dialogue["source"] = "hybrid" if was_empty else existing_dialogue["source"]
                updated_dialogues += 1
        
        print(f"JSON data supplement complete:")
        print(f"  - Updated dialogues: {updated_dialogues}")
        print(f"  - Activated dialogues (from no-model to has-model): {dialogues_activated}")
        print(f"  - New models added: {sorted(list(added_models))}")
    
    # 3. Categorize statistics and filtering
    dialogues_with_models = {}  # Dialogues with at least 2 models
    dialogues_single_model = {}  # Dialogues with only 1 model
    dialogues_no_models = {}  # Dialogues without models
    
    for dialogue_id, dialogue_data in dialogues.items():
        model_count = len(dialogue_data["models"])
        formatted_dialogue_id = f"{EVAL_TYPE}_question{dialogue_id}"
        
        if model_count >= 2:
            dialogues_with_models[formatted_dialogue_id] = dialogue_data
        elif model_count == 1:
            dialogues_single_model[formatted_dialogue_id] = dialogue_data
        else:
            dialogues_no_models[formatted_dialogue_id] = dialogue_data
    
    print(f"\n=== Final Statistics ===")
    print(f"Total dialogues: {len(dialogues)}")
    print(f"  - Evaluable dialogues (>=2 models): {len(dialogues_with_models)}")
    print(f"  - Single-model dialogues (=1 model): {len(dialogues_single_model)}")
    print(f"  - No-model dialogues (=0 models): {len(dialogues_no_models)}")
    print(f"All models ({len(all_models)}): {sorted(list(all_models))}")
    
    # Statistics by data source
    source_stats = defaultdict(int)
    for dialogue_data in dialogues_with_models.values():
        source_stats[dialogue_data["source"]] += 1
    print(f"Evaluable dialogue sources: {dict(source_stats)}")
    
    # If no evaluable dialogues, show detailed info for debugging
    if len(dialogues_with_models) == 0:
        print(f"\nWarning: No evaluable dialogues!")
        if len(dialogues_single_model) > 0:
            print(f"Suggestion: {len(dialogues_single_model)} single-model dialogues exist, add more model answers")
        if len(dialogues_no_models) > 0:
            print(f"Suggestion: {len(dialogues_no_models)} no-model dialogues exist, add model answers")
        
        # Show detailed info for first few dialogues
        sample_dialogues = list(dialogues.items())[:3]
        for dialogue_id, dialogue_data in sample_dialogues:
            print(f"\nSample dialogue {dialogue_id}:")
            print(f"  - Turn count: {len(dialogue_data['turns'])}")
            print(f"  - Model count: {len(dialogue_data['models'])}")
            print(f"  - Models: {dialogue_data['models']}")
    
    # Return evaluable dialogues
    return dialogues_with_models, sorted(list(all_models))

# Audio processing

def concatenate_audio_hybrid(file_list, output_path):
    """Hybrid audio concatenation supporting HF records and file paths"""
    combined = AudioSegment.empty()
    
    # Load silence separator
    silence_path = os.path.join(TTS_HINT_DIR, "silence_1s.wav")
    if os.path.exists(silence_path):
        silence = AudioSegment.from_wav(silence_path)
    else:
        silence = AudioSegment.silent(duration=1000)  # 1 second silence
    
    target_rate = None
    target_channels = None
    target_width = None
    
    # Process file list, handle different audio source types
    for item in file_list:
        try:
            if isinstance(item, str):
                # Handle file path
                filepath = item
                if filepath.startswith('.'):
                    # Ensure correct path handling
                    filepath = f'.{filepath}'
                
                if os.path.exists(filepath):
                    audio = AudioSegment.from_wav(filepath)
                else:
                    print(f"Warning: Audio file not found: {filepath}")
                    continue
            elif isinstance(item, dict) and 'audio' in item:
                # HF audio record
                temp_path = save_audio_to_temp(item, "concat")
                audio = AudioSegment.from_wav(temp_path)
                # Clean up temp file
                os.unlink(temp_path)
            else:
                print(f"Warning: Unsupported audio type: {type(item)}")
                continue
            
            # Set target format
            if target_rate is None:
                target_rate = audio.frame_rate
                target_channels = audio.channels
                target_width = audio.sample_width
                print(f"Audio concat target format: {target_rate}Hz, {target_channels}ch, {target_width*8}bit")
            
            # Standardize format
            if (audio.frame_rate != target_rate or 
                audio.channels != target_channels or 
                audio.sample_width != target_width):
                audio = audio.set_frame_rate(target_rate)
                audio = audio.set_channels(target_channels)
                audio = audio.set_sample_width(target_width)
            
            combined += audio + silence
            
        except Exception as e:
            print(f"Warning: Error processing audio {item}: {e}")
            continue
    
    if combined.duration_seconds > 0:
        combined.export(output_path, format="wav")
        print(f"Audio concatenation complete: {output_path}, total duration: {combined.duration_seconds:.2f}s")
    else:
        print(f"Error: Failed to concatenate any audio")

def get_audio_source(record):
    """Get audio source based on record type"""
    if 'audio' in record:
        # HF record
        return record
    elif 'audio_path' in record:
        # JSON file path record
        return record['audio_path']
    else:
        print(f"Warning: Cannot determine audio source: {record}")
        return None

def build_comparison_audio_hybrid(dialogue_data, model_a, model_b, question_id, thread_id=0):
    """Build comparison audio for hybrid data"""
    audio_segments = []
    
    # Model A's Dialogue
    model_a_hint = os.path.join(TTS_HINT_DIR, "dialogue_with_model_a.wav")
    if os.path.exists(model_a_hint):
        audio_segments.append(model_a_hint)
    
    for turn_num in sorted(dialogue_data["turns"].keys()):
        turn_data = dialogue_data["turns"][turn_num]
        
        # Add turn hint
        turn_hint = os.path.join(TTS_HINT_DIR, f"turn_{turn_num}.wav")
        if os.path.exists(turn_hint):
            audio_segments.append(turn_hint)
        
        # Add user question
        user_audio = get_audio_source(turn_data['user'])
        if user_audio:
            audio_segments.append(user_audio)
        
        # Add model response hint
        response_hint = os.path.join(TTS_HINT_DIR, "models_response.wav")
        if os.path.exists(response_hint):
            audio_segments.append(response_hint)
        
        # Add Model A response
        if model_a in turn_data:
            model_a_audio = get_audio_source(turn_data[model_a])
            if model_a_audio:
                audio_segments.append(model_a_audio)
    
    # Model B's Dialogue
    model_b_hint = os.path.join(TTS_HINT_DIR, "dialogue_with_model_b.wav")
    if os.path.exists(model_b_hint):
        audio_segments.append(model_b_hint)
    
    for turn_num in sorted(dialogue_data["turns"].keys()):
        turn_data = dialogue_data["turns"][turn_num]
        
        # Add turn hint
        turn_hint = os.path.join(TTS_HINT_DIR, f"turn_{turn_num}.wav")
        if os.path.exists(turn_hint):
            audio_segments.append(turn_hint)
        
        # Add user question
        user_audio = get_audio_source(turn_data['user'])
        if user_audio:
            audio_segments.append(user_audio)
        
        # Add model response hint
        response_hint = os.path.join(TTS_HINT_DIR, "models_response.wav")
        if os.path.exists(response_hint):
            audio_segments.append(response_hint)
        
        # Add Model B response
        if model_b in turn_data:
            model_b_audio = get_audio_source(turn_data[model_b])
            if model_b_audio:
                audio_segments.append(model_b_audio)
    
    output_filename = f"{question_id}_{model_a}_vs_{model_b}_thread{thread_id}_{int(time.time())}.wav"
    output_path = os.path.join(TEMP_AUDIO_DIR, output_filename)
    
    concatenate_audio_hybrid(audio_segments, output_path)
    return output_path

def get_arena_prompt_hybrid(dialogue_data):
    """Get arena prompt from hybrid data"""
    # Get arena_prompt from first question record
    for turn_num in sorted(dialogue_data["turns"].keys()):
        turn_data = dialogue_data["turns"][turn_num]
        if 'user' in turn_data:
            user_record = turn_data['user']
            # Prioritize arena_prompt from HF data
            arena_prompt = user_record.get('arena_prompt', '')
            if arena_prompt and arena_prompt.strip():
                print(f"Got arena_prompt from turn {turn_num} (length: {len(arena_prompt)})")
                return arena_prompt
    
    print("Warning: No valid arena_prompt found")
    return ""

# Judge model calling

class JudgeModel:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def evaluate(self, audio_path, prompt_text):
        with open(audio_path, "rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')

        for _ in range(3):  # Retry mechanism
            try:
                completion = self.client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {
                                    "type": "input_audio",
                                    "input_audio": {"data": encoded_string, "format": "wav"}
                                }
                            ]
                        }
                    ],
                    temperature=0.1,
                )
                response_content = completion.choices[0].message.content
                print(response_content)
                
                # Parse JSON
                try:
                    result = json.loads(response_content)
                except json.JSONDecodeError:
                    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_content)
                    if not json_match:
                        raise ValueError("No valid JSON format or JSON code block found in response.")
                    
                    result = json.loads(json_match.group(1))
                
                if "winner" not in result or "reason" not in result:
                    raise ValueError("JSON response missing 'winner' or 'reason' field.")
                
                return result

            except Exception as e:
                print(f"Error calling judge model: {e}. Retrying...")
                time.sleep(5)
        
        print(f"Error: Unable to get valid response from judge model after multiple retries.")
        return None

# Elo rating logic (keep original logic)

def load_elo_ratings_from_log(models, log_filepath):
    """Restore Elo ratings and evaluated battle records from log file"""
    elo_ratings = {model: INIT_RATING for model in models}
    evaluated_battles_by_round = {}
    
    if not os.path.exists(log_filepath):
        print("   No historical log file found, starting from initial ratings.")
        return elo_ratings, evaluated_battles_by_round
    
    print("   Restoring Elo ratings and battle records from historical log...")
    battle_count = 0
    
    try:
        with open(log_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    
                    required_fields = ["question_id", "model_a", "model_b", "elo_after", "round_num"]
                    if not all(field in entry for field in required_fields):
                        continue
                    
                    model_a = entry["model_a"]
                    model_b = entry["model_b"]
                    
                    if model_a in elo_ratings and model_b in elo_ratings:
                        elo_ratings[model_a] = entry["elo_after"]["a"]
                        elo_ratings[model_b] = entry["elo_after"]["b"]
                        
                        round_num = entry["round_num"]
                        question_id = entry["question_id"]
                        
                        if round_num not in evaluated_battles_by_round:
                            evaluated_battles_by_round[round_num] = set()
                        evaluated_battles_by_round[round_num].add(question_id)
                        
                        battle_count += 1
                
                except (json.JSONDecodeError, KeyError):
                    continue
    
    except Exception as e:
        print(f"   Warning: Error reading log file: {e}")
        return {model: INIT_RATING for model in models}, {}
    
    print(f"   Successfully restored Elo ratings from {battle_count} battles")
    return elo_ratings, evaluated_battles_by_round

def get_remaining_dialogues_for_round(round_num, dialogues, evaluated_battles_by_round):
    """Get remaining dialogues to evaluate for specified round"""
    all_dialogues = list(dialogues.items())
    completed_dialogue_ids = evaluated_battles_by_round.get(round_num, set())
    
    remaining_dialogues = [
        (question_id, data) for question_id, data in all_dialogues
        if question_id not in completed_dialogue_ids and len(data["models"]) >= 2
    ]
    
    random.shuffle(remaining_dialogues)
    
    print(f"   Round {round_num+1}: total={len(all_dialogues)}, completed={len(completed_dialogue_ids)}, remaining={len(remaining_dialogues)}")
    
    return remaining_dialogues

def select_model_pair_with_randomness(available_models, elo_ratings, thread_id):
    """Select model pair"""
    model_pairs = [(m1, m2) for i, m1 in enumerate(available_models) 
                  for j, m2 in enumerate(available_models) if i < j]
    
    if not model_pairs:
        return None, None
    
    pairs_with_scores = []
    for pair in model_pairs:
        score_diff = abs(elo_ratings[pair[0]] - elo_ratings[pair[1]])
        pairs_with_scores.append((pair, score_diff))
    
    pairs_with_scores.sort(key=lambda x: x[1])
    
    top_n = min(3, len(pairs_with_scores))
    selected_pair, _ = random.choice(pairs_with_scores[:top_n])
    
    return selected_pair

def compute_elo_update(ra, rb, sa):
    ea = 1 / (1 + 10 ** ((rb - ra) / SCALE))
    eb = 1 / (1 + 10 ** ((ra - rb) / SCALE))
    new_ra = ra + K * (sa - ea)
    new_rb = rb + K * ((1 - sa) - eb)
    return new_ra, new_rb

def evaluate_single_battle(args):
    """Evaluate single battle"""
    (thread_id, round_num, question_id, data, elo_ratings, log_filepath, evaluated_battles_by_round) = args
    
    available_models = data["models"]
    if len(available_models) < 2:
        return None

    with THREAD_LOCK:
        if question_id in evaluated_battles_by_round.get(round_num, set()):
            print(f"Thread-{thread_id}: Skip completed: Round {round_num+1} | {question_id}")
            return None
        
        current_elo, _ = load_elo_ratings_from_log(list(elo_ratings.keys()), log_filepath)
    
    selected_pair = select_model_pair_with_randomness(available_models, current_elo, thread_id)
    if not selected_pair:
        return None
    
    model_a, model_b = selected_pair
    if random.random() < 0.5:
        model_a, model_b = model_b, model_a
    
    print(f"\n--- Thread-{thread_id} | Round {round_num+1} | Dialogue: {question_id} ---")
    print(f"Evaluating: {model_a} (Elo: {current_elo[model_a]:.0f}) vs {model_b} (Elo: {current_elo[model_b]:.0f})")

    try:
        # Build comparison audio
        print(f"   Thread-{thread_id}: Concatenating audio...")
        audio_path = build_comparison_audio_hybrid(data, model_a, model_b, question_id, thread_id)
        
        # Get prompt
        prompt_text = get_arena_prompt_hybrid(data)
        if not prompt_text:
            print(f"   Thread-{thread_id}: Warning: No arena prompt found, skipping this dialogue.")
            return None

        # Call judge model
        print(f"   Thread-{thread_id}: Calling judge model for evaluation...")
        judge = JudgeModel(api_key=API_KEY, base_url=BASE_URL)
        eval_result = judge.evaluate(audio_path, prompt_text)

        if not eval_result:
            print(f"   Thread-{thread_id}: Evaluation failed, skipping this battle.")
            return None

        winner_str = eval_result.get("winner")
        reason = eval_result.get("reason")
        
        if winner_str == "Model A":
            winner = model_a
            sa = 1.0
        elif winner_str == "Model B":
            winner = model_b
            sa = 0.0
        else:
            print(f"   Thread-{thread_id}: Invalid winner '{winner_str}', skipping this battle.")
            return None

        # Update Elo ratings
        with THREAD_LOCK:
            latest_elo, latest_battles = load_elo_ratings_from_log(list(elo_ratings.keys()), log_filepath)
            
            if question_id in latest_battles.get(round_num, set()):
                print(f"   Thread-{thread_id}: Dialogue already processed by another thread, skipping.")
                return None
            
            ra, rb = latest_elo[model_a], latest_elo[model_b]
            new_ra, new_rb = compute_elo_update(ra, rb, sa)

            print(f"   Thread-{thread_id}: Judge choice: {winner_str} ({winner})")
            print(f"   Thread-{thread_id}: Reason: {reason}")
            print(f"   Thread-{thread_id}: Elo update: {model_a}: {ra:.0f} -> {new_ra:.0f} | {model_b}: {rb:.0f} -> {new_rb:.0f}")

            # Log entry
            log_entry = {
                "thread_id": thread_id,
                "round_num": round_num,
                "question_id": question_id,
                "model_a": model_a,
                "model_b": model_b,
                "winner": winner,
                "reason": reason,
                "elo_before": {"a": ra, "b": rb},
                "elo_after": {"a": new_ra, "b": new_rb},
                "data_source": data["source"],
                "timestamp": time.time()
            }
            
            with open(log_filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            if round_num not in evaluated_battles_by_round:
                evaluated_battles_by_round[round_num] = set()
            evaluated_battles_by_round[round_num].add(question_id)
        
        # Clean up temp audio
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        return {
            "thread_id": thread_id,
            "round_num": round_num,
            "question_id": question_id,
            "model_a": model_a,
            "model_b": model_b,
            "winner": winner,
            "elo_change": (new_ra - ra, new_rb - rb)
        }
        
    except Exception as e:
        print(f"   Thread-{thread_id}: Error during evaluation: {e}")
        return None

def main():
    print(f"=== Hybrid Arena Evaluation: {EVAL_TYPE.upper()} ===")
    print(f"Using judge model: {JUDGE_MODEL}")
    print(f"HF dataset: {HF_DATASET_ID}")
    if NEW_DATA_FILE:
        print(f"New data file: {NEW_DATA_FILE}")
    
    print("1. Loading hybrid HF data and new JSON data...")
    dialogues, all_models = discover_dialogues_hybrid()
    if not dialogues:
        print("Error: No valid dialogue data found.")
        return
    print(f"   Found {len(dialogues)} dialogues, {len(all_models)} models: {all_models}")

    print("2. Initializing Elo ratings...")
    log_filepath = os.path.join(OUTPUT_DIR, "elo_arena_hybrid_log.jsonl")
    elo_ratings, evaluated_battles_by_round = load_elo_ratings_from_log(all_models, log_filepath)

    print("3. Starting multithreaded Elo arena evaluation...")
    
    total_battles = 0
    new_battles = 0
    
    # Collect all evaluation tasks
    tasks = []
    for round_num in range(1):
        print(f"\nPreparing Round {round_num+1} evaluation tasks")
        
        remaining_dialogues = get_remaining_dialogues_for_round(round_num, dialogues, evaluated_battles_by_round)
        
        for question_id, data in remaining_dialogues:
            available_models = data["models"]
            if len(available_models) < 2:
                continue

            total_battles += 1
            
            task_args = (
                len(tasks) % MAX_WORKERS,
                round_num,
                question_id,
                data,
                elo_ratings,
                log_filepath,
                evaluated_battles_by_round
            )
            tasks.append(task_args)
    
    print(f"\nTask statistics:")
    print(f"   - Dialogues to evaluate: {total_battles}")
    print(f"   - Concurrent threads: {MAX_WORKERS}")
    
    if not tasks:
        print("   All dialogues completed!")
        final_elo, _ = load_elo_ratings_from_log(all_models, log_filepath)
        print(f"\n   Final Elo ratings:")
        sorted_elo = sorted(final_elo.items(), key=lambda item: item[1], reverse=True)
        for rank, (model, rating) in enumerate(sorted_elo, 1):
            print(f"     {rank}. {model}: {rating:.2f}")
        return
    
    print(f"\nStarting multithreaded evaluation...")
    
    # Execute tasks using thread pool
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(evaluate_single_battle, task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                if result:
                    new_battles += 1
                    print(f"Thread-{result['thread_id']}: Completed evaluation {result['question_id']}")
            except Exception as exc:
                print(f"Task execution error: {exc}")

    print(f"\n4. Evaluation complete!")
    print(f"   Statistics:")
    print(f"     - Dialogues processed: {total_battles}")
    print(f"     - New evaluations: {new_battles}")
    
    # Final results
    final_elo, _ = load_elo_ratings_from_log(all_models, log_filepath)
    
    print(f"\n   Final Elo ratings:")
    sorted_elo = sorted(final_elo.items(), key=lambda item: item[1], reverse=True)
    for rank, (model, rating) in enumerate(sorted_elo, 1):
        print(f"     {rank}. {model}: {rating:.2f}")

    # Save final results
    final_elo_path = os.path.join(OUTPUT_DIR, "final_elo_ratings_hybrid.json")
    final_result = {
        "eval_type": EVAL_TYPE,
        "judge_model": JUDGE_MODEL,
        "ratings": dict(sorted_elo),
        "metadata": {
            "processed_battles": total_battles,
            "new_battles": new_battles,
            "max_workers": MAX_WORKERS,
            "hf_dataset_id": HF_DATASET_ID,
            "new_data_file": NEW_DATA_FILE,
            "timestamp": time.time()
        }
    }
    with open(final_elo_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    print(f"\nFinal results saved to {final_elo_path}")

if __name__ == "__main__":
    main()
