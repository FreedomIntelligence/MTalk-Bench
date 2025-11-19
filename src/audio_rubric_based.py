import os
import json
import re
import base64
import argparse
from pathlib import Path
from openai import OpenAI
from pydub import AudioSegment
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datasets import load_dataset
import tempfile
import soundfile as sf

# Command line arguments
parser = argparse.ArgumentParser(description="Run hybrid rubric evaluation with HF dataset and new JSON data")
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

# HuggingFace dataset
HF_DATASET_ID = "./MTalk-Bench"

# Directory paths
OUTPUT_DIR = f"rubric_{JUDGE_MODEL}_hybrid_eval_output_{EVAL_TYPE}"
TEMP_AUDIO_DIR = os.path.join(OUTPUT_DIR, "temp_audio")
TTS_HINT_DIR = "../asset"

# Multithreading configuration
MAX_WORKERS = 1
THREAD_LOCK = threading.Lock()

# API configuration
API_KEY = "API_KEY"
BASE_URL = 'BASE_URL'

# Initialization
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Check if hint sounds exist
required_hints = [
    "turn_1.wav", "turn_2.wav", "turn_3.wav",
    "models_response.wav", "silence_1s.wav"
]
for hint in required_hints:
    hint_path = os.path.join(TTS_HINT_DIR, hint)
    if not os.path.exists(hint_path):
        print(f"Warning: Hint audio '{hint}' not found in '{TTS_HINT_DIR}'. Some features may be affected.")

# Data loading and processing (based on audio_arena_api.py)

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
    """Discover dialogue structure from hybrid HF dataset and new JSON data (based on audio_arena_api.py)"""
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
                        "user": questions[turn]  # Contains question audio and rubric_prompt
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
                            "rubric_prompt_general": "",   # Get from question record
                            "rubric_prompt_specific": "",
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
    
    # 3. Filter dialogues with models
    final_dialogues = {}
    for dialogue_id, dialogue_data in dialogues.items():
        if len(dialogue_data["models"]) > 0:  # At least need 1 model for rubric evaluation
            final_dialogues[dialogue_id] = dialogue_data
    
    print(f"\n=== Final Statistics ===")
    print(f"Evaluable dialogues: {len(final_dialogues)}")
    print(f"All models ({len(all_models)}): {sorted(list(all_models))}")
    
    # Statistics by data source
    source_stats = defaultdict(int)
    for dialogue_data in final_dialogues.values():
        source_stats[dialogue_data["source"]] += 1
    print(f"Data source statistics: {dict(source_stats)}")
    
    return final_dialogues, sorted(list(all_models))

# Resume logic

def load_completed_evaluations(output_file):
    """Load completed evaluation records from output file"""
    completed_evaluations = set()
    
    if not os.path.exists(output_file):
        print("   No historical evaluation file found, starting from scratch.")
        return completed_evaluations
    
    print("   Restoring progress from historical evaluation file...")
    evaluation_count = 0
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    
                    # Validate required fields
                    required_fields = ["dialogue_id", "model_name", "general_evaluation", "sample_evaluation"]
                    if not all(field in entry for field in required_fields):
                        continue
                    
                    # Check if evaluation is complete (both prompts have results)
                    if (entry["general_evaluation"] is not None and 
                        entry["sample_evaluation"] is not None):
                        
                        evaluation_key = (entry["dialogue_id"], entry["model_name"])
                        completed_evaluations.add(evaluation_key)
                        evaluation_count += 1
                
                except (json.JSONDecodeError, KeyError):
                    continue
    
    except Exception as e:
        print(f"   Warning: Error reading evaluation file: {e}")
        return set()
    
    print(f"   Successfully restored {evaluation_count} completed evaluations")
    return completed_evaluations

def is_evaluation_completed(dialogue_id, model_name, completed_evaluations):
    """Check if evaluation for specified dialogue and model is completed"""
    evaluation_key = (dialogue_id, model_name)
    return evaluation_key in completed_evaluations

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

def build_model_dialogue_audio_hybrid(dialogue_data, model_name, dialogue_id, thread_id=0):
    """Build complete dialogue audio for single model (hybrid data version)"""
    audio_segments = []
    
    # Concatenate all turns for this model
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
        
        # Add model response
        if model_name in turn_data:
            model_audio = get_audio_source(turn_data[model_name])
            if model_audio:
                audio_segments.append(model_audio)
        
    # Add thread ID and timestamp to avoid filename conflicts
    output_filename = f"{dialogue_id}_{model_name}_dialogue_thread{thread_id}_{int(time.time())}.wav"
    output_path = os.path.join(TEMP_AUDIO_DIR, output_filename)
    
    concatenate_audio_hybrid(audio_segments, output_path)
    return output_path

def get_rubric_prompts_hybrid(dialogue_data, dialogue_id):
    """Get rubric prompts from hybrid data"""
    # 1. Prioritize getting from HF data
    for turn_num in sorted(dialogue_data["turns"].keys()):
        turn_data = dialogue_data["turns"][turn_num]
        if 'user' in turn_data:
            user_record = turn_data['user']
            general_prompt = user_record.get('rubric_prompt_general', '')
            specific_prompt = user_record.get('rubric_prompt_specific', '')
            
            if general_prompt and specific_prompt:
                return general_prompt, specific_prompt
    
    print(f"Warning: Dialogue {dialogue_id} no valid rubric prompts found")
    return None, None

# Judge model calling

class RubricJudge:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def evaluate_with_prompt(self, audio_path, prompt_text):
        """Evaluate audio with specified prompt"""
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
                
                return result

            except Exception as e:
                print(f"Error calling judge model: {e}. Retrying...")
                time.sleep(5)
        
        print(f"Error: Unable to get valid response from judge model after multiple retries.")
        return None

def find_prompt_files(prompt_dir):
    """Auto discover two files in prompt directory, distinguish general and sample"""
    if not os.path.exists(prompt_dir):
        return None, None
    
    txt_files = [f for f in os.listdir(prompt_dir) if f.endswith('.txt')]
    
    if len(txt_files) != 2:
        return None, None
    
    general_prompt_path = None
    sample_prompt_path = None
    
    for txt_file in txt_files:
        file_path = os.path.join(prompt_dir, txt_file)
        if "sample" in txt_file.lower():
            sample_prompt_path = file_path
        else:
            general_prompt_path = file_path
    
    if general_prompt_path is None or sample_prompt_path is None:
        return None, None
    
    return general_prompt_path, sample_prompt_path

# Multithreaded evaluation function

def evaluate_single_model(args):
    """Evaluate single model function for multithreading"""
    (thread_id, dialogue_id, model_name, dialogue_data, 
     general_prompt, sample_prompt, output_file, completed_evaluations) = args
    
    try:
        # Check again if completed (prevent multithreading duplicate processing)
        with THREAD_LOCK:
            if is_evaluation_completed(dialogue_id, model_name, completed_evaluations):
                return {"status": "skipped", "thread_id": thread_id, "dialogue_id": dialogue_id, "model_name": model_name}
        
        print(f"\n--- Thread-{thread_id} | Evaluating model: {model_name} | Dialogue: {dialogue_id} ---")
        
        # Build dialogue audio for this model
        print(f"   Thread-{thread_id}: Concatenating audio...")
        audio_path = build_model_dialogue_audio_hybrid(dialogue_data, model_name, dialogue_id, thread_id)
        
        if not os.path.exists(audio_path):
            print(f"   Thread-{thread_id}: Error: Audio concatenation failed, skipping model {model_name}")
            return {"status": "failed", "thread_id": thread_id, "dialogue_id": dialogue_id, "model_name": model_name, "error": "audio_build_failed"}
        
        # Create judge model instance (independent for each thread)
        judge = RubricJudge(api_key=API_KEY, base_url=BASE_URL)
        
        # Evaluate with general prompt
        print(f"   Thread-{thread_id}: Evaluating with general prompt...")
        general_result = judge.evaluate_with_prompt(audio_path, general_prompt)
        
        if not general_result:
            print(f"   Thread-{thread_id}: General prompt evaluation failed, skipping this model.")
            os.remove(audio_path)
            return {"status": "failed", "thread_id": thread_id, "dialogue_id": dialogue_id, "model_name": model_name, "error": "general_eval_failed"}
        
        # Evaluate with sample prompt
        print(f"   Thread-{thread_id}: Evaluating with sample prompt...")
        sample_result = judge.evaluate_with_prompt(audio_path, sample_prompt)
        
        if not sample_result:
            print(f"   Thread-{thread_id}: Sample prompt evaluation failed, skipping this model.")
            os.remove(audio_path)
            return {"status": "failed", "thread_id": thread_id, "dialogue_id": dialogue_id, "model_name": model_name, "error": "sample_eval_failed"}
        
        # Combine results
        combined_result = {
            "dialogue_id": dialogue_id,
            "model_name": model_name,
            "general_evaluation": general_result,
            "sample_evaluation": sample_result,
            "data_source": dialogue_data["source"],
            "thread_id": thread_id,
            "timestamp": time.time()
        }
        
        # Thread-safe file writing
        with THREAD_LOCK:
            # Check again if already processed by other thread
            if is_evaluation_completed(dialogue_id, model_name, completed_evaluations):
                print(f"   Thread-{thread_id}: Evaluation already completed by other thread, skipping write.")
                os.remove(audio_path)
                return {"status": "skipped", "thread_id": thread_id, "dialogue_id": dialogue_id, "model_name": model_name}
            
            # Write to JSONL file
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(combined_result, ensure_ascii=False) + '\n')
            
            # Update completed records (in memory)
            evaluation_key = (dialogue_id, model_name)
            completed_evaluations.add(evaluation_key)
        
        print(f"   Thread-{thread_id}: Evaluation completed: {model_name} in dialogue {dialogue_id}")
        
        # Clean up temp audio file
        os.remove(audio_path)
        
        return {"status": "completed", "thread_id": thread_id, "dialogue_id": dialogue_id, "model_name": model_name}
        
    except Exception as e:
        print(f"   Thread-{thread_id}: Error during evaluation: {e}")
        return {"status": "error", "thread_id": thread_id, "dialogue_id": dialogue_id, "model_name": model_name, "error": str(e)}

def generate_evaluation_tasks(dialogues, completed_evaluations):
    """Generate all evaluation tasks to be performed"""
    tasks = []
    
    for dialogue_id, dialogue_data in dialogues.items():
        # Get rubric prompts
        general_prompt, sample_prompt = get_rubric_prompts_hybrid(dialogue_data, dialogue_id)
        
        if general_prompt is None or sample_prompt is None:
            print(f"   Skip dialogue {dialogue_id}: Cannot find rubric prompts")
            continue
        
        # Generate task for each model in this dialogue
        for model_name in dialogue_data["models"]:
            # Check if evaluation already completed
            if is_evaluation_completed(dialogue_id, model_name, completed_evaluations):
                continue
            
            task = (
                len(tasks) % MAX_WORKERS,  # thread_id
                dialogue_id,
                model_name,
                dialogue_data,
                general_prompt,
                sample_prompt,
                None,  # output_file will be set in main function
                None   # completed_evaluations will be set in main function
            )
            tasks.append(task)
    
    return tasks

def main():
    print(f"=== Hybrid Rubric Evaluation: {EVAL_TYPE.upper()} ===")
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

    print("2. Initializing judge model and progress recovery...")
    
    # Output file path
    output_file = os.path.join(OUTPUT_DIR, "rubric_evaluation_results_hybrid.jsonl")
    
    # Load completed evaluation records
    completed_evaluations = load_completed_evaluations(output_file)
    
    # Calculate progress statistics
    total_evaluations = sum(len(data["models"]) for data in dialogues.values())
    completed_count = len(completed_evaluations)
    print(f"   Evaluation progress: {completed_count}/{total_evaluations} ({completed_count/total_evaluations*100:.1f}%)")

    print("3. Generating evaluation tasks...")
    tasks = generate_evaluation_tasks(dialogues, completed_evaluations)
    
    # Update task parameters
    updated_tasks = []
    for task in tasks:
        updated_task = list(task)
        updated_task[6] = output_file  # output_file
        updated_task[7] = completed_evaluations  # completed_evaluations
        updated_tasks.append(tuple(updated_task))
    
    print(f"   Task statistics:")
    print(f"     - Total evaluations: {total_evaluations}")
    print(f"     - Completed: {completed_count}")
    print(f"     - Pending: {len(updated_tasks)}")
    print(f"     - Concurrent threads: {MAX_WORKERS}")
    
    if not updated_tasks:
        print("   All evaluations completed!")
        return

    print("4. Starting multithreaded rubric-based evaluation...")
    
    # Statistics variables
    completed_results = 0
    skipped_results = 0
    failed_results = 0
    error_results = 0
    
    # Execute tasks using thread pool
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(evaluate_single_model, task): task for task in updated_tasks}
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                status = result.get("status", "unknown")
                
                if status == "completed":
                    completed_results += 1
                    print(f"Thread-{result['thread_id']}: Completed evaluation {result['dialogue_id']} + {result['model_name']}")
                elif status == "skipped":
                    skipped_results += 1
                elif status == "failed":
                    failed_results += 1
                    print(f"Thread-{result['thread_id']}: Evaluation failed {result['dialogue_id']} + {result['model_name']} - {result.get('error', 'unknown')}")
                elif status == "error":
                    error_results += 1
                    print(f"Thread-{result['thread_id']}: System error {result['dialogue_id']} + {result['model_name']} - {result.get('error', 'unknown')}")
                
            except Exception as exc:
                error_results += 1
                print(f"Task execution exception: {exc}")

    print(f"\n5. Evaluation complete!")
    print(f"   Current run statistics:")
    print(f"     - Successfully completed: {completed_results}")
    print(f"     - Skipped already completed: {skipped_results}")
    print(f"     - Evaluation failed: {failed_results}")
    print(f"     - System errors: {error_results}")
    
    # Calculate final progress
    final_completed = completed_count + completed_results
    print(f"     - Final progress: {final_completed}/{total_evaluations} ({final_completed/total_evaluations*100:.1f}%)")
    print(f"\n   Results saved to: {output_file}")

if __name__ == "__main__":
    main()
