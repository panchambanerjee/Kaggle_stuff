import csv
import json
import time
import os
# pip install google-genai
from google import genai
from google.genai import types

# 1. Setup your API Key
from dotenv import load_dotenv
load_dotenv()
client = genai.Client()

TRAIN_CSV = 'train.csv'
OUTPUT_JSONL = 'new_equation_cots.jsonl'

def generate_reasoning(prompt_text):
    # We append the exact same Kaggle suffix the author used
    full_prompt = prompt_text + "\n\nPlease put your final answer inside \\boxed{}. For example: \\boxed{your answer}\nLet's think step by step to deduce the secret rules."
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7, # The author used 0.7 to get diverse reasoning paths
                max_output_tokens=2048,
            )
        )
        return response.text
    except Exception as e:
        print(f"API Error: {e}")
        return ""

def extract_boxed_answer(text):
    import re
    # Find the last occurrence of \boxed{...}
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        return matches[-1].strip()
    return None

def main():
    success_count = 0
    with open(TRAIN_CSV, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            if len(row) < 3: continue
            id_, prompt, true_answer = row[0], row[1], row[2]
            
            # Focus strictly on Equation Transformation
            if "secret set of transformation rules is applied to equations" in prompt:
                print(f"Processing {id_}...")
                
                # Ask the LLM to solve it
                generated_cot = generate_reasoning(prompt)
                if not generated_cot:
                    continue
                    
                # Extract the LLM's answer
                llm_answer = extract_boxed_answer(generated_cot)
                
                # Rule-based correctness filtering (Exact match for equations)
                if llm_answer == true_answer:
                    print(f"✅ SUCCESS! Found valid CoT for {id_}")
                    
                    # Save to our new dataset
                    with open(OUTPUT_JSONL, 'a') as out_f:
                        json.dump({
                            "id": id_,
                            "prompt": prompt,
                            "answer": true_answer,
                            "generated_cot": generated_cot,
                            "type": "Equation Transformation"
                        }, out_f)
                        out_f.write('\n')
                        
                    success_count += 1
                else:
                    print(f"❌ Failed. LLM guessed: {llm_answer}, True: {true_answer}")
                
                time.sleep(2) # Avoid rate limits

    print(f"\nFinished! Successfully generated {success_count} new high-quality CoT samples.")

if __name__ == "__main__":
    main()
