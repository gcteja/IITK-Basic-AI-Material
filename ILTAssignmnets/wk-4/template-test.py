"""
This program is build with Flan-T5-XL LLM to be able to determine output of a MCQ question with four options. 

> It accepts five parameters provided as a command line input. 
> The first input represents the question and the next four input are the options. 
> The output should be the option number: A/B/C/D 
> Output should be upper-case
> There should be no additional output including any warning messages in the terminal.
> Remember that your output will be tested against test cases, therefore any deviation from the test cases will be considered incorrect during evaluation.
"""

"""
ALERT: * * * No changes are allowed to import statements  * * *
"""
import sys
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import os  # Added for dynamic path handling

##### You may comment this section to see verbose -- but you must un-comment this before final submission. ######
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
#################################################################################################################

"""
* * * Changes allowed from here  * * * """

def llm_function(model, tokenizer, q, a, b, c, d):
    '''
    The steps are given for your reference:
    1. Properly formulate the prompt
    2. Tokenize the prompt
    3. Pass to model to get deterministic output
    4. Extract and 5. Clean output
    6. Output is case-sensitive: A, B, C or D
    '''
    # 1. Formulate the prompt [cite: 10, 11]
    # We use a structured format to help Flan-T5-XL identify the options [cite: 10]
    prompt = (
        f"Question: {q}\n"
        f"Options:\n(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\n"
        "Constraint: Answer with only the letter (A, B, C, or D) of the correct option.\n"
        "Answer:"
    )

    # 2. Tokenize the prompt [cite: 11]
    inputs = tokenizer(prompt, return_tensors="pt")

    # 3. Pass to model to get output (max_new_tokens=2 for deterministic letter) [cite: 11]
    output_tokens = model.generate(**inputs, max_new_tokens=2)
    
    # 4. Extract the correct option from the model [cite: 15]
    decoded_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip().upper()
    
    # 5. Clean output and return A/B/C/D [cite: 15]
    # Use regex to ensure only the letter is returned [cite: 15]
    match = re.search(r'[A-D]', decoded_output)
    final_output = match.group(0) if match else "A"

    return final_output

"""
ALERT: * * * No changes are allowed below this comment  * * *
"""

if __name__ == '__main__':
    question = sys.argv[1].strip()
    option_a = sys.argv[2].strip()
    option_b = sys.argv[3].strip()
    option_c = sys.argv[4].strip()
    option_d = sys.argv[5].strip()

    ##################### Loading Model and Tokenizer ########################
    # Dynamically find the path: one level up and then into 'pretrainedmodels'
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    model_folder_path = os.path.join(current_script_path, "..", "pretrainedmodels")
    
    # Check if path exists to prevent EnvironmentError
    if not os.path.exists(model_folder_path):
        # Fallback to the direct path shown in your terminal if relative path fails
        model_folder_path = "/Users/g0c0715/Library/CloudStorage/OneDrive-WalmartInc/Desktop/Work/Learnings-KTs/GenAI-IIT-Kharagpur/InstructorLed-sessions/Assignments/pretrainedmodels"

    tokenizer = T5Tokenizer.from_pretrained(model_folder_path, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(model_folder_path, local_files_only=True)
    ##########################################################################

    """  Call to function that will perform the computation. """
    torch.manual_seed(42)
    out = llm_function(model,tokenizer,question,option_a,option_b,option_c,option_d)
    print(out.strip())

    """ End to call """