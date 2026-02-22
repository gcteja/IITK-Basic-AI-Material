"""
This program is build with Flan-T5-XL LLM to be able to answer the final question using the output from the previous questions as in-context learning/few-shot learning. 

Consider three related questions from a search session: Question 1, Question 2, Question 3
1. Answer to Question 1 needs to be generated. 
2. Answer to Question 2 needs to be generated with the answer to Question 1 as one-shot example / context. 
3. Answer to Question 3 needs to be generated with the answer to Question 2 as one-shot example / context.
4. Answer to Question 3 will be either YES or NO and nothing else.


> The program accepts three parameters provided as a command line input. 
> The three inputs represent the questions.
> The output of the first two question is Generation based whereas the last question output is deterministic i.e. its either YES or NO.
> Output should be in upper-case: YES or NO
> There should be no additional output including any warning messages in the terminal.
> Remember that your output will be tested against test cases, therefore any deviation from the test cases will be considered incorrect during evaluation.


Syntax: python template.py <string> <string> <string> 

The following example is given for your reference:

 Terminal Input: python template.py "Who is Rabindranath Tagore?" "Where was he born?" "Is it in America?"
Terminal Output: NO

 Terminal Input: python template.py "Who is Rabindranath Tagore?" "Where was he born?" "Is it in India?"
Terminal Output: YES

You are expected to create some examples of your own to test the correctness of your approach.

Following exmales I tried:

Terminal Input: python template.py "Who is Rabindranath Tagore?" "Where was he born?" "Is it in India?"
Terminal Output: YES

Terminal Input: python template.py "Who is Elon Musk?" "Which company is he associated with?" "Is that company Tesla?"
Terminal Output: YES

Terminal Input: python template.py "Who is Albert Einstein?" "Was he a physicist?" "Was he born in America?"
Terminal Output: NO 

Terminal Input: python template.py "What country is the Taj Mahal in?" "Is it located in India?" "Is India in Asia?"
Terminal Output: YES

Terminal Input: python template.py "What country is the Great Wall in?" "Is it in China?" "Is China in Europe?"
Terminal Output: NO

Terminal Input: python template.py "A triangle has how many sides?" "Is that number 3?" "Is 3 less than 5?"
Terminal Output: NO --> exepcetd is YES. But getting output as NO. 

Terminal Input: python template.py "A triangle has how many sides?" "Is that number 3?" "Is 3 less than 4?"
Terminal Output: YES

Terminal Input: python template.py "How many days are there in a week?" "Is it 7?" "Is 7 more than 10?"
Terminal Output: NO

Terminal Input: python template.py "What is 4 plus 4?" "Is the answer 8?" "Is that number even?"
Terminal Output: YES

Terminal Input: python template.py "What is 7 times 2?" "Is the answer 14?" "Is 14 divisible by 7?"
Terminal Output: YES

Terminal Input: python template.py "What is the capital of France?" "Is it famous for the Eiffel Tower?" "Is the city Paris?"
Terminal Output: YES


ALL THE BEST!!
"""

"""
ALERT: * * * No changes are allowed to import statements  * * *
"""
import sys
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

##### You may comment this section to see verbose -- but you must un-comment this before final submission. ######
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
#################################################################################################################

"""
* * * Changes allowed from here  * * * 
"""

def llm_function(model,tokenizer,questions):
    '''
    The steps are given for your reference:

    1. Generate answer for the first question.
    2. Generate answer for the second question use the answer for first question as context.
    3. Generate a deterministic output either 'YES' or 'NO' for the third question using the context from second question.  
    5. Clean output and return.
    6. Output is case-sensative: YES or NO
    Note: The model (Flan-T5-XL) and tokenizer is already initialized. Do not modify that section.
    '''
    q1, q2, q3 = questions
    # --------- Helper function for generation ----------
    def generate_answer(prompt, max_len=64):
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_len,
                do_sample=False
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # --------- Step 1: Answer Question 1 ----------
    prompt1 = f"Question: {q1}\nAnswer:"
    answer1 = generate_answer(prompt1)

    # --------- Step 2: Answer Question 2 with Q1 context ----------
    prompt2 = f"Question: {q1}\nAnswer: {answer1}\n\nQuestion: {q2}\nAnswer:"
    answer2 = generate_answer(prompt2)

    # --------- Step 3: Deterministic YES/NO for Question 3 ----------
    prompt3 = f"Question: {q2}\nAnswer: {answer2}\n\nQuestion: {q3}\nAnswer YES or NO only:"
    answer3 = generate_answer(prompt3, max_len=8)

    # --------- Clean and Normalize Output ----------
    answer3 = answer3.upper()

    if "YES" in answer3:
        final_output = "YES"
    else:
        final_output = "NO"
    return final_output

"""
ALERT: * * * No changes are allowed below this comment  * * *
"""

if __name__ == '__main__':

    question_a = sys.argv[1].strip()
    question_b = sys.argv[2].strip()
    question_c = sys.argv[3].strip()

    questions = [question_a, question_b, question_c]
    ##################### Loading Model and Tokenizer ########################
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    ##########################################################################

    """  Call to function that will perform the computation. """
    torch.manual_seed(42)
    out = llm_function(model,tokenizer,questions)
    print(out.strip())

    """ End to call """
