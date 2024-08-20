from prompt import PromptChatOpenAIFormatter, PromptDefaultFormatter, PromptFormatter
from dataset_manager import DatasetManager
from test_dataset_manager import create_validated_dataset

PROMPT_FORMAT = """<start_of_turn>user
{system}

{instruction}
<end_of_turn>
<start_of_turn>model
{output}
<end_of_turn>
"""

data_path = './sample.jsonl'

def generate_text_field(data):
    messages = data["messages"]
    system = ""
    instruction = ""
    output = ""
    for message in messages:
        if message["role"] == "system":
            system = message["content"]
        elif message["role"] == "user":
            instruction = message["content"]
        elif message["role"] == "assistant":
            output = message["content"]  
    full_prompt = PROMPT_FORMAT.format(system=system, instruction=instruction, output=output) 
    return {"text": full_prompt}

def test_prompt_format():
    formatter = PromptChatOpenAIFormatter.get_prompt_format()
    assert formatter == PROMPT_FORMAT

def test_full_prompt():
    test_dataset = DatasetManager(data_path, 'chat_openai').dataset
    validated_dataset = create_validated_dataset()
    assert test_dataset['text'] == validated_dataset['text']