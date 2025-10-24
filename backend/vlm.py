# Vision Language Model

import os, json
from agno.agent import Agent
from agno.media import Image
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini

gemini_api_key  = os.getenv("GEMINI_API_KEY", "")  
openai_api_key  = os.getenv("OPENAI_API_KEY", "")

inst = """You are given a image of a traffic incident. Return a single-line JSON format exactly matching the schema below.
Only output valid JSON and no extra explanation or commentary.

Schema:
 {
 "Category": ["Pedestrian","Vehicle","Bicycle","Other","Unknown"],
 "Contact_level": ["Collision","Near-Miss","No-contact","Unknown"],
 "Derivation_object": ["Another vs Others","Self vs Object","Environment","Unknown"],
 "Environment": ["Highway","Major road","Urban","Residential","Unknown"],
 "Time": ["Daytime","Nighttime","Dawn/Dusk","Unknown"],
 "Traffic_lane_of_the_object": ["Right-hand traffic","Left-hand traffic","Unknown"],
 "Weather": ["Sunny/Cloudy","Rainy","Snowy","Foggy","Unknown"],
 "Severity": ["Low","Medium","High","Unknown"],
 "Emergency_lights": ["True","False","Unknown"],
 "Vehicles_count": "integer or 0",
 "Confidence": { "Category": "0.0-1.0", "Contact_level":"0.0-1.0", "Derivation_object":"0.0-1.0", "Environment":"0.0-1.0", "Time":"0.0-1.0", "Traffic_lane_of_the_object":"0.0-1.0", "Weather":"0.0-1.0", "Severity":"0.0-1.0", "Emergency_lights":"0.0-1.0", "Vehicles_count":"0.0-1.0" },
 "Image_id": "string or null",
 "Evidence": ["short strings..."]
}

Example:
{"Category":"Pedestrian","Contact_level":"Near-Miss","Derivation_object":"Another vs Others","Environment":"Major road","Time":"Daytime","Traffic_lane_of_the_object":"Right-hand traffic","Weather":"Sunny/Cloudy"}"""

vlm_model = OpenAIChat(id=os.getenv("OPENAI_MODEL_ID", "gpt-4.1"), api_key=openai_api_key)
# vlm_model = Gemini(id=os.getenv("GEMINI_MODEL_ID", "gemini-2.5-pro"), api_key=gemini_api_key)

VLM_agent = Agent(
    model=vlm_model,
    name="VLM agent",
    # knowledge=pdf_knowledge_base,
    # search_knowledge=True,
    # storage=SqliteStorage(table_name="RAG_agent", db_file=agent_storage),
    # db = db,
    read_chat_history=True,
    # add_history_to_messages=True,
    read_tool_call_history=True,
    instructions=inst,
    add_history_to_context=True,
    num_history_runs=3,
)

#def vlm_analyze_image(image_path: str, prompt: str) -> str:
#    """
#    input:  image file path and prompt.
#    Output: raw text from the agent.
#    """
#    resp = VLM_agent.run(
#        [prompt, Image(filepath=image_path)]
#    )
#    return str(resp).strip()
def vlm_analyze_image(image_path: str) -> str:
    prompt = "Now analyze the image and output the JSON format following the schema above."
    resp = VLM_agent.run([prompt, Image(filepath=image_path)])
    
    # Extract actual text from RunOutput object
    txt = (
        getattr(resp, "output_text", None)
        or getattr(resp, "text", None)
        or getattr(resp, "content", None)
    )
    if txt is None:
        try:
            txt = resp.messages[-1].content  # last fallback
        except Exception:
            txt = str(resp)

    return str(txt).strip()
