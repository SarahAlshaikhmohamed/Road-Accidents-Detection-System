# Vision Language Model

import os, json, tempfile, traceback
from agno.agent import Agent
from agno.media import Image
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini

openai_api_key  = os.getenv("OPENAI_API_KEY", "")

# >> VLM Instructions
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

# >> Model Initialization
vlm_model = OpenAIChat(id=os.getenv("OPENAI_MODEL_ID", "gpt-5"), api_key=openai_api_key)

VLM_agent = Agent(
    model=vlm_model,
    name="VLM agent",
    read_chat_history=True,
    read_tool_call_history=True,
    instructions=inst,
    add_history_to_context=True,
    num_history_runs=3,
)

# >> Extract Json
def _extract_json_from_text(text: str) -> dict:
    """Extract JSON from text output, removing any stray content."""
    if not isinstance(text, str):
        raise ValueError("VLM agent returned non-text output")

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in VLM output")

    json_text = text[start:end]
    return json.loads(json_text)


# >> Analyze image with the VLM
def vlm_analyze_image(image_path: str) -> str:
    """"Analyze an accident image using VLM agent and returns JSON strings"""
    prompt = "Now analyze the image and output the JSON format following the schema above."
    try:
        response = VLM_agent.run([prompt, 
                            Image(filepath=image_path)],
                            markdown = True)
    
    
        # Extract text 
        raw_text = getattr(response ,"Content", None)
        if raw_text is None:
            raw_text = getattr(response, "output_text", None) or getattr(response, "text", None)
        if raw_text is None:
            raw_text = str(response)

        # Parse JSON
        parsed = _extract_json_from_text(raw_text)
        return json.dumps(parsed, ensure_ascii=False)
    
    except Exception as e:
        # very detailed debug payload and raise a RuntimeError
        tb = traceback.format_exc()
        debug_info = {
            "error": str(e),
            "traceback": tb
        }
        # Return error info
        return json.dumps({"error": "VLM analysis failed", "debug": debug_info})

