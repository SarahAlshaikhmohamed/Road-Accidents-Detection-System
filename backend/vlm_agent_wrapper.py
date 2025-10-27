# backend/vlm_agent_wrapper.py

import os
import json
import tempfile
import traceback

from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.media import Image
from agno.models.openai import OpenAIChat
from agno.knowledge.embedder.openai import OpenAIEmbedder

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize 
embeddings = OpenAIEmbedder()
VLM_agent = Agent(
    model=OpenAIChat(id="gpt-4.1", api_key=OPENAI_API_KEY),
    name="VLM agent",
    read_chat_history=True,
    read_tool_call_history=True,
    instructions="""You are given a image of a traffic incident. Return a single-line JSON format exactly matching the schema below.
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
{"Category":"Pedestrian","Contact_level":"Near-Miss","Derivation_object":"Another vs Others","Environment":"Major road","Time":"Daytime","Traffic_lane_of_the_object":"Right-hand traffic","Weather":"Sunny/Cloudy"}""",
    add_history_to_context=True,
    num_history_runs=3,
)

def _extract_json_from_text(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError("VLM agent returned non-text")
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == -1:
        raise ValueError("No JSON found in agent output")
    json_text = text[start:end]
    return json.loads(json_text)

def analyze_image_with_vlm(image_bytes: bytes, image_id: str = None) -> dict:
    """
    Save bytes, call agent, return parsed dict.
    On failure, raise Exception with detailed info.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        # call agent
        response = VLM_agent.run(
            "Now analyze the accident image and output the JSON format following the schema above.",
            images=[Image(filepath=tmp_path)],
            markdown=True
        )
        # per your example, response.content is where the text lives
        raw_text = getattr(response, "content", None)
        if raw_text is None:
            # try str fallback
            raw_text = response if isinstance(response, str) else str(response)
        # try parse
        parsed = _extract_json_from_text(raw_text)
        return parsed
    except Exception as e:
        # prepare a very detailed debug payload and raise a RuntimeError
        tb = traceback.format_exc()
        debug_info = {
            "error": str(e),
            "traceback": tb,
            "raw_response_preview": None
        }
        # try to include raw text if exists
        try:
            debug_info["raw_response_preview"] = raw_text[:200] if raw_text else None
        except:
            debug_info["raw_response_preview"] = None
        # raise with debug info embedded
        raise RuntimeError(f"VLM analysis failed: {json.dumps(debug_info)}")
