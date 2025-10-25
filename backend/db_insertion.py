# DB INSERT

import json
from sqlalchemy import text


def normalize_vlm_result(vlm_result):
    if isinstance(vlm_result, str):
        try:
            return json.loads(vlm_result)
        except Exception:
            return {"error": "VLM returned non-JSON", "raw": vlm_result[:800]}
    if vlm_result is None:
        return {}
    if isinstance(vlm_result, dict):
        return vlm_result
    return {"error": "invalid_vlm_result_type", "type": str(type(vlm_result))}

# >> insert accident row into public.incidents
async def db_insert_accident( engine, knowledge, *,
    event_id: str,
    camera_id: str,
    time_utc: str,
    public_url: str,
    image_uuid: str,
    bbox_xyxy=None,
    latitude=None,
    longitude=None,
    vlm_result: dict | None = None,   #
):
    vlm_result = normalize_vlm_result(vlm_result)
    # >> confidences 
    conf = vlm_result.get("Confidence", {}) or {}

    def get_conf(key):
        try:
            v = conf.get(key)
            if v is not None:
                return float(v) 
        except Exception:
            return None

    params = {
        "event_id": event_id,
        "camera_id": camera_id,
        "time_utc": time_utc,
        "bbox_xyxy": bbox_xyxy,
        "mean_conf": None, 
        "thumbnail_url": public_url,
        "image_id": image_uuid,
        "latitude": latitude,
        "longitude": longitude,
        "status": "NEW",
        # parsed VLM fields 
        "category": vlm_result.get("Category"),
        "contact_level": vlm_result.get("Contact_level"),
        "derivation_object": vlm_result.get("Derivation_object"),
        "environment": vlm_result.get("Environment"),
        "time_of_day": vlm_result.get("Time"),
        "traffic_lane_of_the_object": vlm_result.get("Traffic_lane_of_the_object"),
        "weather": vlm_result.get("Weather"),
        "severity": vlm_result.get("Severity"),
        "emergency_lights": vlm_result.get("Emergency_lights"),
        "vehicles_count": int(vlm_result.get("Vehicles_count") or 0),

        "conf_category": get_conf("Category"),
        "conf_contact_level": get_conf("Contact_level"),
        "conf_derivation_object": get_conf("Derivation_object"),
        "conf_environment": get_conf("Environment"),
        "conf_time": get_conf("Time"),
        "conf_traffic_lane": get_conf("Traffic_lane_of_the_object"),
        "conf_weather": get_conf("Weather"),
        "conf_severity": get_conf("Severity"),
        "conf_emergency_lights": get_conf("Emergency_lights"),
        "conf_vehicles_count": get_conf("Vehicles_count"),

        "raw_report": json.dumps(vlm_result),
    }

    insert_stmt = text("""
        INSERT INTO public.incidents
        (event_id, camera_id, time_utc, bbox_xyxy, mean_conf, thumbnail_url, image_id,
         latitude, longitude, status,
         category, contact_level, derivation_object, environment, time_of_day,
         traffic_lane_of_the_object, weather, severity, emergency_lights, vehicles_count,
         conf_category, conf_contact_level, conf_derivation_object, conf_environment,
         conf_time, conf_traffic_lane, conf_weather, conf_severity, conf_emergency_lights, conf_vehicles_count,
         raw_report)
        VALUES
        (:event_id, :camera_id, :time_utc, :bbox_xyxy, :mean_conf, :thumbnail_url, :image_id,
         :latitude, :longitude, :status,
         :category, :contact_level, :derivation_object, :environment, :time_of_day,
         :traffic_lane_of_the_object, :weather, :severity, :emergency_lights, :vehicles_count,
         :conf_category, :conf_contact_level, :conf_derivation_object, :conf_environment,
         :conf_time, :conf_traffic_lane, :conf_weather, :conf_severity, :conf_emergency_lights, :conf_vehicles_count,
         :raw_report)
        RETURNING id
    """)

    # >> insert row
    with engine.connect() as conn:
        res = conn.execute(insert_stmt, params)
        conn.commit()
        inserted_id = res.scalar()

    # >> Accident_reports
    # >> optional: add to vector DB 
    try:
        if knowledge is not None and vlm_result is not None:
            text_content = json.dumps(vlm_result, ensure_ascii=False).replace('"', '')
            metadata = {"image_id": image_uuid, "event_id": event_id, "camera_id": camera_id}
            await knowledge.add_content_async(text_content=text_content, metadata=metadata)
    except Exception as e:
        print("Warning: vector DB insert failed:", e)

    return inserted_id

# >> insert pothole row into public.potholes + vector insert
async def db_insert_pothole(engine, knowledge_pothole,*,
    event_id: str,
    camera_id: str,
    time_utc: str,
    public_url: str,
    image_uuid: str,
    bbox_xyxy=None,
    latitude=None,
    longitude=None,
    size: str | None = None,
    notes: str | None = None,
):
    params = {
        "event_id": event_id,
        "camera_id": camera_id,
        "time_utc": time_utc,
        "bbox_xyxy": bbox_xyxy,
        "mean_conf": None,
        "thumbnail_url": public_url,
        "image_id": image_uuid,
        "latitude": latitude,
        "longitude": longitude,
        "status": "NEW",
        "size": size,
        "notes": notes,
        "raw_payload": json.dumps({}),
    }

    insert_stmt = text("""
        INSERT INTO public.potholes
        (event_id, camera_id, time_utc, bbox_xyxy, mean_conf, thumbnail_url, image_id,
         latitude, longitude, status, size, notes, raw_payload)
        VALUES
        (:event_id, :camera_id, :time_utc, :bbox_xyxy, :mean_conf, :thumbnail_url, :image_id,
         :latitude, :longitude, :status, :size, :notes, :raw_payload)
        RETURNING id
    """)

    with engine.connect() as conn:
        res = conn.execute(insert_stmt, params)
        conn.commit()
        inserted_id = res.scalar()

    # >> Pothole report
    # >> optional: add to vector DB  
    try:
        if knowledge_pothole is not None:
            pothole_result = {
                "size": size,
                "latitude": latitude,
                "longitude": longitude,
                "time_utc": time_utc,
                "notes": notes,
                "image_uuid": image_uuid
            }
            text_content = json.dumps(pothole_result, ensure_ascii=False).replace('"', '')
            metadata = {"image_id": image_uuid, "event_id": event_id, "camera_id": camera_id}
            await knowledge_pothole.add_content_async(text_content=text_content, metadata=metadata, name=image_uuid)
    except Exception as e:
        print("Warning: vector DB insert failed (pothole):", e)

    return inserted_id