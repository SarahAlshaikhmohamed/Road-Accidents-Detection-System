# DB INSERT

import json, os
from sqlalchemy import text
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector, SearchType
from agno.knowledge.embedder.openai import OpenAIEmbedder

embedder = OpenAIEmbedder()

DATABASE_URL = os.getenv("DATABASE_URL")

knowledge_Accident = Knowledge(
        vector_db=PgVector(
            table_name="Accident_reports",
            db_url=DATABASE_URL,
            embedder=embedder,
            search_type=SearchType.hybrid
        ),
    )

knowledge_Pothole = Knowledge(
    vector_db=PgVector(
        table_name="Pothole_report",
        db_url=DATABASE_URL,  
        embedder=embedder,
        search_type=SearchType.hybrid,
    ),
)

# >> insert accident row into public.incidents
async def db_insert_accident( engine, knowledge, *,
    event_id: str,
    camera_id: str,
    time_utc: str,
    public_url: str,
    image_uuid: str,
    bbox_xyxy=None,
    mean_conf = None,
    latitude=None,
    longitude=None,
    vlm_result: dict | None = None,   #
):
    
    
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
        "mean_conf": mean_conf, 
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
    text_content = json.dumps(vlm_result).replace('"', '')
    content = f"latitude:{latitude} ,longitude:{longitude}  ,time_utc:{time_utc},content:{ text_content}  "
    metadata = {"image_id": image_uuid, "event_id": event_id, "camera_id": camera_id}

    await knowledge_Accident.add_content_async(
                text_content=content,
                metadata=metadata,
                name=image_uuid
                )

    return inserted_id

# >> insert pothole row into public.potholes + vector insert
async def db_insert_pothole(engine, knowledge_pothole,*,
    event_id: str,
    camera_id: str,
    time_utc: str,
    public_url: str,
    image_uuid: str,
    bbox_xyxy=None,
    mean_conf = None,
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
        "mean_conf": mean_conf,
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
    try:
        #if knowledge_pothole is not None:
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

            await knowledge_Pothole.add_content_async(
                        text_content=text_content,
                        metadata=metadata,
                        name=image_uuid
                        )
    except Exception as e:
        print("Warning: vector DB insert failed (pothole):", e)

    return inserted_id