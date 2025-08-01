import streamlit as st
import os
import json
from dotenv import load_dotenv

from core import (
    schema_validator,
    prompt_engine,
    llm_interface,
    json_extractor,
    logger_service,
    session_manager,
    token_estimator
)

load_dotenv()

st.set_page_config(page_title="AI JSON Extractor", layout="centered")
st.title("🧠 Structured JSON Extractor")

schema_file = st.file_uploader("Upload JSON Schema", type=["json"])
text_file = st.file_uploader("Upload Text File", type=["txt", "pdf"])

if st.button("Generate JSON"):
    if not schema_file or not text_file:
        st.error("Please upload both JSON schema and input text.")
    else:
        schema_str = schema_file.read().decode()
        text_str = text_file.read().decode()

        schema_json = json.loads(schema_str)
        is_valid, schema_err = schema_validator.is_valid_schema(schema_json)

        if not is_valid:
            st.error(f"Invalid schema: {schema_err}")
        else:
            session_id = session_manager.create_session()
            logger_service.log(session_id, "schema", schema_str)
            logger_service.log(session_id, "text", text_str)

            st.info(f"🔢 Estimated tokens: {token_estimator.estimate_tokens(schema_str + text_str)}")

            err = None
            success = False

            for attempt in range(3):
                with st.status(f"🧪 Attempt {attempt+1}...", expanded=True):
                    prompt = prompt_engine.create_prompt(schema_str, text_str, error=err)
                    logger_service.log(session_id, f"prompt_{attempt+1}", prompt)

                    output = llm_interface.call_llm(prompt)
                    logger_service.log(session_id, f"output_{attempt+1}", output)

                    try:
                        result = json_extractor.extract_json(output)
                        valid, err = json_extractor.validate_against_schema(schema_json, result)
                        if valid:
                            success = True
                            with open(f"logs/{session_id}/final_output.json", "w") as f:
                                json.dump(result, f, indent=2)
                            st.success("✅ JSON generated successfully!")
                            st.json(result)
                            st.download_button("📥 Download JSON", data=json.dumps(result, indent=2), file_name="output.json")
                            break
                    except Exception as e:
                        err = str(e)
                        st.warning(f"⚠️ Error: {err}")

            if not success:
                st.error("❌ Failed to extract valid JSON after 3 attempts.")
                st.download_button("📄 Download Logs", data=json.dumps({"error": err}), file_name=f"{session_id}_error.log")
