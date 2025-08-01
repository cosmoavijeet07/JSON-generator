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

# === 🔘 Model Selector ===
model_map = {
    "GPT 4.1": "gpt-4.1-2025-04-14",
    "GPT O4 Mini": "o4-mini-2025-04-16"
}

selected_model_label = st.selectbox(
    "Choose Model:",
    options=list(model_map.keys()),
    index=0
)

selected_model = model_map[selected_model_label]

st.markdown(
    f"""
**Model Notes**  
- 🧠 **GPT 4.1**: Recommended for well-structured small/medium schemas with fewer fields  
- 🧩 **GPT O4 Mini**: Best for very large schemas, complex nesting, and long input contexts (up to 100k tokens)
"""
)

# === 📝 File Uploads ===
schema_file = st.file_uploader("Upload JSON Schema", type=["json"])
text_file = st.file_uploader("Upload Text File", type=["txt", "pdf"])

generated_result = None
session_id = None

if st.button("Generate JSON"):
    if not schema_file or not text_file:
        st.error("Please upload both JSON schema and input text.")
    else:
        schema_str = schema_file.read().decode()
        text_str = text_file.read().decode()

        schema_json = json.loads(schema_str)
        is_valid, schema_err = schema_validator.is_valid_schema(schema_json)

        if not is_valid:
            st.error(f"❌ Invalid schema: {schema_err}")
        else:
            session_id = session_manager.create_session()
            logger_service.log(session_id, "model_used", selected_model)
            logger_service.log(session_id, "schema", schema_str)
            logger_service.log(session_id, "text", text_str)

            est_tokens = token_estimator.estimate_tokens(schema_str + text_str)
            st.info(f"🔢 Estimated tokens for input: {est_tokens}")

            err = None
            success = False

            for attempt in range(3):
                with st.status(f"🧪 Attempt {attempt+1}...", expanded=True):
                    prompt = prompt_engine.create_prompt(schema_str, text_str, error=err)
                    logger_service.log(session_id, f"prompt_{attempt+1}", prompt)

                    output = llm_interface.call_llm(prompt, model=selected_model)
                    logger_service.log(session_id, f"output_{attempt+1}", output)

                    try:
                        result = json_extractor.extract_json(output)
                        valid, err = json_extractor.validate_against_schema(schema_json, result)
                        if valid:
                            success = True
                            generated_result = result
                            output_path = f"logs/{session_id}/final_output.json"
                            with open(output_path, "w") as f:
                                json.dump(result, f, indent=2)
                            break
                    except Exception as e:
                        err = str(e)
                        st.warning(f"⚠️ Error: {err}")

            if not success:
                st.error("❌ Failed to extract valid JSON after 3 attempts.")
                st.download_button(
                    "📄 Download Logs",
                    data=json.dumps({"error": err}),
                    file_name=f"{session_id}_error.log"
                )
            else:
                st.success("✅ JSON generated successfully!")

# === 📤 Post-Generation Actions ===
if generated_result:
    st.markdown("### 🎯 Extracted JSON")
    st.json(generated_result)

    st.download_button(
        "📥 Download JSON",
        data=json.dumps(generated_result, indent=2),
        file_name="output.json"
    )

    if session_id:
        st.download_button(
            "📄 Download Logs",
            data="\n".join(
                open(f"logs/{session_id}/{f}", encoding="utf-8").read()
                for f in os.listdir(f"logs/{session_id}")
                if f.endswith(".log")
            ),
            file_name=f"{session_id}_session.log"
        )
