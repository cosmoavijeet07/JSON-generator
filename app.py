import streamlit as st
import os, json
from dotenv import load_dotenv

from core import (
    schema_validator,
    prompt_engine,
    llm_interface,
    json_extractor,
    logger_service,
    session_manager,
    token_estimator,
    semantic_chunker,
    aggregator,
    iterative_extractor,
    preprocessor
)
load_dotenv()

st.set_page_config(page_title="AI JSON Extractor", layout="centered")
st.title("🧠 Structured JSON Extractor")

ADV = st.sidebar.expander("⚙️ Advanced Settings", expanded=False)
chunk_size = ADV.slider("Chunk token size", 1000, 8000, 3500)
overlap = ADV.slider("Chunk overlap", 0, 700, 200)
passes = ADV.slider("Max extraction passes per chunk", 1, 4, 2)

model_map = {
    "GPT 4.1": "gpt-4.1-2025-04-14",
    "GPT O4 Mini": "o4-mini-2025-04-16"
}

selected_model = st.selectbox("Choose Model:", list(model_map.values()), 0)
schema_file = st.file_uploader("Upload JSON Schema", type=["json"])
text_file = st.file_uploader("Upload Text File", type=["txt", "pdf"])

if st.button("Generate JSON"):
    if not schema_file or not text_file:
        st.error("Please upload both JSON schema and input text.")
    else:
        schema_str = schema_file.read().decode()
        text_str = text_file.read().decode()
        try:
            schema_json = json.loads(schema_str)
        except Exception as e:
            st.error(f"Schema JSON is invalid: {e}")
            st.stop()

        is_valid, schema_err = schema_validator.is_valid_schema(schema_json)
        if not is_valid:
            st.error(f"❌ Invalid schema: {schema_err}")
            st.stop()

        session_id = session_manager.create_session()
        logger_service.log(session_id, "model_used", selected_model)
        text_cleaned = preprocessor.clean_ocr(text_str)
        chunks = semantic_chunker.split_semantic(text_cleaned, max_len=chunk_size, overlap=overlap)
        st.info(f"Chunked input into {len(chunks)} sections.")
        def llm_fn(prompt): return llm_interface.call_llm(prompt, model=selected_model)
        results = iterative_extractor.iterative_schema_extract(
            llm_fn, prompt_engine, schema_str, chunks, json_extractor.validate_against_schema, max_passes=passes
        )
        logger_service.log(session_id, "raw_outputs", json.dumps(results, indent=2))
        final = aggregator.merge_json_outputs(results, schema_json)
        with open(f"logs/{session_id}/final_output.json", "w") as f:
            json.dump(final, f, indent=2)
        st.success("Extraction complete!")
        st.json(final)
        st.download_button("Download JSON", data=json.dumps(final, indent=2), file_name="output.json")
