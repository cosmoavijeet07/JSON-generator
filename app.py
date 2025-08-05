# app.py (COMPLETELY UPDATED)
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
from core.pipeline_orchestrator import PipelineOrchestrator

load_dotenv()
st.set_page_config(page_title="Enhanced AI JSON Extractor", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.processing-mode {
    background-color: #f0f8ff;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 1rem 0;
}
.cost-estimate {
    background-color: #fff8dc;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #ffa500;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🧠 Enhanced Structured JSON Extractor</h1>', unsafe_allow_html=True)
st.markdown("**Advanced pipeline for complex, unstructured, and long documents**")

# === Processing Mode Selection ===
st.markdown('<div class="processing-mode">', unsafe_allow_html=True)
st.subheader("🔧 Processing Mode")

processing_mode = st.radio(
    "Choose processing approach:",
    ["Simple Mode (Original)", "Enhanced Mode (Complex Documents)"],
    index=1,
    help="Enhanced Mode uses advanced chunking, context awareness, and multi-pass extraction for complex documents"
)

if processing_mode == "Enhanced Mode (Complex Documents)":
    st.info("✨ Enhanced mode includes: semantic chunking, context awareness, quality assurance, and multi-pass extraction")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        chunk_size = st.slider("Max Chunk Size (characters)", 1000, 4000, 2000)
        overlap_size = st.slider("Chunk Overlap (characters)", 100, 500, 200)
        enable_qa = st.checkbox("Enable Quality Assurance", value=True)
        confidence_threshold = st.slider("Minimum Confidence Threshold", 0.0, 1.0, 0.5)

st.markdown('</div>', unsafe_allow_html=True)

# === Model Selector ===
model_map = {
    "GPT 4.1 (Recommended)": "gpt-4.1-2025-04-14",
    "GPT O4 Mini (Large Documents)": "o4-mini-2025-04-16"
}

selected_model_label = st.selectbox(
    "Choose Model:",
    options=list(model_map.keys()),
    index=0
)
selected_model = model_map[selected_model_label]

st.markdown(f"""
**Model Notes**
- 🧠 **GPT 4.1**: Best for well-structured schemas and moderate complexity
- 🧩 **GPT O4 Mini**: Optimized for very large documents and complex nested schemas
""")

# === File Uploads ===
col1, col2 = st.columns(2)

with col1:
    schema_file = st.file_uploader("📋 Upload JSON Schema", type=["json"])
    if schema_file:
        schema_content = schema_file.read().decode()
        try:
            schema_json = json.loads(schema_content)
            st.success("✅ Schema loaded successfully")
            
            # Show schema preview
            with st.expander("👀 Schema Preview"):
                st.json(schema_json)
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON schema: {e}")
            schema_json = None

with col2:
    text_file = st.file_uploader("📄 Upload Text File", type=["txt", "pdf"])
    if text_file:
        text_content = text_file.read().decode()
        st.success(f"✅ Text loaded ({len(text_content)} characters)")
        
        # Show text preview
        with st.expander("👀 Text Preview"):
            st.text_area("First 500 characters:", text_content[:500], height=150, disabled=True)

# === Cost Estimation ===
if schema_file and text_file and 'schema_json' in locals() and schema_json:
    st.markdown('<div class="cost-estimate">', unsafe_allow_html=True)
    st.subheader("💰 Processing Estimate")
    
    cost_estimate = token_estimator.estimate_processing_cost(
        text_content, schema_content, selected_model
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Input Tokens", f"{cost_estimate['input_tokens']:,}")
    with col2:
        st.metric("Est. Total Tokens", f"{cost_estimate['total_estimated_tokens']:,}")
    with col3:
        st.metric("Est. Cost", f"${cost_estimate['estimated_cost_usd']}")
    with col4:
        st.metric("Est. Time", f"{cost_estimate['processing_time_estimate_minutes']:.1f}m")
    
    if processing_mode == "Enhanced Mode (Complex Documents)":
        st.info(f"📊 Document will be processed in ~{cost_estimate['estimated_chunks']} chunks")
    
    st.markdown('</div>', unsafe_allow_html=True)

# === Processing Button ===
if st.button("🚀 Extract JSON", type="primary", use_container_width=True):
    if not schema_file or not text_file:
        st.error("Please upload both JSON schema and input text.")
    elif 'schema_json' not in locals() or not schema_json:
        st.error("Please fix the JSON schema errors first.")
    else:
        session_id = session_manager.create_session()
        
        # Log session details
        logger_service.log(session_id, "config", json.dumps({
            'processing_mode': processing_mode,
            'model': selected_model,
            'text_length': len(text_content),
            'advanced_settings': {
                'chunk_size': chunk_size if processing_mode == "Enhanced Mode (Complex Documents)" else None,
                'overlap_size': overlap_size if processing_mode == "Enhanced Mode (Complex Documents)" else None,
                'enable_qa': enable_qa if processing_mode == "Enhanced Mode (Complex Documents)" else None,
                'confidence_threshold': confidence_threshold if processing_mode == "Enhanced Mode (Complex Documents)" else None
            }
        }))
        
        # Process based on selected mode
        if processing_mode == "Simple Mode (Original)":
            # Original simple processing
            with st.status("🔄 Processing with simple mode...", expanded=True):
                st.write("Validating schema...")
                is_valid, schema_err = schema_validator.is_valid_schema(schema_json)
                
                if not is_valid:
                    st.error(f"❌ Invalid schema: {schema_err}")
                else:
                    st.write("Processing with LLM...")
                    err = None
                    success = False
                    
                    for attempt in range(3):
                        st.write(f"Attempt {attempt + 1}/3...")
                        prompt = prompt_engine.create_prompt(schema_content, text_content, error=err)
                        logger_service.log(session_id, f"prompt_{attempt+1}", prompt)
                        
                        output = llm_interface.call_llm(prompt, model=selected_model)
                        logger_service.log(session_id, f"output_{attempt+1}", output)
                        
                        try:
                            result = json_extractor.extract_json(output)
                            valid, err = json_extractor.validate_against_schema(schema_json, result)
                            
                            if valid:
                                success = True
                                st.session_state["generated_result"] = result
                                st.session_state["session_id"] = session_id
                                st.session_state["processing_metadata"] = {"mode": "simple", "attempts": attempt + 1}
                                break
                        except Exception as e:
                            err = str(e)
                    
                    if success:
                        st.success("✅ JSON generated successfully!")
                    else:
                        st.error("❌ Failed to extract valid JSON after 3 attempts.")
        
        else:
            # Enhanced processing mode
            with st.status("🔄 Processing with enhanced pipeline...", expanded=True):
                st.write("Initializing enhanced pipeline...")
                orchestrator = PipelineOrchestrator()
                
                st.write("Analyzing document structure and creating semantic chunks...")
                try:
                    result_data = orchestrator.process_complex_extraction(
                        text=text_content,
                        schema=schema_json,
                        session_id=session_id,
                        model=selected_model
                    )
                    
                    result = result_data['result']
                    metadata = result_data['metadata']
                    
                    st.write(f"Processed {metadata['chunks_processed']} chunks")
                    st.write(f"Confidence score: {metadata['confidence_score']:.2f}")
                    
                    # Check if result meets quality threshold
                    if enable_qa and metadata['confidence_score'] < confidence_threshold:
                        st.warning(f"⚠️ Confidence score ({metadata['confidence_score']:.2f}) is below threshold ({confidence_threshold})")
                        
                        if st.button("Accept Result Anyway"):
                            st.session_state["generated_result"] = result
                            st.session_state["session_id"] = session_id
                            st.session_state["processing_metadata"] = metadata
                    else:
                        st.session_state["generated_result"] = result
                        st.session_state["session_id"] = session_id
                        st.session_state["processing_metadata"] = metadata
                        st.success("✅ Enhanced extraction completed successfully!")
                    
                    # Show quality issues if any
                    if metadata.get('validation_issues'):
                        with st.expander("⚠️ Quality Issues Detected"):
                            for issue in metadata['validation_issues']:
                                st.warning(issue)
                
                except Exception as e:
                    st.error(f"❌ Enhanced processing failed: {str(e)}")
                    st.info("You can try Simple Mode as a fallback.")

# === Results Display ===
if "generated_result" in st.session_state and st.session_state["generated_result"]:
    st.markdown("---")
    st.subheader("🎯 Extraction Results")
    
    # Show processing metadata
    if "processing_metadata" in st.session_state:
        metadata = st.session_state["processing_metadata"]
        
        if isinstance(metadata, dict) and "chunks_processed" in metadata:
            # Enhanced mode metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chunks Processed", metadata.get('chunks_processed', 'N/A'))
            with col2:
                st.metric("Confidence Score", f"{metadata.get('confidence_score', 0):.2f}")
            with col3:
                st.metric("Mode", "Enhanced")
        else:
            # Simple mode metadata
            st.info(f"Processed in {metadata.get('attempts', 'unknown')} attempts using Simple Mode")
    
    # Display the extracted JSON
    st.json(st.session_state["generated_result"])
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "📥 Download JSON Result",
            data=json.dumps(st.session_state["generated_result"], indent=2),
            file_name="extracted_data.json",
            mime="application/json",
            key="download-json",
            use_container_width=True
        )
    
    with col2:
        if "session_id" in st.session_state and st.session_state["session_id"]:
            try:
                sid = st.session_state["session_id"]
                log_files = [f for f in os.listdir(f"logs/{sid}") if f.endswith('.log') or f.endswith('.json')]
                
                if log_files:
                    log_data = ""
                    for log_file in log_files:
                        log_data += f"\n=== {log_file} ===\n"
                        with open(f"logs/{sid}/{log_file}", 'r', encoding='utf-8') as f:
                            log_data += f.read() + "\n"
                    
                    st.download_button(
                        "📄 Download Processing Logs",
                        data=log_data,
                        file_name=f"{sid}_complete_logs.txt",
                        mime="text/plain",
                        key="download-logs",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Could not prepare logs: {e}")

# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h4>🔧 Pipeline Features</h4>
    <p><strong>Enhanced Mode:</strong> Semantic chunking • Context awareness • Multi-pass extraction • Quality assurance</p>
    <p><strong>Simple Mode:</strong> Direct extraction • Retry logic • Basic validation</p>
</div>
""", unsafe_allow_html=True)