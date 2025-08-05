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
st.write("Advanced AI-powered extraction with chunking, multi-pass processing, and semantic analysis")

# Advanced Settings Sidebar
ADV = st.sidebar.expander("⚙️ Advanced Settings", expanded=False)
chunk_size = ADV.slider("Chunk token size", 1000, 8000, 3500)
overlap = ADV.slider("Chunk overlap", 0, 700, 200)
passes = ADV.slider("Max extraction passes per chunk", 1, 4, 2)

# Model Selection
model_map = {
    "GPT 4.1": "gpt-4.1-2025-04-14",
    "GPT O4 Mini": "o4-mini-2025-04-16"
}

selected_model = st.selectbox("Choose Model:", list(model_map.values()), 0)

# File Uploads
schema_file = st.file_uploader("Upload JSON Schema", type=["json"])
text_file = st.file_uploader("Upload Text File", type=["txt", "pdf"])

# Token Estimation (before processing)
if schema_file and text_file:
    schema_str = schema_file.read().decode()
    text_str = text_file.read().decode()
    
    # Reset file pointers for later use
    schema_file.seek(0)
    text_file.seek(0)
    
    # Estimate tokens
    total_tokens = token_estimator.estimate_tokens(schema_str + text_str, model=selected_model)
    text_cleaned = preprocessor.clean_ocr(text_str)
    chunks = semantic_chunker.split_semantic(text_cleaned, max_len=chunk_size, overlap=overlap)
    
    # Display token info
    st.info(f"""
    📊 **Processing Estimates:**
    - Total input tokens: **{total_tokens:,}**
    - Text will be split into: **{len(chunks)} chunks**
    - Max passes per chunk: **{passes}**
    - Expected API calls: **~{len(chunks) * passes}**
    """)

# Main Processing Button
if st.button("🚀 Generate JSON", type="primary"):
    if not schema_file or not text_file:
        st.error("Please upload both JSON schema and input text.")
    else:
        # Read files
        schema_str = schema_file.read().decode()
        text_str = text_file.read().decode()
        
        try:
            schema_json = json.loads(schema_str)
        except Exception as e:
            st.error(f"❌ Schema JSON is invalid: {e}")
            st.stop()

        # Validate schema
        is_valid, schema_err = schema_validator.is_valid_schema(schema_json)
        if not is_valid:
            st.error(f"❌ Invalid schema: {schema_err}")
            st.stop()

        # Initialize session and logging
        session_id = session_manager.create_session()
        logger_service.log(session_id, "model_used", selected_model)
        
        # Store session info in session state
        st.session_state['current_session_id'] = session_id
        
        # Preprocessing phase
        with st.status("🔧 Preprocessing text...", expanded=True) as status:
            st.write("Cleaning OCR artifacts and noise...")
            text_cleaned = preprocessor.clean_ocr(text_str)
            
            st.write("Performing semantic chunking...")
            chunks = semantic_chunker.split_semantic(text_cleaned, max_len=chunk_size, overlap=overlap)
            
            st.write(f"✅ Split into {len(chunks)} semantic chunks")
            status.update(label="✅ Preprocessing complete!", state="complete")

        # Extraction phase with progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def llm_fn(prompt): 
            return llm_interface.call_llm(prompt, model=selected_model)
        
        # Custom iterative extraction with progress tracking
        all_results = []
        total_chunks = len(chunks)
        
        with st.status("🧠 Extracting structured data...", expanded=True) as extraction_status:
            for i, chunk in enumerate(chunks):
                status_text.text(f"Processing chunk {i+1}/{total_chunks}...")
                progress_bar.progress((i) / total_chunks)
                
                st.write(f"**Chunk {i+1}/{total_chunks}**: {len(chunk.split())} words")
                
                partial = None
                errors = None
                
                # Multi-pass extraction for this chunk
                for attempt in range(passes):
                    try:
                        prompt = prompt_engine.create_advanced_prompt(
                            schema_str, chunk, 
                            error=errors, 
                            pass_num=attempt+1,
                            chunk_id=i+1
                        )
                        
                        st.write(f"  - Pass {attempt+1}...")
                        out = llm_fn(prompt)
                        
                        partial = json.loads(out) if isinstance(out, str) else out
                        valid, errors = json_extractor.validate_against_schema(schema_json, partial)
                        
                        if valid:
                            st.write(f"  ✅ Valid JSON extracted")
                            break
                        else:
                            st.write(f"  ⚠️ Validation failed: {errors[:100]}...")
                            
                    except Exception as e:
                        errors = str(e)
                        st.write(f"  ❌ Error: {str(e)[:100]}...")
                
                all_results.append(partial or {})
                progress_bar.progress((i + 1) / total_chunks)
            
            extraction_status.update(label="✅ Extraction complete!", state="complete")
        
        # Aggregation phase
        with st.status("🔗 Merging chunk results...", expanded=True) as merge_status:
            st.write("Deduplicating and merging partial results...")
            final = aggregator.merge_json_outputs(all_results, schema_json)
            
            # Log results
            logger_service.log(session_id, "raw_outputs", json.dumps(all_results, indent=2))
            
            # Save final output
            final_output_path = f"logs/{session_id}/final_output.json"
            with open(final_output_path, "w") as f:
                json.dump(final, f, indent=2)
            
            merge_status.update(label="✅ Aggregation complete!", state="complete")
        
        # Store results in session state for persistence
        st.session_state['generated_result'] = final
        st.session_state['processing_complete'] = True
        st.session_state['chunk_count'] = len(chunks)
        st.session_state['total_passes'] = len(chunks) * passes
        
        progress_bar.progress(1.0)
        status_text.text("🎉 Processing complete!")
        
        st.success(f"""
        ✅ **Extraction Successful!**
        - Processed {len(chunks)} chunks
        - Total extraction attempts: {len(all_results)}
        - Session ID: `{session_id}`
        """)

# Results Display Section (Persistent)
if st.session_state.get('processing_complete', False):
    st.markdown("---")
    st.markdown("### 🎯 **Extracted JSON Result**")
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunks Processed", st.session_state.get('chunk_count', 0))
    with col2:
        st.metric("Total Passes", st.session_state.get('total_passes', 0))
    with col3:
        st.metric("Fields Extracted", len(st.session_state.get('generated_result', {})))
    
    # Display JSON
    st.json(st.session_state['generated_result'])
    
    # Download Buttons Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "📥 Download JSON",
            data=json.dumps(st.session_state['generated_result'], indent=2),
            file_name="extracted_output.json",
            mime="application/json",
            help="Download the extracted JSON result"
        )
    
    with col2:
        # Prepare logs for download
        if 'current_session_id' in st.session_state:
            session_id = st.session_state['current_session_id']
            log_dir = f"logs/{session_id}"
            
            if os.path.exists(log_dir):
                log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
                all_logs = {}
                
                for log_file in log_files:
                    with open(f"{log_dir}/{log_file}", "r", encoding="utf-8") as f:
                        all_logs[log_file] = f.read()
                
                log_data = json.dumps(all_logs, indent=2)
                
                st.download_button(
                    "📄 Download Logs",
                    data=log_data,
                    file_name=f"session_{session_id}_logs.json",
                    mime="application/json",
                    help="Download complete session logs"
                )
            else:
                st.button("📄 Download Logs", disabled=True, help="No logs available")
    
    with col3:
        if st.button("🔄 Process New Document"):
            # Clear session state for new processing
            for key in ['generated_result', 'processing_complete', 'current_session_id', 'chunk_count', 'total_passes']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Sidebar Info
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📋 **Features**")
    st.markdown("""
    - ✅ **Semantic Chunking**: Smart text splitting
    - ✅ **Multi-pass Extraction**: Error correction
    - ✅ **Schema Validation**: Strict conformance  
    - ✅ **Progress Tracking**: Real-time status
    - ✅ **Persistent Results**: Download anytime
    - ✅ **Complete Logging**: Full audit trail
    """)
    
    if 'current_session_id' in st.session_state:
        st.markdown(f"**Current Session:** `{st.session_state['current_session_id'][:8]}...`")
