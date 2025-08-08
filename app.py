import streamlit as st
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from core import schema_analyze
from core import (
    prompt_engine,
    llm_interface,
    json_extractor,
    logger_service,
    session_manager,
    token_estimator,
    text_analyzer,
    schema_analyzer,
    extraction_engine,
    merger_engine,
    embedding_manager
)

load_dotenv()

st.set_page_config(page_title="Advanced AI JSON Extractor", layout="wide")
st.title("🧠 Advanced Structured JSON Extractor v2.0")
st.markdown("**Intelligent Document Processing with Adaptive Chunking & Multi-Pass Extraction**")

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'pipeline_choice' not in st.session_state:
    st.session_state.pipeline_choice = None

def log_and_display(message, level="INFO"):
    """Log message and display in UI"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {level}: {message}"
    st.session_state.logs.append(log_entry)
    if st.session_state.session_id:
        logger_service.log(st.session_state.session_id, "process_log", log_entry)

# === Model Selection ===
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("🤖 Model Selection")
    model_map = {
        "GPT 4.1 (Default)": "gpt-4.1-2025-04-14",
        "GPT O4 Mini": "o4-mini-2025-04-16",
        "Claude Sonnet 4.0": "claude-sonnet-4-20250514",
        "Chat GPT O3": "o3-2025-04-16"
    }
    
    selected_model_label = st.selectbox(
        "Choose Model:",
        options=list(model_map.keys()),
        index=0,
        help="Select the LLM model for extraction"
    )
    st.markdown(
    f"""
**Model Notes**  
- 🧠 **GPT 4.1 (Default)**: Recommended for well-structured schemas with fewer fields  
- ⚡ **GPT O4 Mini**: Best for very large text files, complex nesting schemas, and long output (up to 100k tokens)  
- 🎨 **Claude Sonnet 4.0**: Excels in creative reasoning(Credits Not avilable currently) 
- 🔍 **Chat GPT O3**: Optimized for step-by-step reasoning and detailed problem solving  
"""
)
    selected_model = model_map[selected_model_label]

with col2:
    st.subheader("📁 File Upload")
    schema_file = st.file_uploader("Upload JSON Schema", type=["json"], key="schema")
    text_file = st.file_uploader("Upload Text File", type=["txt", "pdf"], key="text")

# === File Analysis Section ===
if schema_file and text_file:
    st.markdown("---")
    st.subheader("📊 File Analysis")
    
    with st.spinner("Analyzing files..."):
        # Read files
        schema_str = schema_file.read().decode('utf-8')
        text_str = text_file.read().decode('utf-8')
        
        # Reset file pointers for potential re-reading
        schema_file.seek(0)
        text_file.seek(0)
        
        try:
            schema_json = json.loads(schema_str)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON schema: {e}")
            st.stop()
        
        # Token estimation
        total_tokens = token_estimator.estimate_tokens(schema_str + text_str, selected_model)
        text_tokens = token_estimator.estimate_tokens(text_str, selected_model)
        schema_tokens = token_estimator.estimate_tokens(schema_str, selected_model)
        
        # Schema analysis
        nesting_level = schema_analyzer.calculate_nesting_level(schema_json)
        field_count = schema_analyzer.count_fields(schema_json)
        required_fields = schema_analyzer.get_required_fields(schema_json)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tokens", f"{total_tokens:,}")
            st.metric("Text Tokens", f"{text_tokens:,}")
        
        with col2:
            st.metric("Schema Tokens", f"{schema_tokens:,}")
            st.metric("Nesting Level", nesting_level)
        
        with col3:
            st.metric("Total Fields", field_count)
            st.metric("Required Fields", len(required_fields))
        
        with col4:
            complexity_score = schema_analyzer.calculate_complexity_score(
                schema_tokens, nesting_level, field_count
            )
            st.metric("Complexity Score", f"{complexity_score:.2f}")
        
        # === Pipeline Selection ===
        st.markdown("---")
        st.subheader("🔄 Pipeline Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Simple Pipeline** ✨
            - Direct extraction
            - Single-pass processing
            - Best for: Small documents (<10k tokens)
            - Faster processing
            """)
            if st.button("Use Simple Pipeline", type="secondary", use_container_width=True):
                st.session_state.pipeline_choice = "simple"
        
        with col2:
            st.success("""
            **Extensive Pipeline** 🚀
            - Intelligent chunking
            - Schema partitioning
            - Multi-pass extraction
            - Best for: Large documents (>10k tokens)
            """)
            if st.button("Use Extensive Pipeline", type="primary", use_container_width=True):
                st.session_state.pipeline_choice = "extensive"

# === Process Execution ===
if st.session_state.pipeline_choice:
    st.markdown("---")
    st.subheader(f"⚙️ Processing with {st.session_state.pipeline_choice.title()} Pipeline")
    
    # Create session
    if not st.session_state.session_id:
        st.session_state.session_id = session_manager.create_session()
        log_and_display(f"Session created: {st.session_state.session_id}")
    
    # Log container for real-time updates
    log_container = st.container()
    progress_bar = st.progress(0)
    
    if st.session_state.pipeline_choice == "simple":
        # === SIMPLE PIPELINE ===
        with st.spinner("Processing with Simple Pipeline..."):
            log_and_display("Starting Simple Pipeline")
            progress_bar.progress(10)
            
            # Validate schema
            is_valid, error = schema_analyze.is_valid_schema(schema_json)
            if not is_valid:
                st.error(f"Schema validation failed: {error}")
                st.stop()
            
            log_and_display("Schema validated successfully")
            progress_bar.progress(30)
            
            # Create prompt and extract
            success = False
            extracted_json = None
            
            for attempt in range(3):
                log_and_display(f"Extraction attempt {attempt + 1}/3")
                progress_bar.progress(40 + (attempt * 20))
                
                prompt = prompt_engine.create_prompt(
                    schema_str, text_str, 
                    error=error if attempt > 0 else None
                )
                
                try:
                    output = llm_interface.call_llm(prompt, model=selected_model)
                    result = json_extractor.extract_json(output)
                    log_and_display(result)
                    valid, error = json_extractor.validate_against_schema(schema_json, result)
                    
                    if valid:
                        success = True
                        extracted_json = result
                        log_and_display("✅ Extraction successful!", "SUCCESS")
                        break
                    else:
                        log_and_display(f"Validation failed: {error}", "WARNING")
                except Exception as e:
                    error = str(e)
                    log_and_display(f"Extraction error: {error}", "ERROR")
            
            progress_bar.progress(100)
            
            if success:
                st.success("✅ JSON extracted successfully!")
                st.json(extracted_json)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download JSON",
                        data=json.dumps(extracted_json, indent=2),
                        file_name=f"{st.session_state.session_id}_output.json",
                        mime="application/json"
                    )
                with col2:
                    log_data = "\n".join(st.session_state.logs)
                    st.download_button(
                        "📄 Download Logs",
                        data=log_data,
                        file_name=f"{st.session_state.session_id}_log.txt",
                        mime="text/plain"
                    )
            else:
                st.error("❌ Failed to extract valid JSON after 3 attempts")
                log_data = "\n".join(st.session_state.logs)
                st.download_button(
                                        "📄 Download Logs",
                                        data=log_data,
                                        file_name=f"{st.session_state.session_id}_log.txt",
                                        mime="text/plain")
    
    else:
        # === EXTENSIVE PIPELINE ===
        with st.spinner("Processing with Extensive Pipeline..."):
            log_and_display("Starting Extensive Pipeline")
            progress_bar.progress(5)
            
            # Step 1: Text Analysis and Chunking
            st.subheader("📝 Step 1: Text Document Analysis")
            log_and_display("Analyzing text document structure...")
            
            # Generate embeddings
            text_embeddings = embedding_manager.create_embeddings(text_str)
            st.session_state.text_embeddings = text_embeddings
            log_and_display(f"Generated text embeddings: {len(text_embeddings)} dimensions")
            progress_bar.progress(10)
            
            # Analyze text structure
            semantic_structure = text_analyzer.analyze_semantic_structure(text_str, selected_model)
            log_and_display("Semantic structure analysis complete")
            
            # Generate chunking strategy
            chunking_code = text_analyzer.generate_chunking_strategy(
                text_str, semantic_structure, selected_model
            )
            
            # Display chunking strategy
            with st.expander("📋 Proposed Chunking Strategy", expanded=True):
                st.code(chunking_code, language="python")
                
                chunk_metrics = text_analyzer.get_chunk_metrics(text_str, chunking_code)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Estimated Chunks", chunk_metrics.get('chunk_count', 'N/A'))
                with col2:
                    st.metric("Avg Chunk Size", f"{chunk_metrics.get('avg_size', 0):.0f} chars")
                with col3:
                    st.metric("Overlap", f"{chunk_metrics.get('overlap', 0)}%")
            
            # Chunking approval
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Approve Chunking", type="primary"):
                    st.session_state.chunking_approved = True
            with col2:
                if st.button("🔄 Regenerate", type="secondary"):
                    st.session_state.chunking_approved = False
                    st.rerun()
            with col3:
                if st.button("⏭️ Skip Chunking"):
                    st.session_state.chunking_approved = "skip"
            
            if hasattr(st.session_state, 'chunking_approved'):
                if st.session_state.chunking_approved == True:
                    # Execute chunking
                    chunks = text_analyzer.execute_chunking(text_str, chunking_code)
                    st.session_state.text_chunks = chunks
                    log_and_display(f"Text chunked into {len(chunks)} pieces")
                    progress_bar.progress(25)
                elif st.session_state.chunking_approved == "skip":
                    st.session_state.text_chunks = [text_str]
                    log_and_display("Skipping chunking, processing as single document")
                    progress_bar.progress(25)
                else:
                    st.stop()
            else:
                st.stop()
            
            # Step 2: Schema Analysis and Partitioning
            st.subheader("🔧 Step 2: JSON Schema Analysis")
            log_and_display("Analyzing JSON schema complexity...")
            
            # Validate schema
            is_valid, error = schema_analyze.is_valid_schema(schema_json)
            if not is_valid:
                st.error(f"Schema validation failed: {error}")
                st.stop()
            
            # Generate schema embeddings
            schema_embeddings = embedding_manager.create_embeddings(json.dumps(schema_json))
            st.session_state.schema_embeddings = schema_embeddings
            log_and_display(f"Generated schema embeddings: {len(schema_embeddings)} dimensions")
            progress_bar.progress(30)
            
            # Analyze schema complexity
            complexity_analysis = schema_analyzer.analyze_complexity(schema_json, selected_model)
            
            # Generate partitioning strategy
            partitioning_code = schema_analyzer.generate_partitioning_strategy(
                schema_json, complexity_analysis, selected_model
            )
            
            # Display partitioning strategy
            with st.expander("🔀 Proposed Schema Partitioning", expanded=True):
                st.code(partitioning_code, language="python")
                
                partition_metrics = schema_analyzer.get_partition_metrics(schema_json, partitioning_code)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Partitions", partition_metrics.get('partition_count', 'N/A'))
                with col2:
                    st.metric("Max Depth", partition_metrics.get('max_depth', 'N/A'))
                with col3:
                    st.metric("Fields/Partition", partition_metrics.get('avg_fields', 'N/A'))
            
            # Partitioning approval
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Approve Partitioning", type="primary", key="partition_approve"):
                    st.session_state.partitioning_approved = True
            with col2:
                if st.button("🔄 Regenerate", type="secondary", key="partition_regen"):
                    st.session_state.partitioning_approved = False
                    st.rerun()
            with col3:
                if st.button("⏭️ Skip Partitioning", key="partition_skip"):
                    st.session_state.partitioning_approved = "skip"
            
            if hasattr(st.session_state, 'partitioning_approved'):
                if st.session_state.partitioning_approved == True:
                    # Execute partitioning
                    partitions = schema_analyzer.execute_partitioning(schema_json, partitioning_code)
                    st.session_state.schema_partitions = partitions
                    log_and_display(f"Schema partitioned into {len(partitions)} parts")
                    progress_bar.progress(40)
                elif st.session_state.partitioning_approved == "skip":
                    st.session_state.schema_partitions = [schema_json]
                    log_and_display("Skipping partitioning, processing as single schema")
                    progress_bar.progress(40)
                else:
                    st.stop()
            else:
                st.stop()
            
            # Step 3: Multi-Pass Extraction
            st.subheader("🎯 Step 3: Multi-Pass Extraction")
            
            # Check optimization status
            single_chunk = len(st.session_state.text_chunks) == 1
            single_partition = len(st.session_state.schema_partitions) == 1
            is_optimized = single_chunk and single_partition
            
            if is_optimized:
                st.info("🚀 **Optimized Mode**: Single chunk and partition detected - merge step will be skipped")
            else:
                st.info(f"📊 Processing {len(st.session_state.text_chunks)} chunks × {len(st.session_state.schema_partitions)} partitions = {len(st.session_state.text_chunks) * len(st.session_state.schema_partitions)} operations")
            
            # Get number of passes
            num_passes = st.slider("Number of extraction passes:", 2, 5, 3)
            
            if st.button("🚀 Start Extraction", type="primary"):
                log_and_display(f"Starting {num_passes}-pass extraction")
                progress_bar.progress(45)
                
                # Keep embeddings in context
                context = {
                    'text_embeddings': st.session_state.text_embeddings,
                    'schema_embeddings': st.session_state.schema_embeddings,
                    'total_chunks': len(st.session_state.text_chunks),
                    'total_partitions': len(st.session_state.schema_partitions)
                }
                
                # Check if we can skip merging (single chunk and single partition)
                single_chunk = len(st.session_state.text_chunks) == 1
                single_partition = len(st.session_state.schema_partitions) == 1
                skip_merging = single_chunk and single_partition
                
                if skip_merging:
                    log_and_display("Single chunk and partition detected - optimizing pipeline", "INFO")
                
                # Extract for each chunk-partition combination
                all_extractions = []
                total_operations = len(st.session_state.text_chunks) * len(st.session_state.schema_partitions)
                current_op = 0
                
                for chunk_idx, chunk in enumerate(st.session_state.text_chunks):
                    for partition_idx, partition in enumerate(st.session_state.schema_partitions):
                        current_op += 1
                        progress = 45 + int((current_op / total_operations) * 40)
                        progress_bar.progress(progress)
                        
                        log_and_display(f"Processing chunk {chunk_idx+1}/{len(st.session_state.text_chunks)} "
                                      f"with partition {partition_idx+1}/{len(st.session_state.schema_partitions)}")
                        
                        # Multi-pass extraction
                        extraction = extraction_engine.multi_pass_extract(
                            chunk, partition, num_passes, selected_model, context
                        )
                        
                        if extraction:
                            all_extractions.append({
                                'chunk_idx': chunk_idx,
                                'partition_idx': partition_idx,
                                'data': extraction
                            })
                            log_and_display(f"✅ Extraction successful for chunk {chunk_idx+1}, partition {partition_idx+1}")
                        else:
                            log_and_display(f"⚠️ No data extracted for chunk {chunk_idx+1}, partition {partition_idx+1}", "WARNING")
                
                progress_bar.progress(85)
                
                # Skip merging if single chunk and partition
                if skip_merging and all_extractions:
                    log_and_display("Skipping merge operation - using direct extraction result")
                    final_json = all_extractions[0]['data']
                    log_and_display(final_json)
                    
                    # Ensure schema compliance even for single extraction
                    # final_json = merger_engine.enforce_schema_compliance(final_json, schema_json)
                    progress_bar.progress(95)
                else:
                    # Step 4: Intelligent Merging (only if needed)
                    st.subheader("🔄 Step 4: Intelligent Merging")
                    log_and_display("Merging extracted data...")
                    
                    # Merge all extractions
                    final_json = merger_engine.intelligent_merge(
                        all_extractions, 
                        schema_json, 
                        context,
                        selected_model
                    )
                    log_and_display(final_json)
                    progress_bar.progress(95)
                
                # Final validation
                log_and_display("Performing final validation...")
                is_valid, error = json_extractor.validate_against_schema(schema_json, final_json)
                
                if is_valid:
                    progress_bar.progress(100)
                    log_and_display("✅ Final validation successful!", "SUCCESS")
                    
                    st.success("✅ Extraction completed successfully!")
                    st.json(final_json)
                    
                    # Save output
                    output_path = f"logs/{st.session_state.session_id}/{st.session_state.session_id}_output.json"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'w') as f:
                        json.dump(final_json, f, indent=2)
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download JSON",
                            data=json.dumps(final_json, indent=2),
                            file_name=f"{st.session_state.session_id}_output.json",
                            mime="application/json"
                        )
                    with col2:
                        log_data = "\n".join(st.session_state.logs)
                        st.download_button(
                            "📄 Download Logs",
                            data=log_data,
                            file_name=f"{st.session_state.session_id}_log.txt",
                            mime="text/plain"
                        )
                else:
                    log_and_display(f"❌ Final validation failed: {error}", "ERROR")
                    
                    # Attempt to fix
                    with st.spinner("Attempting to fix validation errors..."):
                        fixed_json = extraction_engine.fix_validation_errors(
                            final_json, schema_json, error, context, selected_model
                        )
                        
                        is_valid, error = json_extractor.validate_against_schema(schema_json, fixed_json)
                        if is_valid:
                            progress_bar.progress(100)
                            log_and_display("✅ Errors fixed successfully!", "SUCCESS")
                            st.success("✅ Extraction completed with fixes!")
                            st.json(fixed_json)
                            
                            # Download buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    "📥 Download JSON",
                                    data=json.dumps(fixed_json, indent=2),
                                    file_name=f"{st.session_state.session_id}_output.json",
                                    mime="application/json"
                                )
                            with col2:
                                log_data = "\n".join(st.session_state.logs)
                                st.download_button(
                                    "📄 Download Logs",
                                    data=log_data,
                                    file_name=f"{st.session_state.session_id}_log.txt",
                                    mime="text/plain"
                                )
                        else:
                            st.error(f"❌ Unable to fix validation errors: {error}")
                            log_data = "\n".join(st.session_state.logs)
                            st.download_button(
                                        "📄 Download Logs",
                                        data=log_data,
                                        file_name=f"{st.session_state.session_id}_log.txt",
                                        mime="text/plain")

# === Log Display ===
if st.session_state.logs:
    with st.expander("📜 Process Logs", expanded=False):
        for log in st.session_state.logs:
            st.text(log)