import streamlit as st
import os
import json
import time
from datetime import datetime

# Import our modules
from config import Config
from models import ModelInterface, ModelSelector
from text_processor import TextProcessor
from json_extractor import JSONExtractor
from utils import FileManager, LogManager, SchemaValidator, UIHelper

# Page configuration
st.set_page_config(
    page_title="Advanced Text-to-JSON Extractor",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processing_logs' not in st.session_state:
    st.session_state.processing_logs = []
if 'extraction_result' not in st.session_state:
    st.session_state.extraction_result = None
if 'usage_stats' not in st.session_state:
    st.session_state.usage_stats = {}

def initialize_app():
    """Initialize application components"""
    config = Config()
    
    # Create necessary directories
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    
    return config

def main():
    config = initialize_app()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-success { color: #28a745; }
    .status-error { color: #dc3545; }
    .status-warning { color: #ffc107; }
    .metrics-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🔄 Advanced Text-to-JSON Extractor</h1>
        <p>Intelligent document processing with hierarchical extraction, parallel processing, and multi-model support</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for API Key configuration
    with st.sidebar:
        st.markdown("---")
        
        # API Key configuration
        st.subheader("🔑 API Configuration")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=config.OPENAI_API_KEY or "",
            help="Your OpenAI API key for model access"
        )
        
        if not api_key:
            st.error("Please provide your OpenAI API key to continue")
            st.stop()
    
    # Main application
    run_main_application(config, api_key)

def run_main_application(config: Config, api_key: str):
    """Run the main application logic"""
    
    # Initialize components
    model_interface = ModelInterface(api_key)
    text_processor = TextProcessor()
    json_extractor = JSONExtractor(model_interface)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔄 Extract", "📊 Analytics", "📋 Logs", "ℹ️ Help"])
    
    with tab1:
        run_extraction_interface(config, model_interface, text_processor, json_extractor)
    
    with tab2:
        run_analytics_interface()
    
    with tab3:
        run_logs_interface()
    
    with tab4:
        run_help_interface()

def run_extraction_interface(config, model_interface, text_processor, json_extractor):
    """Main extraction interface"""
    
    # Model selection
    selected_model = ModelSelector.render_model_selection()
    
    st.markdown("---")
    
    # File upload section
    st.subheader("📁 File Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**JSON Schema File**")
        schema_file = st.file_uploader(
            "Upload JSON schema file",
            type=['json'],
            key="schema_upload",
            help="Upload a JSON schema file that defines the structure for extraction"
        )
        
        if schema_file:
            try:
                schema_content = json.loads(schema_file.read().decode('utf-8'))
                is_valid, validation_msg = SchemaValidator.validate_json_schema(schema_content)
                
                if is_valid:
                    st.success(f"✅ Valid schema: {validation_msg}")
                    st.json(schema_content)
                    
                    # Analyze schema complexity
                    complexity = SchemaValidator.get_schema_complexity(schema_content)
                    st.info(f"Schema complexity: **{complexity.upper()}**")
                    
                    # Model recommendation
                    recommended_model = model_interface.get_model_recommendation(0, complexity)
                    if recommended_model != selected_model:
                        st.warning(f"💡 Recommended model for this schema: **{config.MODELS[recommended_model]['name']}**")
                
                else:
                    st.error(f"❌ Invalid schema: {validation_msg}")
                    schema_content = None
            
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON file: {str(e)}")
                schema_content = None
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                schema_content = None
        else:
            schema_content = None
    
    with col2:
        st.write("**Text File to Process**")
        text_file = st.file_uploader(
            "Upload text file for extraction",
            type=['txt', 'md'],
            key="text_upload",
            help="Upload a text file from which to extract structured data"
        )
        
        if text_file:
            # Validate file
            is_valid, validation_msg = FileManager.validate_uploaded_file(
                text_file, config.ALLOWED_TEXT_EXTENSIONS, config.MAX_FILE_SIZE
            )
            
            if is_valid:
                try:
                    text_content = text_file.read().decode('utf-8')
                    st.success(f"✅ File loaded: {len(text_content):,} characters")
                    
                    # Show text preview
                    with st.expander("📄 Text Preview"):
                        st.text_area("Content preview:", text_content[:1000] + "..." if len(text_content) > 1000 else text_content, height=200)
                
                except Exception as e:
                    st.error(f"❌ Error reading text file: {str(e)}")
                    text_content = None
            else:
                st.error(f"❌ {validation_msg}")
                text_content = None
        else:
            text_content = None
    
    # Processing options
    if schema_content and text_content:
        st.markdown("---")
        st.subheader("⚙️ Processing Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            processing_mode = st.selectbox(
                "Processing Mode",
                options=["chunked_parallel", "chunked_sequential", "iterative", "single_pass"],
                format_func=lambda x: {
                    "chunked_parallel": "Chunked Parallel",
                    "chunked_sequential": "Chunked Sequential", 
                    "iterative": "Iterative Refinement",
                    "single_pass": "Single Pass"
                }[x],
                help="Choose the processing strategy"
            )
        
        with col2:
            max_workers = st.slider(
                "Parallel Workers",
                min_value=1,
                max_value=5,
                value=3,
                disabled=processing_mode != "chunked_parallel",
                help="Number of parallel workers for chunk processing"
            )
        
        with col3:
            chunk_size = st.slider(
                "Chunk Size (words)",
                min_value=500,
                max_value=5000,
                value=config.CHUNK_SIZE,
                step=500,
                help="Size of text chunks for processing"
            )
        
        # Start extraction
        if st.button("🚀 Start Extraction", type="primary", use_container_width=True):
            run_extraction_process(
                schema_content, text_content, selected_model, processing_mode,
                max_workers, chunk_size, config, model_interface, text_processor, json_extractor
            )
    
    # Display results
    if st.session_state.extraction_result:
        display_extraction_results()

def run_extraction_process(schema_content, text_content, selected_model, processing_mode, 
                         max_workers, chunk_size, config, model_interface, text_processor, json_extractor):
    """Run the extraction process"""
    
    # Clear previous logs and results
    json_extractor.clear_logs()
    st.session_state.processing_logs = []
    st.session_state.extraction_result = None
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        st.subheader("🔄 Processing Status")
        
        # Overall progress bar
        overall_progress = st.progress(0)
        status_placeholder = st.empty()
        
        try:
            # Step 1: Text preprocessing
            overall_progress.progress(0.1)
            status_placeholder.info("🔄 Preprocessing text...")
            
            # Preprocess text
            processed_text = text_processor.preprocess_text(text_content)
            json_extractor.log_step("Text Preprocessing", "success", f"Processed {len(processed_text):,} characters")
            
            # Detect structure
            structure = text_processor.detect_structure(processed_text)
            json_extractor.log_step("Structure Detection", "success", 
                                  f"Headers: {structure['has_headers']}, Lists: {structure['has_lists']}, Paragraphs: {structure['paragraph_count']}")
            
            overall_progress.progress(0.2)
            
            # Step 2: Chunking
            if processing_mode in ["chunked_parallel", "chunked_sequential"]:
                status_placeholder.info("🔄 Chunking text...")
                
                # Update chunk size in config
                config.CHUNK_SIZE = chunk_size
                text_processor.config.CHUNK_SIZE = chunk_size
                
                # Perform chunking
                if structure['has_headers']:
                    chunks = text_processor.chunk_by_structure(processed_text, structure)
                else:
                    chunks = text_processor.semantic_chunk(processed_text, chunk_size)
                
                json_extractor.log_step("Text Chunking", "success", f"Created {len(chunks)} chunks")
                overall_progress.progress(0.3)
                
                # Step 3: Extraction
                status_placeholder.info("🔄 Extracting data from chunks...")
                
                if processing_mode == "chunked_parallel":
                    results = json_extractor.parallel_extraction(
                        schema_content, chunks, selected_model, max_workers
                    )
                else:
                    results = []
                    for i, chunk in enumerate(chunks):
                        result = json_extractor.extract_from_single_chunk(
                            schema_content, chunk, selected_model
                        )
                        results.append(result)
                        
                        progress = 0.3 + (i + 1) / len(chunks) * 0.5
                        overall_progress.progress(progress)
                
                overall_progress.progress(0.8)
                
                # Step 4: Merge results
                status_placeholder.info("🔄 Merging extraction results...")
                
                final_result = json_extractor.merge_extraction_results(results, schema_content)
                
            elif processing_mode == "iterative":
                status_placeholder.info("🔄 Performing iterative extraction...")
                
                final_result = json_extractor.iterative_extraction(
                    schema_content, processed_text, selected_model, max_iterations=3
                )
                
                overall_progress.progress(0.8)
                
            else:  # single_pass
                status_placeholder.info("🔄 Performing single-pass extraction...")
                
                result = json_extractor.extract_from_single_chunk(
                    schema_content, {'chunk_id': 0, 'text': processed_text}, selected_model
                )
                
                if result['success']:
                    final_result = {
                        'success': True,
                        'data': result['data'],
                        'total_usage': result['usage']
                    }
                else:
                    final_result = {
                        'success': False,
                        'error': result['error']
                    }
                
                overall_progress.progress(0.8)
            
            overall_progress.progress(1.0)
            
            # Store results
            st.session_state.extraction_result = final_result
            st.session_state.processing_logs = json_extractor.get_processing_logs()
            
            if 'total_usage' in final_result:
                st.session_state.usage_stats = final_result['total_usage']
            
            if final_result['success']:
                status_placeholder.success("✅ Extraction completed successfully!")
            else:
                status_placeholder.error(f"❌ Extraction failed: {final_result.get('error', 'Unknown error')}")
        
        except Exception as e:
            overall_progress.progress(1.0)
            status_placeholder.error(f"❌ Processing error: {str(e)}")
            json_extractor.log_step("Processing Error", "error", str(e))
            
            st.session_state.processing_logs = json_extractor.get_processing_logs()

def display_extraction_results():
    """Display extraction results"""
    st.markdown("---")
    st.subheader("📊 Extraction Results")
    
    result = st.session_state.extraction_result
    
    if result['success']:
        # Usage metrics
        if 'total_usage' in result:
            st.write("### 💰 Usage Metrics")
            UIHelper.render_usage_metrics(result['total_usage'])
        
        # Extracted JSON
        st.write("### 📄 Extracted JSON")
        st.json(result['data'])
        
        # Download buttons
        st.write("### 📥 Download Results")
        UIHelper.create_download_buttons(
            result['data'], 
            st.session_state.processing_logs,
            "extraction"
        )
        
        # Merge summary (if applicable)
        if 'merge_summary' in result:
            st.write("### 📈 Processing Summary")
            summary = result['merge_summary']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Chunks", summary['total_chunks'])
            with col2:
                st.metric("Successful", summary['successful_chunks'])
            with col3:
                st.metric("Failed", summary['failed_chunks'])
    
    else:
        st.error(f"❌ Extraction failed: {result.get('error', 'Unknown error')}")

def run_analytics_interface():
    """Analytics and metrics interface"""
    st.subheader("📊 Processing Analytics")
    
    if not st.session_state.usage_stats:
        st.info("No usage data available. Run an extraction first.")
        return
    
    # Usage overview
    st.write("### Token Usage Overview")
    UIHelper.render_usage_metrics(st.session_state.usage_stats)
    
    # Cost breakdown
    if st.session_state.usage_stats.get('total_cost', 0) > 0:
        st.write("### Cost Analysis")
        
        usage = st.session_state.usage_stats
        input_cost = (usage.get('total_input_tokens', 0) / 1000) * 0.03  # Approximate
        output_cost = (usage.get('total_output_tokens', 0) / 1000) * 0.06  # Approximate
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Input Cost", f"${input_cost:.4f}")
        with col2:
            st.metric("Output Cost", f"${output_cost:.4f}")

def run_logs_interface():
    """Logs viewing interface"""
    st.subheader("📋 Processing Logs")
    
    if not st.session_state.processing_logs:
        st.info("No logs available. Run an extraction first.")
        return
    
    # Convert logs to DataFrame for better display
    logs_df = LogManager.create_log_dataframe(st.session_state.processing_logs)
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        status_filter = st.multiselect(
            "Filter by Status",
            options=logs_df['Status'].unique(),
            default=logs_df['Status'].unique()
        )
    
    with col2:
        step_filter = st.multiselect(
            "Filter by Step",
            options=logs_df['Step'].unique(),
            default=logs_df['Step'].unique()
        )
    
    # Apply filters
    filtered_logs = logs_df[
        (logs_df['Status'].isin(status_filter)) &
        (logs_df['Step'].isin(step_filter))
    ]
    
    # Display filtered logs
    st.dataframe(filtered_logs, use_container_width=True)
    
    # Export logs
    if st.button("💾 Export Logs"):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logs_json = json.dumps(st.session_state.processing_logs, indent=2)
        
        st.download_button(
            label="📥 Download Logs JSON",
            data=logs_json,
            file_name=f"processing_logs_{timestamp}.json",
            mime="application/json"
        )

def run_help_interface():
    """Help and documentation interface"""
    st.subheader("ℹ️ Help & Documentation")
    
    # Model selection guidance
    with st.expander("🤖 Model Selection Guide"):
        st.write("""
        **GPT-4.1** - Best for:
        - Complex, deeply nested JSON schemas
        - High-accuracy requirements
        - Programming and technical content
        - When cost is not the primary concern
        
        **GPT-4o Mini** - Best for:
        - Simple to moderate schemas
        - High-volume processing
        - Budget-conscious projects
        - Quick turnaround requirements
        
        **GPT-O3** - Best for:
        - Balanced performance needs
        - General-purpose extraction
        - Moderate complexity schemas
        - Good cost/performance ratio
        """)
    
    # Processing modes
    with st.expander("⚙️ Processing Modes Explained"):
        st.write("""
        **Chunked Parallel**: 
        - Splits text into chunks and processes them simultaneously
        - Fastest for large documents
        - Best for documents with clear structure
        
        **Chunked Sequential**:
        - Processes chunks one at a time
        - More reliable for rate-limited APIs
        - Better error handling and recovery
        
        **Iterative Refinement**:
        - Multiple passes with progressive improvement
        - Best accuracy for complex extractions
        - Slower but more thorough
        
        **Single Pass**:
        - Processes entire document at once
        - Fastest for small documents
        - Limited by model context window
        """)
    
    # Schema guidelines
    with st.expander("📝 JSON Schema Guidelines"):
        st.write("""
        **Required Elements**:
        - `type`: Must be "object" for main schema
        - `properties`: Define each field to extract
        
        **Field Types**:
        - `string`: Text data
        - `number`/`integer`: Numeric data
        - `boolean`: True/false values
        - `array`: Lists of items
        - `object`: Nested structures
        
        **Best Practices**:
        - Use descriptive field names
        - Include examples in descriptions
        - Mark required vs optional fields
        - Keep nested structures reasonable
        """)
    
    # Troubleshooting
    with st.expander("🔧 Troubleshooting"):
        st.write("""
        **Common Issues**:
        
        1. **"Invalid JSON Schema"**
           - Ensure your schema has proper structure
           - Check for syntax errors in JSON
           - Verify all required properties are present
        
        2. **"Extraction Failed"**
           - Try a simpler processing mode first
           - Reduce chunk size for large documents
           - Check if text contains extractable information
        
        3. **"API Rate Limit"**
           - Switch to sequential processing
           - Reduce number of parallel workers
           - Add delays between requests
        
        4. **"High Costs"**
           - Use GPT-4o Mini for simple extractions
           - Optimize chunk sizes
           - Use single-pass for small documents
        """)

if __name__ == "__main__":
    main()
