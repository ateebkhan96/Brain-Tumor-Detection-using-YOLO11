import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from ultralytics import YOLO
import cv2
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Brain Tumour Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .detection-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    """Load the YOLO model with caching for better performance."""
    try:
        model = YOLO(model_path)
        return model, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False

def get_model_info(model):
    """Extract model information for display."""
    if model is not None:
        try:
            # Get model metadata
            model_info = {
                "Model Type": "YOLOv11",
                "Task": model.task,
                "Model Size": f"{len(model.model.parameters())} parameters" if hasattr(model.model, 'parameters') else "N/A",
                "Input Shape": "640x640 (default)",
                "Classes": len(model.names) if hasattr(model, 'names') else "N/A"
            }
            return model_info
        except:
            return {"Model Type": "YOLOv11", "Status": "Loaded Successfully"}
    return {}

def analyze_detection_results(results):
    """Analyze detection results and return statistics."""
    if not results or len(results) == 0:
        return None
    
    result = results[0]
    boxes = result.boxes
    
    if boxes is None or len(boxes) == 0:
        return None
    
    # Extract detection data
    confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []
    classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
    
    # Get class names
    class_names = [result.names[int(cls)] for cls in classes] if len(classes) > 0 else []
    
    analysis = {
        "total_detections": len(boxes),
        "confidence_scores": confidences,
        "detected_classes": class_names,
        "class_counts": pd.Series(class_names).value_counts().to_dict() if class_names else {},
        "avg_confidence": np.mean(confidences) if len(confidences) > 0 else 0,
        "max_confidence": np.max(confidences) if len(confidences) > 0 else 0,
        "min_confidence": np.min(confidences) if len(confidences) > 0 else 0
    }
    
    return analysis

def create_confidence_chart(analysis):
    """Create a confidence score distribution chart."""
    if analysis and len(analysis["confidence_scores"]) > 0:
        fig = px.histogram(
            x=analysis["confidence_scores"],
            nbins=10,
            title="Detection Confidence Score Distribution",
            labels={'x': 'Confidence Score', 'y': 'Count'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(showlegend=False)
        return fig
    return None

def create_class_distribution_chart(analysis):
    """Create a class distribution pie chart."""
    if analysis and analysis["class_counts"]:
        fig = px.pie(
            values=list(analysis["class_counts"].values()),
            names=list(analysis["class_counts"].keys()),
            title="Detected Object Classes Distribution"
        )
        return fig
    return None

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Tumour Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load model with fixed path
        model_path = "train/weights/last.pt"
        model, model_loaded = load_model(model_path)
        
        if model_loaded:
            st.success("✅ Model loaded successfully!")
        else:
            st.error("❌ Failed to load model")
            return
        
        # Detection parameters with explanations
        st.subheader("🎛️ Detection Parameters")
        
        # Confidence threshold with info
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        with st.expander("ℹ️ What is Confidence Threshold?"):
            st.markdown("""
            **Confidence Threshold** determines the minimum confidence score required for a detection to be considered valid.
            
            - **Higher values (0.7-0.9)**: More strict detection, fewer false positives, but might miss some tumours
            - **Lower values (0.1-0.4)**: More sensitive detection, catches more potential tumours, but may include false positives
            - **Default (0.25)**: Balanced approach for general use
            """)
        
        # IoU threshold with info
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
        with st.expander("ℹ️ What is IoU Threshold?"):
            st.markdown("""
            **IoU (Intersection over Union) Threshold** controls how much overlap is allowed between detection boxes before they are merged.
            
            - **Higher values (0.6-0.8)**: Allows more overlapping detections, useful when tumours are close together
            - **Lower values (0.2-0.4)**: Removes more overlapping detections, cleaner results but might merge separate tumours
            - **Default (0.45)**: Good balance for most medical imaging scenarios
            """)
        

        
    # Main content
    # File uploader - always visible
    st.subheader("📁 Upload Medical Image")
    uploaded_file = st.file_uploader(
        "Choose a brain scan image for tumour detection", 
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload a brain MRI or CT scan image for automated tumour detection analysis"
    )
    
    if uploaded_file is not None:
        # Show both original and detection results side by side
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            # Display original image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📷 Original Image", use_container_width=True)
            
            # Image information
            st.markdown('<div class="detection-info">', unsafe_allow_html=True)
            st.write(f"**Image Info:**")
            st.write(f"- Filename: {uploaded_file.name}")
            st.write(f"- Size: {image.size[0]} x {image.size[1]} pixels")
            st.write(f"- Mode: {image.mode}")
            st.write(f"- Upload time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("🎯 Detection Results")
            
            # Run detection
            with st.spinner("🔄 Running brain tumour detection..."):
                start_time = time.time()
                
                # Convert to numpy array
                image_np = np.array(image)
                
                # Run inference with custom parameters
                results = model(
                    image_np,
                    conf=conf_threshold,
                    iou=iou_threshold
                )
                
                inference_time = time.time() - start_time
                
                # Get annotated image
                annotated_image = results[0].plot()
                
                # Display results
                st.image(annotated_image, caption="🎯 Detected Tumours", use_container_width=True)
                
                # Performance metrics
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("⏱️ Inference Time", f"{inference_time:.3f}s")
                with col_b:
                    fps = 1 / inference_time if inference_time > 0 else 0
                    st.metric("⚡ FPS", f"{fps:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis section - simplified for brain tumour detection
    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("📈 Detection Summary")
        
        # Analyze results
        analysis = analyze_detection_results(results)
        
        if analysis:
            # Simple summary metrics only
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Tumours Detected", analysis["total_detections"])
            with col2:
                st.metric("📊 Highest Confidence", f"{analysis['max_confidence']:.3f}")
            with col3:
                confidence_level = "High" if analysis['max_confidence'] > 0.7 else "Medium" if analysis['max_confidence'] > 0.4 else "Low"
                st.metric("🔍 Detection Quality", confidence_level)
        else:
            st.info("✅ No tumours detected in the brain scan.")
        
        # Export results - simplified
        st.subheader("💾 Export Results")
        if st.button("📥 Download Annotated Image"):
            # Convert annotated image to PIL
            annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            
            # Save to bytes
            import io
            img_buffer = io.BytesIO()
            annotated_pil.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            st.download_button(
                label="Download PNG",
                data=img_buffer,
                file_name=f"tumour_detection_{uploaded_file.name.split('.')[0]}.png",
                mime="image/png"
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit • Powered by Ultralytics YOLO</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()