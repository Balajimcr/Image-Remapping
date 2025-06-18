import gradio as gr
import random
from barrel_distortion_processing import (
    process_distortion,
    export_grid_csv,
    export_gdc_grid_csv,
    test_correction_accuracy,
    validate_correction_quality
)

def create_gradio_interface():
    """
    Create the Gradio interface for the barrel distortion simulator
    """
    
    with gr.Blocks(title="Barrel Distortion Simulator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🔍 Barrel Distortion Simulator & Corrector
        
        This tool simulates barrel distortion effects commonly found in camera lenses and provides correction capabilities.
        Adjust the parameters below to see how different distortion coefficients affect the image and grid mapping.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📐 Image & Grid Parameters")
                
                image_width = gr.Slider(
                    minimum=640, maximum=2560, step=64, value=1280,
                    label="Image Width", info="Width of the image in pixels"
                )
                
                image_height = gr.Slider(
                    minimum=480, maximum=1920, step=48, value=720,
                    label="Image Height", info="Height of the image in pixels"
                )
                
                grid_rows = gr.Slider(
                    minimum=3, maximum=21, step=2, value=7,
                    label="Grid Rows", info="Number of grid rows (odd numbers recommended)"
                )
                
                grid_cols = gr.Slider(
                    minimum=3, maximum=21, step=2, value=9,
                    label="Grid Columns", info="Number of grid columns (odd numbers recommended)"
                )
                
                gr.Markdown("### 🌀 Distortion Parameters")
                
                k1 = gr.Slider(
                    minimum=-0.5, maximum=0.5, step=0.01, value=-0.2,
                    label="K1 (Radial)", info="Primary radial distortion coefficient"
                )
                
                k2 = gr.Slider(
                    minimum=-0.2, maximum=0.2, step=0.01, value=0.05,
                    label="K2 (Radial)", info="Secondary radial distortion coefficient"
                )
                
                k3 = gr.Slider(
                    minimum=-0.1, maximum=0.1, step=0.005, value=0.0,
                    label="K3 (Radial)", info="Tertiary radial distortion coefficient"
                )
                
                p1 = gr.Slider(
                    minimum=-0.1, maximum=0.1, step=0.005, value=0.0,
                    label="P1 (Tangential)", info="First tangential distortion coefficient"
                )
                
                p2 = gr.Slider(
                    minimum=-0.1, maximum=0.1, step=0.005, value=0.0,
                    label="P2 (Tangential)", info="Second tangential distortion coefficient"
                )
                
                gr.Markdown("### ⚙️ Processing Parameters")
                
                pattern_type = gr.Dropdown(
                    choices=["checkerboard", "grid", "circles", "text"],
                    value="checkerboard",
                    label="Test Pattern", info="Type of test pattern to generate"
                )
                
                correction_method = gr.Dropdown(
                    choices=["original", "advanced_iterative", "advanced_analytical"],
                    value="advanced_iterative",
                    label="Correction Method", 
                    info="Choose correction algorithm"
                )
                
                gr.Markdown("### 📁 Export Options")
                
                with gr.Row():
                    export_csv_btn = gr.Button("📊 Export Displacement CSV", variant="secondary")
                    process_btn = gr.Button("🔄 Process Distortion", variant="primary", size="lg")
                
                with gr.Row():
                    test_accuracy_btn = gr.Button("🎯 Test Correction Accuracy", variant="secondary")
                    validate_quality_btn = gr.Button("✅ Validate Quality", variant="secondary")
                
                # GDC Export Parameters
                gr.Markdown("### 🔧 GDC Grid Export")
                
                gdc_width = gr.Slider(
                    minimum=1024, maximum=16384, step=256, value=8192,
                    label="GDC Target Width", info="Target image width for GDC format"
                )
                
                gdc_height = gr.Slider(
                    minimum=768, maximum=12288, step=192, value=6144,
                    label="GDC Target Height", info="Target image height for GDC format"
                )
                
                export_gdc_csv_btn = gr.Button("🎯 Export GDC Grid CSV", variant="secondary")
                
                csv_output = gr.Textbox(
                    label="Displacement CSV Data",
                    lines=8,
                    placeholder="Click 'Export Displacement CSV' to generate displacement data...",
                    visible=True
                )
                
                gdc_csv_output = gr.Textbox(
                    label="GDC Grid CSV Data",
                    lines=8,
                    placeholder="Click 'Export GDC Grid CSV' to generate GDC format data...",
                    visible=True
                )
                
                accuracy_output = gr.Textbox(
                    label="Correction Accuracy Test Results",
                    lines=12,
                    placeholder="Click 'Test Correction Accuracy' to compare different methods...",
                    visible=True
                )
                
                validation_output = gr.Textbox(
                    label="Quality Validation Results",
                    lines=15,
                    placeholder="Click 'Validate Quality' to analyze correction performance...",
                    visible=True
                )
                
                gr.Markdown("""
                ### 💡 Quick Presets
                - **Barrel Distortion**: K1=-0.2, K2=0.05
                - **Pincushion**: K1=0.15, K2=-0.03
                - **Mild Fisheye**: K1=-0.4, K2=0.1, K3=-0.02
                
                ### 🔧 Correction Methods
                - **Original**: Basic correction using forward distortion mapping
                - **Advanced Iterative**: Newton-Raphson iterative inverse mapping (most accurate)
                - **Advanced Analytical**: Closed-form solution for simple radial distortion (fastest)
                
                ### ℹ️ GDC Format Info
                - **GDC**: Geometric Distortion Correction format
                - **Fixed-point**: Integer values for hardware implementation
                - **Scale factors**: Based on bit-shifting operations
                - **Applications**: Camera ISP, FPGA processing, embedded systems
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Image Results")
                
                with gr.Tab("Original vs Distorted vs Corrected"):
                    with gr.Row():
                        original_out = gr.Image(label="Original Image", height=250)
                        distorted_out = gr.Image(label="Distorted Image", height=250)
                        corrected_out = gr.Image(label="Corrected Image", height=250)
                
                with gr.Tab("Grid Visualization"):
                    grid_vis_out = gr.Image(label="Distortion Grid Mapping", height=400)
                
                with gr.Tab("Distortion Heatmap"):
                    heatmap_out = gr.Image(label="Distortion Magnitude Heatmap", height=400)
                
                gr.Markdown("""
                ### 📊 Understanding the Results
                
                - **Grid Visualization**: Shows how the regular grid is deformed by distortion
                - **Heatmap**: Color-coded magnitude of distortion at each grid point
                - **Original**: Test pattern without any distortion
                - **Distorted**: Test pattern with applied barrel distortion
                - **Corrected**: Distorted image after correction algorithm
                
                ### 🎯 Export Formats
                
                - **Displacement CSV**: Raw pixel displacements for analysis
                - **GDC Grid CSV**: Hardware-ready fixed-point format with hex values
                
                ### 🔬 Quality Assessment
                
                - **Test Accuracy**: Compare correction methods on specific points
                - **Validate Quality**: Comprehensive quality metrics (PSNR, correlation, geometric error)
                """)
        
        # Connect the processing function
        process_btn.click(
            fn=process_distortion,
            inputs=[image_width, image_height, grid_rows, grid_cols, 
                   k1, k2, k3, p1, p2, pattern_type, correction_method],
            outputs=[original_out, distorted_out, corrected_out, grid_vis_out, heatmap_out]
        )
        
        # Connect CSV export functions
        export_csv_btn.click(
            fn=export_grid_csv,
            outputs=csv_output
        )
        
        export_gdc_csv_btn.click(
            fn=export_gdc_grid_csv,
            inputs=[gdc_width, gdc_height],
            outputs=gdc_csv_output
        )
        
        # Connect accuracy testing and validation functions
        test_accuracy_btn.click(
            fn=test_correction_accuracy,
            outputs=accuracy_output
        )
        
        validate_quality_btn.click(
            fn=validate_correction_quality,
            inputs=[pattern_type, correction_method],
            outputs=validation_output
        )
        
        # Auto-process with default values on load
        interface.load(
            fn=process_distortion,
            inputs=[image_width, image_height, grid_rows, grid_cols, 
                   k1, k2, k3, p1, p2, pattern_type, correction_method],
            outputs=[original_out, distorted_out, corrected_out, grid_vis_out, heatmap_out]
        )
    
    return interface

# Launch the interface
if __name__ == "__main__":
    # Generate a random port to avoid conflicts
    port = random.randint(7861, 7890)
    
    # Create and launch the interface
    interface = create_gradio_interface()
    
    try:
        interface.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=port,
            show_error=True
        )
    except OSError as e:
        print(f"Port {port} is busy, trying another port...")
        # Try a few different ports
        for backup_port in range(7891, 7900):
            try:
                interface.launch(
                    share=True,
                    server_name="0.0.0.0",
                    server_port=backup_port,
                    show_error=True
                )
                break
            except OSError:
                continue
        else:
            print("Could not find an available port. Please try again later.")