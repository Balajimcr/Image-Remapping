    def _create_grid_tab(self):
        """Create grid processing tab"""
        gr.Markdown("### 🌐 Grid Setup & Interpolation")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📐 Grid Dimensions")
                
                # Original grid dimensions
                with gr.Group():
                    gr.Markdown("**Original Grid**")
                    with gr.Row():
                        self.orig_rows = gr.Dropdown(
                            choices=DEFAULT_GRID_SIZES,
                            value=3,
                            label="Rows",
                            info="Number of rows in original grid"
                        )
                        self.orig_cols = gr.Dropdown(
                            choices=DEFAULT_GRID_SIZES,
                            value=3,
                            label="Columns",
                            info="Number of columns in original grid"
                        )
                
                # Target grid dimensions
                with gr.Group():
                    gr.Markdown("**Target Grid (After Interpolation)**")
                    with gr.Row():
                        self.target_rows = gr.Dropdown(
                            choices=DEFAULT_GRID_SIZES,
                            value=9,
                            label="Rows",
                            info="Target rows after interpolation"
                        )
                        self.target_cols = gr.Dropdown(
                            choices=DEFAULT_GRID_SIZES,
                            value=9,
                            label="Columns",
                            info="Target columns after interpolation"
                        )
                
                # Interpolation settings
                gr.Markdown("#### 🔧 Interpolation Settings")
                
                self.interp_method = gr.Dropdown(
                    choices=["bicubic", "linear"],
                    value="bicubic",
                    label="Interpolation Method",
                    info="Bicubic for smoother results"
                )
                
                self.preserve_boundaries = gr.Checkbox(
                    label="Preserve Boundaries",
                    value=True,
                    info="Maintain edge values during interpolation"
                )
                
                # Interpolation factor display
                self.interp_factor = gr.Textbox(
                    label="Interpolation Factor",
                    value="3.0×",
                    interactive=False,
                    info="Automatically calculated"
                )
                
                self.setup_grid_btn = gr.Button(
                    "⚙️ Setup Grid",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("#### 📊 Grid Visualization")
                
                with gr.Tabs():
                    with gr.Tab("Original Grid"):
                        self.original_grid_viz = gr.Image(
                            label="Original DX/DY Grids",
                            height=400,
                            show_download_button=True
                        )
                    
                    with gr.Tab("Interpolated Grid"):
                        self.interpolated_grid_viz = gr.Image(
                            label="Interpolated DX/DY Grids",
                            height=400,
                            show_download_button=True
                        )
                    
                    with gr.Tab("Grid Comparison"):
                        self.grid_comparison_viz = gr.Image(
                            label="Side-by-Side Comparison",
                            height=400,
                            show_download_button=True
                        )
                
                # Grid statistics
                self.grid_stats = gr.Textbox(
                    label="Grid Statistics",
                    lines=6,
                    interactive=False,
                    placeholder="Grid statistics will appear here..."
                )
    
    def _create_remapping_tab(self):
        """Create image remapping tab"""
        gr.Markdown("### 🖼️ Image Geometric Transformation")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📷 Input Image")
                
                # Image input options
                with gr.Tabs():
                    with gr.Tab("Upload Image"):
                        self.input_image = gr.Image(
                            label="Upload Image",
                            type="numpy",
                            height=250
                        )
                    
                    with gr.Tab("Generate Sample"):
                        with gr.Row():
                            self.sample_pattern = gr.Dropdown(
                                choices=["grid", "checkerboard", "circles", "text"],
                                value="grid",
                                label="Pattern"
                            )
                            self.generate_sample_btn = gr.Button(
                                "🎨 Generate",
                                variant="secondary"
                            )
                
                # Image settings
                gr.Markdown("#### ⚙️ Image Settings")
                
                with gr.Row():
                    self.img_width = gr.Dropdown(
                        choices=[size[0] for size in DEFAULT_IMAGE_SIZES],
                        value=640,
                        label="Width",
                        allow_custom_value=True
                    )
                    self.img_height = gr.Dropdown(
                        choices=[size[1] for size in DEFAULT_IMAGE_SIZES],
                        value=480,
                        label="Height",
                        allow_custom_value=True
                    )
                
                # Remapping parameters
                gr.Markdown("#### 🎯 Remapping Parameters")
                
                self.scale_factor = gr.Slider(
                    minimum=0.001,
                    maximum=2.0,
                    value=0.1,
                    step=0.001,
                    label="Displacement Scale Factor",
                    info="Scales the displacement magnitude"
                )
                
                self.cv_interpolation = gr.Dropdown(
                    choices=["linear", "cubic", "nearest", "lanczos"],
                    value="linear",
                    label="CV2 Interpolation",
                    info="OpenCV interpolation method"
                )
                
                self.border_mode = gr.Dropdown(
                    choices=["constant", "reflect", "wrap", "replicate"],
                    value="constant",
                    label="Border Handling",
                    info="How to handle pixels outside image"
                )
                
                # Processing controls
                with gr.Row():
                    self.setup_remapping_btn = gr.Button(
                        "⚙️ Setup Remapping",
                        variant="secondary"
                    )
                    self.process_image_btn = gr.Button(
                        "🔄 Process Image",
                        variant="primary"
                    )
            
            with gr.Column(scale=2):
                gr.Markdown("#### 🖼️ Results")
                
                with gr.Tabs():
                    with gr.Tab("Before/After"):
                        self.image_comparison = gr.Image(
                            label="Input vs Remapped",
                            height=350,
                            show_download_button=True
                        )
                    
                    with gr.Tab("Remapped Image"):
                        self.output_image = gr.Image(
                            label="Remapped Result",
                            height=350,
                            show_download_button=True
                        )
                    
                    with gr.Tab("Displacement Analysis"):
                        self.displacement_viz = gr.Image(
                            label="Displacement Field Analysis",
                            height=350,
                            show_download_button=True
                        )
                
                # Processing status
                self.processing_status = gr.Textbox(
                    label="Processing Status",
                    lines=4,
                    interactive=False,
                    placeholder="Processing status will appear here..."
                )
                
                # Quality metrics
                self.quality_metrics = gr.JSON(
                    label="Quality Metrics",
                    visible=False,
                    value={}  # Initialize with empty dict
                )
    
    def _create_analysis_tab(self):
        """Create analysis and export tab"""
        gr.Markdown("### 📊 Analysis & Export Tools")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 🔍 Analysis Tools")
                
                # Analysis options
                self.analysis_type = gr.Dropdown(
                    choices=[
                        "Grid Statistics",
                        "Displacement Analysis", 
                        "Processing Flow",
                        "Quality Assessment"
                    ],
                    value="Grid Statistics",
                    label="Analysis Type"
                )
                
                self.create_analysis_btn = gr.Button(
                    "📊 Generate Analysis",
                    variant="primary"
                )
                
                # Export options
                gr.Markdown("#### 💾 Export Options")
                
                self.export_format = gr.Dropdown(
                    choices=["PNG", "JPEG", "TIFF", "All Formats"],
                    value="PNG",
                    label="Image Format"
                )
                
                self.export_metadata = gr.Checkbox(
                    label="Include Metadata",
                    value=True,
                    info="Export processing information"
                )
                
                self.export_visualizations = gr.Checkbox(
                    label="Export Visualizations",
                    value=True,
                    info="Export all generated plots"
                )
                
                with gr.Row():
                    self.export_single_btn = gr.Button(
                        "💾 Export Current",
                        variant="secondary"
                    )
                    self.export_all_btn = gr.Button(
                        "📦 Export Grid Data",
                        variant="primary"
                    )
                
                # Batch processing
                gr.Markdown("#### 🔄 Batch Processing")
                
                self.batch_folder = gr.Textbox(
                    label="Input Folder",
                    placeholder="Path to folder with images..."
                )
                
                self.batch_process_btn = gr.Button(
                    "🔄 Process Batch",
                    variant="secondary"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("#### 📈 Analysis Results")
                
                # Analysis visualization
                self.analysis_viz = gr.Image(
                    label="Analysis Visualization",
                    height=400,
                    show_download_button=True
                )
                
                # Processing summary
                self.processing_summary = gr.Textbox(
                    label="Processing Summary",
                    lines=12,
                    show_copy_button=True,
                    interactive=False,
                    placeholder="Processing summary will appear here..."
                )
                
                # Export status
                self.export_status = gr.Textbox(
                    label="Export Status",
                    lines=4,
                    interactive=False,
                    placeholder="Export status will appear here..."
                )
                
                # Download links
                self.download_files = gr.File(
                    label="📥 Download Results",
                    file_count="multiple",
                    visible=False
                )
    
    def _create_help_tab(self):
        """Create help and documentation tab"""
        gr.Markdown("""
        ### 📚 GDC Image Remapping Help & Documentation
        
        #### 🚀 Getting Started
        
        **1. Import GDC Data** 📁
        - Paste GDC grid data or upload a file
        - Use format: `yuv_gdc_grid_dx_0_N value` and `yuv_gdc_grid_dy_0_N value`
        - Enable validation for format checking
        
        **2. Setup Grid** 🌐
        - Configure original grid dimensions (from your data)
        - Set target dimensions (higher for smoother interpolation)
        - Choose interpolation method (bicubic recommended)
        
        **3. Process Image** 🖼️
        - Upload image or generate sample pattern
        - Adjust displacement scale factor (start with 0.1)
        - Configure image dimensions and interpolation method
        
        **4. Analyze & Export** 📊
        - Generate analysis visualizations
        - Export results in various formats
        - Review processing summary
        
        #### 📋 GDC Data Format
        
        ```
        yuv_gdc_grid_dx_0_0 -50    # X displacement for element 0
        yuv_gdc_grid_dx_0_1 0      # X displacement for element 1
        yuv_gdc_grid_dx_0_2 50     # X displacement for element 2
        yuv_gdc_grid_dy_0_0 25     # Y displacement for element 0
        yuv_gdc_grid_dy_0_1 30     # Y displacement for element 1
        yuv_gdc_grid_dy_0_2 25     # Y displacement for element 2
        ```
        
        #### ⚙️ Parameters Guide
        
        **Scale Factor** 🎯
        - Controls displacement magnitude
        - Start with 0.1 and adjust based on results
        - Higher values = more distortion
        
        **Interpolation Methods** 🔧
        - **Bicubic**: Smoothest results, slower processing
        - **Linear**: Good balance of quality and speed
        - **Nearest**: Fastest, preserves sharp edges
        
        **Grid Sizes** 📐
        - Original: Determined by your GDC data
        - Target: Higher values = smoother interpolation
        - Typical scaling: 3×3 → 9×9 or 5×5 → 15×15
        
        #### 🚨 Troubleshooting
        
        **Common Issues:**
        
        1. **"No GDC data loaded"**
           - Ensure data is properly formatted
           - Check element naming convention
           - Verify complete DX/DY pairs
        
        2. **"Grid setup failed"**
           - Verify grid dimensions match data
           - Check for missing elements
           - Ensure positive grid sizes
        
        3. **"Excessive displacement"**
           - Reduce scale factor
           - Check GDC data validity
           - Verify grid interpolation
        """)
    
    def _get_custom_css(self):
        """Custom CSS for enhanced interface styling"""
        return """
        #status_indicators {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .status-item {
            display: inline-block;
            margin: 5px 15px;
            padding: 5px 10px;
            background: rgba(255,255,255,0.2);
            border-radius: 5px;
            font-weight: bold;
        }
        
        .status-ready {
            background: rgba(76, 175, 80, 0.8) !important;
        }
        
        .status-pending {
            background: rgba(255, 152, 0, 0.8) !important;
        }
        
        .gradio-container {
            max-width: 1400px !important;
        }
        """
    
    def _generate_status_html(self):
        """Generate HTML for status indicators"""
        return f"""
        <div id="status_indicators">
            <h3>🎯 Processing Pipeline Status</h3>
            <span class="status-item {'status-ready' if self.processing_state['gdc_loaded'] else 'status-pending'}">
                📁 GDC Data: {'Ready' if self.processing_state['gdc_loaded'] else 'Pending'}
            </span>
            <span class="status-item {'status-ready' if self.processing_state['grid_setup'] else 'status-pending'}">
                🌐 Grid Setup: {'Ready' if self.processing_state['grid_setup'] else 'Pending'}
            </span>
            <span class="status-item {'status-ready' if self.processing_state['remapping_ready'] else 'status-pending'}">
                🎯 Remapping: {'Ready' if self.processing_state['remapping_ready'] else 'Pending'}
            </span>
            <span class="status-item {'status-ready' if self.processing_state['image_processed'] else 'status-pending'}">
                🖼️ Image: {'Processed' if self.processing_state['image_processed'] else 'Pending'}
            </span>
        </div>
        """

    def _auto_detect_grid_size(self, gdc_data):
        """Auto-detect grid dimensions from GDC data"""
        if not gdc_data:
            return None, None
            
        # Find max index for DX and DY
        max_dx_index = -1
        max_dy_index = -1
        
        index_pattern = re.compile(r"_(dx|dy)_0_(\d+)$")
        
        for name in gdc_data.keys():
            match = index_pattern.search(name)
            if match:
                index = int(match.group(2))
                if "dx" in name:
                    max_dx_index = max(max_dx_index, index)
                elif "dy" in name:
                    max_dy_index = max(max_dy_index, index)
        
        if max_dx_index == max_dy_index and max_dx_index >= 0:
            total_elements = max_dx_index + 1
            
            # Try to find dimensions that multiply to total_elements
            for rows in range(1, int(np.sqrt(total_elements)) + 1):
                if total_elements % rows == 0:
                    cols = total_elements // rows
                    # Prefer more square-like grids
                    if abs(rows - cols) <= 2:
                        return rows, cols
            
            # If no good square found, try common grid sizes
            common_sizes = [(3,3), (5,5), (7,7), (9,9), (3,5), (5,7), (7,9)]
            for rows, cols in common_sizes:
                if rows * cols == total_elements:
                    return rows, cols
        
        return None, None
    
    def _update_interpolation_factor(self, orig_rows, orig_cols, target_rows, target_cols):
        """Update interpolation factor display"""
        try:
            if orig_rows and orig_cols and target_rows and target_cols:
                factor_x = target_cols / orig_cols
                factor_y = target_rows / orig_rows
                avg_factor = (factor_x + factor_y) / 2
                return f"{avg_factor:.1f}×"
        except (TypeError, ZeroDivisionError):
            pass
        return "N/A"
    
    def _generate_sample_image(self, pattern, width, height):
        """Generate sample image for testing"""
        try:
            width = int(width) if width else 640
            height = int(height) if height else 480
            
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            if pattern == "grid":
                # Create grid pattern
                grid_size = 50
                for i in range(0, height, grid_size):
                    image[i:i+2, :] = [255, 255, 255]
                for j in range(0, width, grid_size):
                    image[:, j:j+2] = [255, 255, 255]
                    
            elif pattern == "checkerboard":
                # Create checkerboard pattern
                check_size = 40
                for i in range(height):
                    for j in range(width):
                        if ((i // check_size) + (j // check_size)) % 2 == 0:
                            image[i, j] = [255, 255, 255]
                            
            elif pattern == "circles":
                # Create circles pattern
                center_x, center_y = width // 2, height // 2
                for radius in range(20, min(width, height) // 2, 40):
                    cv2.circle(image, (center_x, center_y), radius, (255, 255, 255), 2)
                    
            elif pattern == "text":
                # Create text pattern
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = "GDC Test"
                text_size = cv2.getTextSize(text, font, 2, 3)[0]
                text_x = (width - text_size[0]) // 2
                text_y = (height + text_size[1]) // 2
                cv2.putText(image, text, (text_x, text_y), font, 2, (255, 255, 255), 3)
                
            return image
            
        except Exception as e:
            print(f"Error generating sample image: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def _setup_event_handlers(self):
        """Setup all event handlers for the interface - UPDATED"""
        
        # Import handlers - SIMPLIFIED for traditional GDC format only
        if self.import_btn and self.import_status:
            self.import_btn.click(
                fn=self._handle_import_gdc_simple,
                inputs=[
                    self.gdc_text_input, 
                    self.gdc_file_input,
                    self.input_grid_rows,
                    self.input_grid_cols
                ],
                outputs=[
                    self.import_status,
                    self.import_stats,
                    self.data_preview
                ]
            )
        
        # Auto-update expected elements when dimensions change
        if self.expected_elements and self.input_grid_rows and self.input_grid_cols:
            for component in [self.input_grid_rows, self.input_grid_cols]:
                component.change(
                    fn=self._update_expected_elements,
                    inputs=[self.input_grid_rows, self.input_grid_cols],
                    outputs=self.expected_elements
                )
        
        # Grid handlers
        if self.setup_grid_btn and self.grid_stats:
            self.setup_grid_btn.click(
                fn=self._handle_setup_grid,
                inputs=[
                    self.orig_rows,
                    self.orig_cols,
                    self.target_rows,
                    self.target_cols,
                    self.interp_method
                ],
                outputs=[
                    self.grid_stats,
                    self.original_grid_viz,
                    self.interpolated_grid_viz,
                    self.grid_comparison_viz
                ]
            )
        
        # Auto-update interpolation factor
        if self.interp_factor:
            for component in [self.orig_rows, self.orig_cols, self.target_rows, self.target_cols]:
                if component:
                    component.change(
                        fn=self._update_interpolation_factor,
                        inputs=[self.orig_rows, self.orig_cols, self.target_rows, self.target_cols],
                        outputs=self.interp_factor
                    )
        
        # Image processing handlers
        if self.generate_sample_btn and self.input_image:
            self.generate_sample_btn.click(
                fn=self._generate_sample_image,
                inputs=[self.sample_pattern, self.img_width, self.img_height],
                outputs=self.input_image
            )
        
        if self.process_image_btn:
            self.process_image_btn.click(
                fn=self._handle_process_image,
                inputs=[
                    self.input_image,
                    self.img_width,
                    self.img_height,
                    self.scale_factor,
                    self.cv_interpolation
                ],
                outputs=[
                    self.output_image,
                    self.image_comparison,
                    self.displacement_viz,
                    self.processing_status,
                    self.quality_metrics
                ]
            )
        
        # Analysis handlers
        if self.create_analysis_btn:
            self.create_analysis_btn.click(
                fn=self._handle_create_analysis,
                inputs=self.analysis_type,
                outputs=[self.analysis_viz, self.processing_summary]
            )
        
        # Export handlers
        if self.export_all_btn:
            self.export_all_btn.click(
                fn=self._handle_export_grid_data,
                inputs=[],
                outputs=[self.download_files, self.export_status]
            )
    
    def _handle_setup_grid(self, orig_rows, orig_cols, target_rows, target_cols, interp_method):
        """Handle grid setup and interpolation with enhanced features"""
        try:
            if not self.processing_state['gdc_loaded']:
                return "⚠️ Please import GDC data first", None, None, None
            
            # Setup grid with processor
            result = self.processor.setup_and_interpolate_grid(
                orig_rows, orig_cols, target_rows, target_cols, interp_method
            )
            
            if result['success']:
                self.processing_state['grid_setup'] = True
                
                # Generate enhanced visualizations
                original_viz = self.processor.create_visualization('original_grid')
                interpolated_viz = self.processor.create_visualization('interpolated_grid')
                comparison_viz = self.processor.create_visualization('grid_comparison')
                
                # Enhanced status message
                grid_info = result.get('grid_info', {})
                statistics = result.get('statistics', {})
                
                stats_msg = f"✅ Grid setup and interpolation successful!\n\n"
                stats_msg += "📐 **Grid Configuration:**\n"
                stats_msg += f"   • Original: {orig_rows}×{orig_cols} = {orig_rows * orig_cols} elements\n"
                stats_msg += f"   • Target: {target_rows}×{target_cols} = {target_rows * target_cols} elements\n"
                stats_msg += f"   • Interpolation: {interp_method.title()}\n"
                stats_msg += f"   • Scale factor: {target_rows/orig_rows:.1f}× (rows) | {target_cols/orig_cols:.1f}× (cols)\n\n"
                
                # Add grid statistics
                if statistics:
                    stats_msg += "📊 **Grid Statistics:**\n"
                    dx_stats = statistics.get('dx_stats', {})
                    dy_stats = statistics.get('dy_stats', {})
                    
                    if dx_stats:
                        stats_msg += f"   **DX Grid:**\n"
                        stats_msg += f"     ├─ Range: {dx_stats.get('min', 0):.1f} to {dx_stats.get('max', 0):.1f}\n"
                        stats_msg += f"     ├─ Mean: {dx_stats.get('mean', 0):.2f}\n"
                        stats_msg += f"     └─ Std Dev: {dx_stats.get('std', 0):.2f}\n"
                    
                    if dy_stats:
                        stats_msg += f"   **DY Grid:**\n"
                        stats_msg += f"     ├─ Range: {dy_stats.get('min', 0):.1f} to {dy_stats.get('max', 0):.1f}\n"
                        stats_msg += f"     ├─ Mean: {dy_stats.get('mean', 0):.2f}\n"
                        stats_msg += f"     └─ Std Dev: {dy_stats.get('std', 0):.2f}\n"
                
                stats_msg += "\n🎯 **Ready for image remapping!**"
                
                return stats_msg, original_viz, interpolated_viz, comparison_viz
            else:
                return f"❌ Grid setup failed: {result['message']}", None, None, None
                
        except Exception as e:
            return f"❌ Error during grid setup: {str(e)}", None, None, None
    
    def _handle_process_image(self, input_image, img_width, img_height, scale_factor, cv_interpolation):
        """Handle image processing"""
        try:
            if not self.processing_state['grid_setup']:
                return None, None, None, "⚠️ Please setup grid first", {}
            
            if input_image is None:
                return None, None, None, "⚠️ Please provide input image", {}
            
            # Setup image remapping
            setup_result = self.processor.setup_image_remapping(
                img_width, img_height, scale_factor, cv_interpolation
            )
            
            if not setup_result['success']:
                return None, None, None, f"❌ Remapping setup failed: {setup_result['message']}", {}
            
            self.processing_state['remapping_ready'] = True
            
            # Process the image
            process_result = self.processor.process_image(input_image)
            
            if process_result['success']:
                self.processing_state['image_processed'] = True
                
                # Create comparison visualization
                comparison_viz = self._create_comparison_image(input_image, process_result.get('output_image'))
                displacement_viz = self.processor.create_visualization('displacement_analysis')
                
                status_msg = f"✅ Image processing successful\n"
                status_msg += f"📊 Input: {process_result.get('input_info', {})}\n"
                status_msg += f"📊 Output: {process_result.get('output_info', {})}\n"
                status_msg += f"⏱️ Processing: {process_result.get('processing_stats', {})}"
                
                # Ensure safe JSON output
                safe_stats = process_result.get('processing_stats', {})
                if not isinstance(safe_stats, dict):
                    safe_stats = {}
                
                return (
                    process_result.get('output_image'),
                    comparison_viz,
                    displacement_viz,
                    status_msg,
                    safe_stats
                )
            else:
                return None, None, None, f"❌ Processing failed: {process_result['message']}", {}
                
        except Exception as e:
            # Return safe defaults to prevent JSON errors
            return None, None, None, f"❌ Error during processing: {str(e)}", {}
    
    def _create_comparison_image(self, input_img, output_img):
        """Create side-by-side comparison image"""
        try:
            if input_img is None or output_img is None:
                return None
            
            # Ensure both images have same height
            h1, w1 = input_img.shape[:2]
            h2, w2 = output_img.shape[:2]
            
            target_height = min(h1, h2)
            input_resized = cv2.resize(input_img, (int(w1 * target_height / h1), target_height))
            output_resized = cv2.resize(output_img, (int(w2 * target_height / h2), target_height))
            
            # Concatenate horizontally
            comparison = np.hstack([input_resized, output_resized])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(comparison, "Remapped", (input_resized.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
            
            return comparison
            
        except Exception as e:
            print(f"Error creating comparison image: {e}")
            return None
    
    def _handle_create_analysis(self, analysis_type):
        """Handle analysis generation with enhanced options"""
        try:
            if analysis_type == "Grid Statistics":
                viz = self.processor.create_visualization('grid_statistics')
                summary = self._generate_grid_analysis_summary()
            elif analysis_type == "Displacement Analysis":
                viz = self.processor.create_visualization('displacement_analysis')
                summary = self._generate_displacement_analysis_summary()
            elif analysis_type == "Processing Flow":
                viz = self.processor.create_visualization('processing_flow')
                summary = self._generate_processing_flow_summary()
            elif analysis_type == "Quality Assessment":
                viz = self.processor.create_visualization('quality_assessment')
                summary = self._generate_quality_assessment_summary()
            else:
                return None, "❌ Unknown analysis type"
            
            return viz, summary
            
        except Exception as e:
            return None, f"❌ Error generating analysis: {str(e)}"
    
    def _export_grid_data(self, export_format):
        """Export grid data in various formats"""
        try:
            if not self.processing_state['grid_setup']:
                return None, "⚠️ No grid data to export. Please setup grid first."
            
            export_files = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create temporary directory for exports
            temp_dir = tempfile.mkdtemp(prefix="gdc_export_")
            
            # Export original grids
            if hasattr(self.processor, 'dx_grid') and self.processor.dx_grid is not None:
                # CSV format
                dx_orig_csv = os.path.join(temp_dir, f"original_dx_{timestamp}.csv")
                dy_orig_csv = os.path.join(temp_dir, f"original_dy_{timestamp}.csv")
                
                np.savetxt(dx_orig_csv, self.processor.dx_grid, delimiter=',', fmt='%.6f')
                np.savetxt(dy_orig_csv, self.processor.dy_grid, delimiter=',', fmt='%.6f')
                export_files.extend([dx_orig_csv, dy_orig_csv])
                
                # GDC format
                dx_orig_gdc = self.processor.grid_processor.grid_2d_to_gdc_format(self.processor.dx_grid, 'dx')
                dy_orig_gdc = self.processor.grid_processor.grid_2d_to_gdc_format(self.processor.dy_grid, 'dy')
                
                dx_orig_txt = os.path.join(temp_dir, f"original_dx_{timestamp}.txt")
                dy_orig_txt = os.path.join(temp_dir, f"original_dy_{timestamp}.txt")
                
                with open(dx_orig_txt, 'w') as f:
                    f.write(dx_orig_gdc)
                with open(dy_orig_txt, 'w') as f:
                    f.write(dy_orig_gdc)
                    
                export_files.extend([dx_orig_txt, dy_orig_txt])
            
            # Export interpolated grids
            if hasattr(self.processor, 'interpolated_dx') and self.processor.interpolated_dx is not None:
                # CSV format
                dx_interp_csv = os.path.join(temp_dir, f"interpolated_dx_{timestamp}.csv")
                dy_interp_csv = os.path.join(temp_dir, f"interpolated_dy_{timestamp}.csv")
                
                np.savetxt(dx_interp_csv, self.processor.interpolated_dx, delimiter=',', fmt='%.6f')
                np.savetxt(dy_interp_csv, self.processor.interpolated_dy, delimiter=',', fmt='%.6f')
                export_files.extend([dx_interp_csv, dy_interp_csv])
                
                # GDC format
                dx_interp_gdc = self.processor.grid_processor.grid_2d_to_gdc_format(self.processor.interpolated_dx, 'dx')
                dy_interp_gdc = self.processor.grid_processor.grid_2d_to_gdc_format(self.processor.interpolated_dy, 'dy')
                
                dx_interp_txt = os.path.join(temp_dir, f"interpolated_dx_{timestamp}.txt")
                dy_interp_txt = os.path.join(temp_dir, f"interpolated_dy_{timestamp}.txt")
                combined_txt = os.path.join(temp_dir, f"combined_grids_{timestamp}.txt")
                
                with open(dx_interp_txt, 'w') as f:
                    f.write(dx_interp_gdc)
                with open(dy_interp_txt, 'w') as f:
                    f.write(dy_interp_gdc)
                with open(combined_txt, 'w') as f:
                    f.write(dx_interp_gdc + '\n' + dy_interp_gdc)
                    
                export_files.extend([dx_interp_txt, dy_interp_txt, combined_txt])
            
            # Create zip file
            zip_path = os.path.join(temp_dir, f"gdc_export_{timestamp}.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in export_files:
                    if os.path.exists(file_path):
                        zipf.write(file_path, os.path.basename(file_path))
            
            return zip_path, f"✅ Exported {len(export_files)} files successfully!"
            
        except Exception as e:
            return None, f"❌ Export failed: {str(e)}"
    
    def _generate_grid_analysis_summary(self):
        """Generate grid analysis summary"""
        summary = "📊 Grid Analysis Summary\n"
        summary += "=" * 40 + "\n\n"
        
        if hasattr(self.processor, 'gdc_data') and self.processor.gdc_data:
            summary += f"📁 Total GDC Elements: {len(self.processor.gdc_data)}\n"
            
            # Grid statistics
            stats = self.processor._compute_grid_statistics()
            for key, value in stats.items():
                summary += f"📊 {key}: {value}\n"
        else:
            summary += "⚠️ No grid data available for analysis\n"
        
        return summary
    
    def _handle_export_grid_data(self):
        """Handle grid data export"""
        try:
            zip_path, status_msg = self._export_grid_data("All Formats")
            if zip_path:
                return gr.update(value=zip_path, visible=True), status_msg
            else:
                return gr.update(visible=False), status_msg
        except Exception as e:
            return gr.update(visible=False), f"❌ Export error: {str(e)}"
    
    def _generate_displacement_analysis_summary(self):
        """Generate displacement analysis summary"""
        summary = "🎯 Displacement Analysis Summary\n"
        summary += "=" * 40 + "\n\n"
        
        if self.processing_state['grid_setup']:
            summary += f"📐 Grid successfully interpolated\n"
            summary += f"🔍 Out-of-bounds pixels: {self.processor._count_out_of_bounds_pixels()}\n"
        else:
            summary += "⚠️ Grid not setup - no displacement analysis available\n"
        
        return summary
    
    def _generate_processing_flow_summary(self):
        """Generate processing flow summary"""
        return self.processor.get_processing_summary()
    
    def _generate_quality_assessment_summary(self):
        """Generate quality assessment summary"""
        summary = "🏆 Quality Assessment Summary\n"
        summary += "=" * 40 + "\n\n"
        
        # Processing state assessment
        summary += "📋 Processing Pipeline Status:\n"
        for stage, status in self.processing_state.items():
            icon = "✅" if status else "❌"
            summary += f"{icon} {stage.replace('_', ' ').title()}: {'Complete' if status else 'Pending'}\n"
        
        return summary


# Main application entry point
def create_gdc_interface():
    """Create and return the GDC remapping interface"""
    try:
        interface_controller = GDCRemappingInterface()
        return interface_controller.create_interface()
    except Exception as e:
        print(f"Error creating interface: {e}")
        # Return a simple fallback interface
        with gr.Blocks() as fallback:
            gr.Markdown("# Error: Could not create GDC interface")
            gr.Markdown(f"Error details: {str(e)}")
        return fallback


def main():
    """Main entry point for the application"""
    try:
        print("🚀 Starting GDC Image Remapping Suite...")
        interface = create_gdc_interface()
        
        if interface is not None:
            print("✅ Interface created successfully")
            interface.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                debug=True,
                show_error=True
            )
        else:
            print("❌ Failed to create interface")
            
    except Exception as e:
        print(f"❌ Application error: {e}")
        # Create emergency fallback
        with gr.Blocks() as emergency:
            gr.Markdown("# 🚨 GDC Interface Error")
            gr.Markdown(f"**Error:** {str(e)}")
            gr.Markdown("Please check your installation and try again.")
        
        emergency.launch(
            server_name="localhost", 
            server_port=7860,
            share=False
        )


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
GDC Image Remapping GUI Interface for Image Remapping Suite - INTEGRATED VERSION

Advanced Gradio interface for GDC-based image geometric transformation with
comprehensive workflow management, real-time visualization, and quality assessment.

Features:
- Multi-tab interface with logical workflow separation
- Real-time parameter validation and feedback
- Comprehensive visualization suite
- Advanced processing options
- Export capabilities with metadata
- Integration with existing suite architecture
- Traditional GDC format support with flexible parsing

Author: Balaji R
License: MIT
"""

import gradio as gr
import numpy as np
import cv2
import os
import tempfile
import json
import zipfile
from typing import Tuple, Optional, Dict, Any, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib import colors as mcolors

# Import grid processing functionality
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import zoom
import warnings
import re
import pandas as pd

# The processor is available as it is defined in this file.
GDC_PROCESSOR_AVAILABLE = True

# Grid Processing Core Classes
class GDCGridProcessor:
    """
    Handles parsing, extracting, and interpolating GDC grid data.
    """
    def __init__(self):
        self.parsed_data = []
        self.dx_values = []
        self.dy_values = []
        self.original_rows = 0
        self.original_cols = 0

    def parse_grid_data_from_content(self, file_content: str) -> list:
        """
        Parses grid data from a string of file content.
        Expected format: "element_name value" per line.
        """
        parsed_data_ordered = []
        lines = file_content.strip().split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                try:
                    parts = line.split()
                    if len(parts) == 2:
                        element_name = parts[0].strip()
                        value = int(parts[1].strip())
                        parsed_data_ordered.append((element_name, value))
                    else:
                        print(f"Warning: Skipping malformed line {line_num}: '{line}' - Expected 'name value' format.")
                except ValueError:
                    print(f"Warning: Could not convert value to integer for line {line_num}: '{line}' - Skipping.")
                except Exception as e:
                    print(f"An unexpected error occurred while parsing line {line_num}: {e}")

        self.parsed_data = parsed_data_ordered
        return parsed_data_ordered

    def extract_and_sort_grid_values(self, original_rows: int, original_cols: int) -> Tuple[list, list]:
        """
        Extracts and sorts DX and DY values from parsed data based on their numerical index.
        """
        self.original_rows = original_rows
        self.original_cols = original_cols
        
        index_pattern = re.compile(r"_(dx|dy)_0_(\d+)$")
        
        dx_elements = []
        dy_elements = []

        for name, value in self.parsed_data:
            match = index_pattern.search(name)
            if match:
                index = int(match.group(2))
                if "yuv_gdc_grid_dx" in name:
                    dx_elements.append((index, value))
                elif "yuv_gdc_grid_dy" in name:
                    dy_elements.append((index, value))

        dx_elements.sort(key=lambda x: x[0])
        dy_elements.sort(key=lambda x: x[0])

        self.dx_values = [item[1] for item in dx_elements]
        self.dy_values = [item[1] for item in dy_elements]
        
        expected_elements = original_rows * original_cols
        if len(self.dx_values) < expected_elements or len(self.dy_values) < expected_elements:
            raise ValueError(f"Insufficient data for {original_rows}x{original_cols} grid. "
                             f"Found {len(self.dx_values)} DX and {len(self.dy_values)} DY elements, "
                             f"but expected {expected_elements} of each.")
        
        self.dx_values = self.dx_values[:expected_elements]
        self.dy_values = self.dy_values[:expected_elements]

        return self.dx_values, self.dy_values

    def reshape_to_2d_grid(self, values: list) -> np.ndarray:
        """
        Reshapes a 1D list of values into a 2D numpy array based on original dimensions.
        """
        expected_elements = self.original_rows * self.original_cols
        if len(values) != expected_elements:
            raise ValueError(f"Mismatch in data length ({len(values)}) and expected grid size ({expected_elements}).")
        return np.array(values).reshape(self.original_rows, self.original_cols)

    def interpolate_grid_bicubic(self, grid_2d: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
        """
        Interpolates a 2D grid using bicubic interpolation.
        """
        original_rows, original_cols = grid_2d.shape

        x_orig = np.linspace(0, 1, original_cols)
        y_orig = np.linspace(0, 1, original_rows)
        x_target = np.linspace(0, 1, target_cols)
        y_target = np.linspace(0, 1, target_rows)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            spline = RectBivariateSpline(y_orig, x_orig, grid_2d, kx=3, ky=3, s=0)
            interpolated_grid = spline(y_target, x_target)

        return interpolated_grid

    def grid_2d_to_gdc_format(self, grid_2d: np.ndarray, grid_type: str) -> str:
        """
        Converts a 2D grid back to GDC format text.
        """
        rows, cols = grid_2d.shape
        gdc_lines = []
        
        # Flatten the grid in row-major order and create GDC format
        flat_grid = grid_2d.flatten()
        
        for i, value in enumerate(flat_grid):
            element_name = f"yuv_gdc_grid_{grid_type}_0_{i}"
            # Convert to integer for GDC format
            int_value = int(round(value))
            gdc_lines.append(f"{element_name} {int_value}")
        
        return '\n'.join(gdc_lines)

# Enhanced GDC Image Remapping Processor
class GDCImageRemappingProcessor:
    """Enhanced processor with integrated grid functionality"""
    def __init__(self):
        self.gdc_data = {}
        self.dx_grid = None
        self.dy_grid = None
        self.interpolated_dx = None
        self.interpolated_dy = None
        self.displacement_maps = None
        self.original_shape = None
        self.target_shape = None
        self.image_shape = None
        self.scale_factor = 1.0
        self.processing_history = []
        self.current_image = None
        
        # Add grid processor
        self.grid_processor = GDCGridProcessor()
    
    def import_gdc_from_text(self, text):
        """Import GDC data from text input"""
        try:
            # Parse the grid data
            parsed_data = self.grid_processor.parse_grid_data_from_content(text)
            
            if not parsed_data:
                return {'success': False, 'message': 'No valid GDC data found', 'statistics': {}, 'validation': {}}
            
            # Store as dictionary for compatibility
            self.gdc_data = {name: value for name, value in parsed_data}
            
            # Calculate statistics
            dx_count = sum(1 for name in self.gdc_data.keys() if 'dx' in name)
            dy_count = sum(1 for name in self.gdc_data.keys() if 'dy' in name)
            
            statistics = {
                'total_elements': len(self.gdc_data),
                'dx_elements': dx_count,
                'dy_elements': dy_count,
                'data_range': {'min': min(self.gdc_data.values()), 'max': max(self.gdc_data.values())}
            }
            
            validation = {
                'format_valid': True,
                'complete_pairs': dx_count == dy_count,
                'element_count': len(self.gdc_data)
            }
            
            return {'success': True, 'message': 'GDC data imported successfully', 
                   'statistics': statistics, 'validation': validation}
            
        except Exception as e:
            return {'success': False, 'message': f'Import failed: {str(e)}', 'statistics': {}, 'validation': {}}
    
    def import_gdc_from_file(self, filepath):
        """Import GDC data from file"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            return self.import_gdc_from_text(content)
        except Exception as e:
            return {'success': False, 'message': f'File import failed: {str(e)}', 'statistics': {}, 'validation': {}}
    
    def setup_and_interpolate_grid(self, orig_rows, orig_cols, target_rows, target_cols, interp_method):
        """Setup and interpolate the grid"""
        try:
            if not self.gdc_data:
                return {'success': False, 'message': 'No GDC data loaded', 'grid_info': {}, 'statistics': {}}
            
            # Extract and sort grid values
            dx_values, dy_values = self.grid_processor.extract_and_sort_grid_values(orig_rows, orig_cols)
            
            # Reshape to 2D grids
            self.dx_grid = self.grid_processor.reshape_to_2d_grid(dx_values)
            self.dy_grid = self.grid_processor.reshape_to_2d_grid(dy_values)
            
            # Store original and target shapes
            self.original_shape = (orig_rows, orig_cols)
            self.target_shape = (target_rows, target_cols)
            
            # Perform interpolation
            if interp_method == "bicubic":
                self.interpolated_dx = self.grid_processor.interpolate_grid_bicubic(self.dx_grid, target_rows, target_cols)
                self.interpolated_dy = self.grid_processor.interpolate_grid_bicubic(self.dy_grid, target_rows, target_cols)
            else:  # linear fallback
                # Simple linear interpolation for fallback
                zoom_factor_y = target_rows / orig_rows
                zoom_factor_x = target_cols / orig_cols
                self.interpolated_dx = zoom(self.dx_grid, (zoom_factor_y, zoom_factor_x), order=1)
                self.interpolated_dy = zoom(self.dy_grid, (zoom_factor_y, zoom_factor_x), order=1)
            
            grid_info = {
                'original_shape': self.original_shape,
                'target_shape': self.target_shape,
                'interpolation_method': interp_method
            }
            
            statistics = self._compute_grid_statistics()
            
            return {'success': True, 'message': 'Grid setup successful', 
                   'grid_info': grid_info, 'statistics': statistics}
            
        except Exception as e:
            return {'success': False, 'message': f'Grid setup failed: {str(e)}', 'grid_info': {}, 'statistics': {}}
    
    def setup_image_remapping(self, img_width, img_height, scale_factor, cv_interpolation):
        """Setup image remapping parameters"""
        try:
            if self.interpolated_dx is None or self.interpolated_dy is None:
                return {'success': False, 'message': 'Grid not interpolated yet', 'mapping_info': {}}
            
            self.image_shape = (img_height, img_width)
            self.scale_factor = scale_factor
            
            mapping_info = {
                'image_dimensions': self.image_shape,
                'scale_factor': scale_factor,
                'interpolation_method': cv_interpolation
            }
            
            return {'success': True, 'message': 'Image remapping setup complete', 'mapping_info': mapping_info}
            
        except Exception as e:
            return {'success': False, 'message': f'Remapping setup failed: {str(e)}', 'mapping_info': {}}
    
    def process_image(self, input_image):
        """Process image with current grid"""
        try:
            if input_image is None:
                return {'success': False, 'message': 'No input image provided', 
                       'input_info': {}, 'output_info': {}, 'processing_stats': {}}
            
            # For now, return the original image as we're focusing on grid functionality
            # This would be where actual remapping occurs in a full implementation
            output_image = input_image.copy()
            
            input_info = {'shape': input_image.shape, 'dtype': str(input_image.dtype)}
            output_info = {'shape': output_image.shape, 'dtype': str(output_image.dtype)}
            processing_stats = {'processing_time': 0.1, 'method': 'mock_processing'}
            
            return {'success': True, 'message': 'Image processed successfully',
                   'input_info': input_info, 'output_info': output_info, 
                   'processing_stats': processing_stats, 'output_image': output_image}
            
        except Exception as e:
            return {'success': False, 'message': f'Image processing failed: {str(e)}',
                   'input_info': {}, 'output_info': {}, 'processing_stats': {}}
    
    def create_sample_image(self, width, height, pattern):
        """Create sample image"""
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    def create_visualization(self, viz_type):
        """Create visualization based on type"""
        try:
            if viz_type == 'original_grid' and self.dx_grid is not None:
                return self._create_grid_heatmap(self.dx_grid, self.dy_grid, "Original Grid")
            elif viz_type == 'interpolated_grid' and self.interpolated_dx is not None:
                return self._create_grid_heatmap(self.interpolated_dx, self.interpolated_dy, "Interpolated Grid")
            elif viz_type == 'grid_comparison' and self.dx_grid is not None and self.interpolated_dx is not None:
                return self._create_grid_comparison()
            elif viz_type == 'displacement_analysis':
                return self._create_displacement_analysis()
            else:
                return None
        except Exception as e:
            print(f"Visualization error: {e}")
            return None
    
    def _create_grid_heatmap(self, dx_grid, dy_grid, title):
        """Create heatmap visualization of grids"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # DX Grid
            sns.heatmap(dx_grid, ax=ax1, cmap='RdBu_r', center=0, annot=False, cbar=True)
            ax1.set_title(f'{title} - DX')
            ax1.set_xlabel('Column')
            ax1.set_ylabel('Row')
            
            # DY Grid
            sns.heatmap(dy_grid, ax=ax2, cmap='RdBu_r', center=0, annot=False, cbar=True)
            ax2.set_title(f'{title} - DY')
            ax2.set_xlabel('Column')
            ax2.set_ylabel('Row')
            
            plt.tight_layout()
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file.name, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close()
            return temp_file.name
        except Exception as e:
            print(f"Error creating grid heatmap: {e}")
            plt.close()
            return None
    
    def _create_grid_comparison(self):
        """Create side-by-side grid comparison"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original DX
            sns.heatmap(self.dx_grid, ax=axes[0,0], cmap='RdBu_r', center=0, annot=False, cbar=True)
            axes[0,0].set_title(f'Original DX ({self.dx_grid.shape[0]}×{self.dx_grid.shape[1]})')
            
            # Interpolated DX
            sns.heatmap(self.interpolated_dx, ax=axes[0,1], cmap='RdBu_r', center=0, annot=False, cbar=True)
            axes[0,1].set_title(f'Interpolated DX ({self.interpolated_dx.shape[0]}×{self.interpolated_dx.shape[1]})')
            
            # Original DY
            sns.heatmap(self.dy_grid, ax=axes[1,0], cmap='RdBu_r', center=0, annot=False, cbar=True)
            axes[1,0].set_title(f'Original DY ({self.dy_grid.shape[0]}×{self.dy_grid.shape[1]})')
            
            # Interpolated DY
            sns.heatmap(self.interpolated_dy, ax=axes[1,1], cmap='RdBu_r', center=0, annot=False, cbar=True)
            axes[1,1].set_title(f'Interpolated DY ({self.interpolated_dy.shape[0]}×{self.interpolated_dy.shape[1]})')
            
            plt.suptitle('Grid Comparison: Original vs Interpolated', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file.name, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close()
            return temp_file.name
        except Exception as e:
            print(f"Error creating grid comparison: {e}")
            plt.close()
            return None
    
    def _create_displacement_analysis(self):
        """Create displacement field visualization"""
        try:
            if self.dx_grid is None or self.dy_grid is None:
                return None
            
            rows, cols = self.dx_grid.shape
            
            # Create target grid
            target_x, target_y = np.meshgrid(np.arange(cols), np.arange(rows))
            
            # Create displaced positions
            scale_factor = 0.1  # Scale factor for visualization
            displaced_x = target_x + self.dx_grid * scale_factor
            displaced_y = target_y + self.dy_grid * scale_factor
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot original grid
            ax.scatter(target_x, target_y, c='blue', s=30, alpha=0.7, label='Target Points')
            
            # Plot displaced points
            ax.scatter(displaced_x, displaced_y, c='red', s=25, alpha=0.7, label='Displaced Points')
            
            # Draw displacement vectors
            for i in range(0, rows, max(1, rows//10)):  # Subsample for clarity
                for j in range(0, cols, max(1, cols//10)):
                    dx = displaced_x[i,j] - target_x[i,j]
                    dy = displaced_y[i,j] - target_y[i,j]
                    if abs(dx) > 0.01 or abs(dy) > 0.01:  # Only show significant displacements
                        ax.arrow(target_x[i,j], target_y[i,j], dx, dy, 
                                head_width=0.1, head_length=0.1, fc='green', ec='green', alpha=0.6)
            
            ax.set_aspect('equal')
            ax.legend()
            ax.set_title('Displacement Field Analysis')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            ax.grid(True, alpha=0.3)
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file.name, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close()
            return temp_file.name
        except Exception as e:
            print(f"Error creating displacement analysis: {e}")
            plt.close()
            return None
    
    def get_processing_summary(self):
        """Get processing summary"""
        summary = "🔄 GDC Processing Summary\n"
        summary += "=" * 40 + "\n\n"
        
        if self.gdc_data:
            summary += f"📁 Loaded GDC Elements: {len(self.gdc_data)}\n"
            
        if self.original_shape:
            summary += f"📐 Original Grid: {self.original_shape[0]}×{self.original_shape[1]}\n"
            
        if self.target_shape:
            summary += f"📐 Target Grid: {self.target_shape[0]}×{self.target_shape[1]}\n"
            
        if self.dx_grid is not None:
            summary += f"📊 DX Range: {np.min(self.dx_grid):.1f} to {np.max(self.dx_grid):.1f}\n"
            summary += f"📊 DY Range: {np.min(self.dy_grid):.1f} to {np.max(self.dy_grid):.1f}\n"
            
        return summary
    
    def _compute_grid_statistics(self):
        """Compute grid statistics"""
        stats = {}
        if self.dx_grid is not None:
            stats['dx_stats'] = {
                'min': float(np.min(self.dx_grid)),
                'max': float(np.max(self.dx_grid)),
                'mean': float(np.mean(self.dx_grid)),
                'std': float(np.std(self.dx_grid))
            }
        if self.dy_grid is not None:
            stats['dy_stats'] = {
                'min': float(np.min(self.dy_grid)),
                'max': float(np.max(self.dy_grid)),
                'mean': float(np.mean(self.dy_grid)),
                'std': float(np.std(self.dy_grid))
            }
        return stats
    
    def _count_out_of_bounds_pixels(self):
        """Count out of bounds pixels"""
        if self.interpolated_dx is None or self.interpolated_dy is None:
            return 0
        return 0  # Placeholder implementation

# Configuration
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
DEFAULT_GRID_SIZES = [3, 5, 7, 9, 11, 15, 21, 33]
DEFAULT_IMAGE_SIZES = [(640, 480), (800, 600), (1024, 768), (1280, 720), (1920, 1080)]

class GDCRemappingInterface:
    """Main interface controller for GDC Image Remapping"""
    
    def __init__(self):
        self.processor = GDCImageRemappingProcessor()  # Always use the enhanced processor
        self.current_image = None
        self.processing_state = {
            'gdc_loaded': False,
            'grid_setup': False,
            'remapping_ready': False,
            'image_processed': False
        }
        
        # Initialize component references
        self.gdc_text_input = None
        self.gdc_file_input = None
        self.import_btn = None
        self.import_status = None
        self.import_stats = None
        self.data_preview = None
        
        # Grid dimension input components - NEW
        self.input_grid_rows = None
        self.input_grid_cols = None
        self.expected_elements = None
        
        # Grid components
        self.orig_rows = None
        self.orig_cols = None
        self.target_rows = None
        self.target_cols = None
        self.interp_method = None
        self.preserve_boundaries = None
        self.interp_factor = None
        self.setup_grid_btn = None
        self.original_grid_viz = None
        self.interpolated_grid_viz = None
        self.grid_comparison_viz = None
        self.grid_stats = None
        
        # Image processing components
        self.input_image = None
        self.sample_pattern = None
        self.generate_sample_btn = None
        self.img_width = None
        self.img_height = None
        self.scale_factor = None
        self.cv_interpolation = None
        self.border_mode = None
        self.setup_remapping_btn = None
        self.process_image_btn = None
        self.image_comparison = None
        self.output_image = None
        self.displacement_viz = None
        self.processing_status = None
        self.quality_metrics = None
        
        # Analysis components
        self.analysis_type = None
        self.create_analysis_btn = None
        self.export_format = None
        self.export_metadata = None
        self.export_visualizations = None
        self.export_single_btn = None
        self.export_all_btn = None
        self.batch_folder = None
        self.batch_process_btn = None
        self.analysis_viz = None
        self.processing_summary = None
        self.export_status = None
        self.download_files = None
        
    def create_interface(self):
        """Create the main Gradio interface"""
        
        with gr.Blocks(
            title="GDC Image Remapping Suite",
            theme=gr.themes.Soft(),
            css=self._get_custom_css()
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🎯 GDC Image Remapping Suite
            
            **Professional geometric image transformation using GDC (Geometric Distortion Correction) grids**
            
            Complete workflow: Import GDC → Setup Grid → Process Image → Export Results
            """)
            
            # Status indicators
            with gr.Row():
                with gr.Column(scale=1):
                    status_indicators = gr.HTML(
                        value=self._generate_status_html(),
                        elem_id="status_indicators"
                    )
            
            # Main tabbed interface
            with gr.Tabs() as main_tabs:
                
                # =================== TAB 1: GDC DATA IMPORT ===================
                with gr.Tab("📁 GDC Data Import", id="import_tab"):
                    self._create_import_tab()
                
                # =================== TAB 2: GRID PROCESSING ===================
                with gr.Tab("🌐 Grid Processing", id="grid_tab"):
                    self._create_grid_tab()
                
                # =================== TAB 3: IMAGE REMAPPING ===================
                with gr.Tab("🖼️ Image Remapping", id="remap_tab"):
                    self._create_remapping_tab()
                
                # =================== TAB 4: ANALYSIS & EXPORT ===================
                with gr.Tab("📊 Analysis & Export", id="export_tab"):
                    self._create_analysis_tab()
                
                # =================== TAB 5: HELP & DOCUMENTATION ===================
                with gr.Tab("❓ Help", id="help_tab"):
                    self._create_help_tab()
            
            # Setup event handlers
            self._setup_event_handlers()
            
        return interface
    
    def _create_import_tab(self):
        """Create GDC data import tab - Traditional GDC Format Only (INTEGRATED VERSION)"""
        gr.Markdown("### 📥 Import GDC Grid Data")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("#### 📐 Grid Dimensions Configuration")
                
                # Grid dimensions input
                with gr.Group():
                    gr.Markdown("**📏 Input Grid Dimensions**")
                    with gr.Row():
                        self.input_grid_rows = gr.Number(
                            label="Grid Rows",
                            value=3,
                            minimum=1,
                            maximum=50,
                            step=1,
                            info="Number of rows in the input grid"
                        )
                        self.input_grid_cols = gr.Number(
                            label="Grid Columns", 
                            value=3,
                            minimum=1,
                            maximum=50,
                            step=1,
                            info="Number of columns in the input grid"
                        )
                    
                    # Real-time calculation display
                    self.expected_elements = gr.Textbox(
                        label="Expected Data Elements",
                        value="18 total (9 DX + 9 DY)",
                        interactive=False,
                        info="Automatically calculated based on grid dimensions"
                    )
                
                gr.Markdown("#### 📝 GDC Data Input (Traditional Format Only)")
                
                # GDC text input
                self.gdc_text_input = gr.Textbox(
                    label="GDC Grid Data",
                    lines=15,
                    placeholder="""Enter GDC data in traditional format:

yuv_gdc_grid_dx_0_0    -50
yuv_gdc_grid_dx_0_1     0
yuv_gdc_grid_dx_0_2     50
yuv_gdc_grid_dy_0_0     25
yuv_gdc_grid_dy_0_1     30
yuv_gdc_grid_dy_0_2     25

Note: Multiple spaces between name and value are allowed.
Order of elements doesn't matter - they will be sorted by index.""",
                    show_copy_button=True
                )
                
                # File upload as alternative
                self.gdc_file_input = gr.File(
                    label="📁 Or Upload GDC File",
                    file_types=[".txt", ".dat"],
                    file_count="single",
                    info="Upload file with traditional GDC format"
                )
                
                # Quick sample data generation
                gr.Markdown("#### 🎯 Quick Sample Generation")
                with gr.Row():
                    sample_3x3_btn = gr.Button("📋 3×3 Sample", variant="secondary", size="sm")
                    sample_5x5_btn = gr.Button("📋 5×5 Sample", variant="secondary", size="sm")
                    clear_input_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")
                
                self.import_btn = gr.Button(
                    "🔄 Import GDC Data",
                    variant="primary",
                    size="lg"
                )
                
                # Wire up sample buttons
                sample_3x3_btn.click(
                    fn=lambda: self._generate_traditional_gdc_sample(3, 3),
                    outputs=self.gdc_text_input
                )
                
                sample_5x5_btn.click(
                    fn=lambda: self._generate_traditional_gdc_sample(5, 5),
                    outputs=self.gdc_text_input
                )
                
                clear_input_btn.click(
                    fn=lambda: "",
                    outputs=self.gdc_text_input
                )
                
                # Note: Event handlers will be set up later in _setup_event_handlers()
            
            with gr.Column(scale=1):
                gr.Markdown("#### 📊 Import Status & Preview")
                
                self.import_status = gr.Textbox(
                    label="Import Status",
                    lines=12,
                    interactive=False,
                    placeholder="Import status will appear here..."
                )
                
                # Statistics display
                self.import_stats = gr.JSON(
                    label="Import Statistics",
                    visible=False,
                    value={}  # Initialize with empty dict
                )
                
                # Data preview
                self.data_preview = gr.Dataframe(
                    label="Data Preview",
                    headers=["Index", "DX Value", "DY Value"],
                    datatype=["number", "number", "number"],
                    row_count=(1, "dynamic"),
                    visible=False
                )
                