#!/usr/bin/env python3
"""
Image Remapping Suite - Enhanced Main Application Launcher
Now includes integrated GDC Grid Processing capabilities

This application provides a comprehensive tool for:
- Simulating various lens distortion effects
- Testing advanced correction algorithms
- Arbitrary geometric transformations
- Visualizing distortion grids and heatmaps
- Exporting data in various formats (CSV, GDC)
- Quality assessment and validation
- GDC Grid Interpolation and Processing

Author: Balaji R
License: MIT
"""

import sys
import os
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'numpy',
        'opencv-python',
        'matplotlib',
        'gradio',
        'scipy',
        'pillow',
        'seaborn',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def launch_main_interface():
    """Launch the main image remapping interface"""
    try:
        from interfaces.gradio_main import create_gradio_interface
        
        print("🚀 Starting main image remapping interface...")
        print("📋 Features: Lens distortion simulation, correction algorithms, quality assessment")
        interface = create_gradio_interface()
        interface.launch(
            share=True,
            server_name="localhost",
            show_error=True,
            quiet=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all module files are properly structured.")
        sys.exit(1)

def launch_gdc_utility():
    """Launch the enhanced GDC grid utility interface"""
    try:
        from interfaces.gradio_gdc import create_gdc_interface
        
        print("🚀 Starting GDC grid interpolation interface...")
        print("📋 Features: Grid parsing, bicubic interpolation, hardware-ready exports")
        interface = create_gdc_interface()
        interface.launch(
            share=True,
            server_name="localhost",
            show_error=True,
            quiet=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all module files are properly structured.")
        sys.exit(1)

def launch_integrated_interface():
    """Launch an integrated interface with both main and GDC capabilities"""
    try:
        import gradio as gr
        from interfaces.gradio_main import create_gradio_interface as create_main_interface
        from interfaces.gradio_gdc import create_gdc_interface
        
        print("🚀 Starting integrated Image Remapping Suite...")
        print("📋 Features: Full lens distortion toolkit + GDC grid processing")
        
        # Create tabbed interface
        with gr.Blocks(title="Image Remapping Suite - Complete", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🔍 Image Remapping Suite - Complete Toolkit
            
            **Professional geometric image transformation and grid processing platform**
            
            This integrated interface provides access to both lens distortion correction capabilities 
            and advanced GDC grid interpolation tools in a single, comprehensive application.
            """)
            
            with gr.Tabs():
                with gr.Tab("🌀 Lens Distortion Toolkit", id="main"):
                    gr.Markdown("""
                    ### Brown-Conrady Distortion Modeling & Correction
                    
                    Complete toolkit for lens distortion simulation, multiple correction algorithms,
                    quality assessment, and data export for analysis and hardware implementation.
                    """)
                    
                    # Import and embed the main interface components
                    # Note: This would require refactoring the main interface to be embeddable
                    gr.Markdown("**Main distortion correction interface components would be embedded here**")
                    gr.Markdown("For now, please use the standalone interfaces via command line options.")
                
                with gr.Tab("🎯 GDC Grid Processing", id="gdc"):
                    gr.Markdown("""
                    ### Geometric Distortion Correction Grid Interpolation
                    
                    Advanced bicubic interpolation of GDC grid files for hardware implementation.
                    Includes comprehensive validation, visualization, and export capabilities.
                    """)
                    
                    # Import and embed the GDC interface components  
                    # Note: This would require refactoring the GDC interface to be embeddable
                    gr.Markdown("**GDC grid processing interface components would be embedded here**")
                    gr.Markdown("For now, please use the standalone interfaces via command line options.")
            
            gr.Markdown("""
            ---
            ### 🚀 Quick Start Options
            
            **Launch Standalone Interfaces:**
            ```bash
            # Lens Distortion Toolkit
            python main.py --interface main
            
            # GDC Grid Processing
            python main.py --interface gdc
            
            # This integrated view
            python main.py --interface integrated
            ```
            
            **Features Overview:**
            - **Lens Distortion**: Brown-Conrady model, multiple correction algorithms, quality metrics
            - **GDC Processing**: Bicubic interpolation, hardware exports, comprehensive validation
            - **Visualization**: Real-time previews, heatmaps, comparison plots
            - **Export**: Multiple formats including CSV, GDC, and complete calibration packages
            """)
        
        interface.launch(
            share=True,
            server_name="localhost", 
            show_error=True,
            quiet=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Falling back to individual interface selection.")
        return False
    
    return True

def show_feature_overview():
    """Display comprehensive feature overview"""
    print("""
🔍 IMAGE REMAPPING SUITE - FEATURE OVERVIEW
==========================================

📋 MAIN LENS DISTORTION TOOLKIT:
  🌀 Brown-Conrady Distortion Model
    • Radial distortion (K1, K2, K3)
    • Tangential distortion (P1, P2)
    • Real-time parameter validation
  
  ⚖️ Multiple Correction Algorithms
    • Iterative Newton-Raphson (most accurate)
    • Analytical closed-form (fastest for K1-only)
    • Polynomial approximation (balanced)
    • Original basic correction (visualization)
  
  📊 Quality Assessment
    • PSNR, SSIM, correlation metrics
    • Geometric error analysis
    • Method comparison tools
  
  🎨 Advanced Visualization
    • Real-time distortion preview
    • Grid mapping visualization
    • Distortion magnitude heatmaps
    • Vector field displays

📋 GDC GRID PROCESSING TOOLKIT:
  🎯 Advanced Interpolation
    • Bicubic spline interpolation
    • Configurable resolution scaling
    • Sub-pixel precision maintenance
  
  📁 Multiple Export Formats
    • CSV for analysis and spreadsheets
    • GDC format for hardware implementation
    • Combined files for complete datasets
    • ZIP packages for easy distribution
  
  🔒 Robust Validation
    • Input format verification
    • Dimension compatibility checks
    • File size and content validation
    • Error handling and recovery
  
  📈 Comprehensive Analysis
    • Statistical grid analysis
    • Quality metrics computation
    • Visual comparison tools
    • Processing history tracking

🚀 SUPPORTED WORKFLOWS:
  • Camera calibration and correction
  • Hardware ISP/FPGA implementation
  • Image processing research
  • Quality assessment and validation
  • Geometric transformation analysis
  • Grid data conversion and scaling

💡 USE CASES:
  • Automotive camera systems
  • Security and surveillance
  • Mobile device cameras
  • Scientific imaging
  • Virtual reality applications
  • Augmented reality systems
""")

def main():
    """Enhanced main application entry point"""
    parser = argparse.ArgumentParser(
        description="Image Remapping Suite - Comprehensive geometric transformation toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Launch main lens distortion interface
  python main.py --interface gdc          # Launch GDC grid processing interface  
  python main.py --interface integrated   # Launch integrated interface
  python main.py --features               # Show detailed feature overview
  python main.py --check                  # Check dependencies only
        """
    )
    
    parser.add_argument(
        '--interface',
        choices=['main', 'gdc', 'integrated'],
        default='main',
        help='Choose which interface to launch (default: main)'
    )
    
    parser.add_argument(
        '--features',
        action='store_true',
        help='Show comprehensive feature overview'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check dependencies and exit'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Image Remapping Suite v2.0.0 - Enhanced with GDC Processing'
    )
    
    args = parser.parse_args()
    
    print("🔍 IMAGE REMAPPING SUITE v2.0.0")
    print("=" * 55)
    print("Enhanced with GDC Grid Processing Capabilities")
    print("")
    
    # Show features and exit if requested
    if args.features:
        show_feature_overview()
        return
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("❌ Missing dependencies. Please install required packages.")
        sys.exit(1)
    
    print("✅ All dependencies found")
    
    # Check only mode
    if args.check:
        print("🎉 System ready for Image Remapping Suite")
        return
    
    print("📝 Use Ctrl+C to stop the server")
    print("")
    
    try:
        if args.interface == 'main':
            print("🎯 Selected: Main Lens Distortion Interface")
            launch_main_interface()
            
        elif args.interface == 'gdc':
            print("🎯 Selected: GDC Grid Processing Interface")
            launch_gdc_utility()
            
        elif args.interface == 'integrated':
            print("🎯 Selected: Integrated Interface")
            success = launch_integrated_interface()
            if not success:
                print("⚠️  Integrated interface not available, launching main interface...")
                launch_main_interface()
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Ensure all dependencies are installed: python main.py --check")
        print("  2. Check file permissions in the application directory")
        print("  3. Verify Python version compatibility (3.8+ required)")
        print("  4. Try launching individual interfaces separately")
        sys.exit(1)

if __name__ == "__main__":
    main()