#!/usr/bin/env python3
"""
Image Remapping Suite
Main application launcher

This application provides a comprehensive tool for:
- Simulating various lens distortion effects
- Testing advanced correction algorithms
- Arbitrary geometric transformations
- Visualizing distortion grids and heatmaps
- Exporting data in various formats (CSV, GDC)
- Quality assessment and validation

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
        'pillow'
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
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def launch_main_interface():
    """Launch the improved 3-tab main interface"""
    try:
        from interfaces.gradio_main_improved import create_gradio_interface
        
        print("🚀 Starting improved 3-tab image remapping interface...")
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
        print("Falling back to original interface...")
        try:
            from interfaces.gradio_main import create_gradio_interface
            interface = create_gradio_interface()
            interface.launch(
                share=True,
                server_name="localhost",
                show_error=True,
                quiet=True
            )
        except ImportError:
            print("❌ Could not load any interface. Please check your installation.")
            sys.exit(1)

def launch_gdc_utility():
    """Launch the GDC grid utility interface"""
    try:
        from interfaces.gradio_gdc import create_gradio_interface
        
        print("🚀 Starting GDC grid utility interface...")
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

def launch_legacy_interface():
    """Launch the original single-page interface"""
    try:
        from interfaces.gradio_main import create_gradio_interface
        
        print("🚀 Starting legacy single-page interface...")
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

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Image Remapping Suite - Comprehensive geometric transformation toolkit"
    )
    parser.add_argument(
        '--interface',
        choices=['main', 'gdc', 'legacy'],
        default='main',
        help='Choose which interface to launch (default: main - improved 3-tab interface)'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='Image Remapping Suite v1.1.0 - Improved 3-Tab Interface'
    )
    
    args = parser.parse_args()
    
    print("🔍 Image Remapping Suite v1.1.0")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Missing dependencies. Please install required packages.")
        sys.exit(1)
    
    print("✅ All dependencies found")
    print("📝 Use Ctrl+C to stop the server")
    print()
    
    # Show interface information
    if args.interface == 'main':
        print("🎯 Launching IMPROVED 3-TAB INTERFACE")
        print("   Features:")
        print("   • Tab 1: Distortion Simulation - Parameter control & preview")
        print("   • Tab 2: Correction & Analysis - Method comparison & quality")
        print("   • Tab 3: Export & Data - Export tools & documentation")
        print("   • Cleaner UI with better organization")
        print("   • Progressive disclosure of features")
    elif args.interface == 'gdc':
        print("🎯 Launching GDC GRID UTILITY")
        print("   Features:")
        print("   • GDC file parsing and interpolation")
        print("   • Bicubic interpolation for resolution enhancement")
        print("   • Multiple export formats")
    elif args.interface == 'legacy':
        print("🎯 Launching LEGACY SINGLE-PAGE INTERFACE")
        print("   Features:")
        print("   • Original all-in-one interface")
        print("   • All features on single page")
        print("   • For users preferring classic layout")
    
    print()
    
    try:
        if args.interface == 'main':
            launch_main_interface()
        elif args.interface == 'gdc':
            launch_gdc_utility()
        elif args.interface == 'legacy':
            launch_legacy_interface()
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()