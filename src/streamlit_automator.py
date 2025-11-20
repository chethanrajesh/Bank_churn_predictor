# src/streamlit_automator.py
"""
Automatically launches Streamlit dashboard after notebook execution
"""

import os
import subprocess
import time
import sys
from pathlib import Path
import webbrowser

def launch_streamlit_dashboard():
    """Launch Streamlit dashboard automatically"""
    
    print("🚀 LAUNCHING STREAMLIT DASHBOARD AUTOMATION...")
    print("This will:")
    print("• Start Streamlit server automatically")
    print("• Open dashboard in your web browser")
    print("• Show beautiful 5-page visualizations")
    print("• Provide interactive data exploration")
    print()
    
    # Check if data files exist
    data_dir = Path("../powerbi/data")
    required_files = [
        'predictions_data.csv',
        'model_metrics.csv',
        'shap_values_summary.csv',
        'driver_breakdown.csv',
        'customer_segments.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing data files: {missing_files}")
        return False
    
    print("✅ All data files verified!")
    
    # Launch Streamlit
    try:
        print("🎯 Starting Streamlit server...")
        
        # Get the path to streamlit dashboard
        dashboard_path = Path(__file__).parent / "streamlit_dashboard.py"
        
        # Launch Streamlit in a subprocess
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path), "--server.port=8501", "--server.headless=true"
        ])
        
        # Wait for server to start
        print("⏳ Waiting for Streamlit server to start...")
        time.sleep(5)
        
        # Open browser automatically
        print("🌐 Opening dashboard in web browser...")
        webbrowser.open("http://localhost:8501")
        
        print("✅ Streamlit dashboard launched successfully!")
        print("\n📊 YOUR DASHBOARD IS NOW RUNNING!")
        print("🔗 Open: http://localhost:8501")
        print("\n🎯 Features:")
        print("• 5 interactive pages with beautiful visuals")
        print("• Real-time data exploration")
        print("• Filtering and drill-down capabilities")
        print("• Export and analysis tools")
        
        # Keep the process running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
            process.terminate()
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit launch failed: {e}")
        print("💡 Try running manually: streamlit run src/streamlit_dashboard.py")
        return False

if __name__ == "__main__":
    launch_streamlit_dashboard()