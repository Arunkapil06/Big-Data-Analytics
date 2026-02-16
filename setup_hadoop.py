import os
import urllib.request
import zipfile
import shutil
import sys

def setup_hadoop():
    # Create Hadoop directory
    hadoop_home = "C:\\hadoop-3.3.6"
    bin_dir = os.path.join(hadoop_home, "bin")
    
    print("Creating Hadoop directories...")
    os.makedirs(bin_dir, exist_ok=True)
    
    # Download winutils.exe
    winutils_url = "https://github.com/steveloughran/winutils/raw/master/hadoop-3.3.6/bin/winutils.exe"
    winutils_path = os.path.join(bin_dir, "winutils.exe")
    
    print("Downloading winutils.exe...")
    try:
        urllib.request.urlretrieve(winutils_url, winutils_path)
    except Exception as e:
        print(f"Error downloading winutils.exe: {str(e)}")
        print("Please download it manually from:")
        print("https://github.com/steveloughran/winutils/raw/master/hadoop-3.3.6/bin/winutils.exe")
        print(f"And place it in: {bin_dir}")
        return False
    
    # Set environment variables
    print("Setting environment variables...")
    os.environ["HADOOP_HOME"] = hadoop_home
    
    # Add to PATH if not already there
    path = os.environ.get("PATH", "")
    hadoop_bin = os.path.join(hadoop_home, "bin")
    if hadoop_bin not in path:
        os.environ["PATH"] = f"{hadoop_bin};{path}"
    
    print("\nHadoop setup completed!")
    print(f"HADOOP_HOME is set to: {hadoop_home}")
    print("Please restart your Python IDE/terminal for changes to take effect.")
    return True

if __name__ == "__main__":
    if not os.path.exists("C:\\hadoop-3.3.6"):
        setup_hadoop()
    else:
        print("Hadoop directory already exists at C:\\hadoop-3.3.6")
        print("If you're still having issues, try deleting the directory and running this script again.") 