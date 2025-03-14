# GUI Module Refactoring

## Overview

This document summarizes the refactoring work done to move GUI-related code into a dedicated module within the SynapseClusterEM system.

## Changes Made

1. **Created a new directory structure**:
   - Created `synapse/gui/` directory for GUI-related code
   - Created `synapse/gui/assets/` for GUI assets
   - Added `__init__.py` to expose the SynapseGUI class
   - Added `README.md` with documentation

2. **Moved files**:
   - Moved `synapse_gui.py` from the root directory to `synapse/gui/synapse_gui.py`
   - Moved GUI assets to `synapse/gui/assets/`
   - Removed the original files from the root directory

3. **Created a launcher script**:
   - Added `run_gui.py` in the root directory
   - The launcher imports the SynapseGUI class from the new module

4. **Updated asset handling**:
   - Created a robust asset path resolution mechanism (`get_asset_path` function)
   - Updated file paths to ensure assets are found regardless of how the module is executed

5. **Updated imports**:
   - Updated `synapse/__init__.py` to include SynapseGUI
   - Made the GUI import optional to avoid tkinter dependency issues

## Benefits

1. **Better organization**: GUI code is now in a dedicated module, making the codebase more organized and easier to navigate.

2. **Clearer dependencies**: The GUI components are separated from the core functionality.

3. **Improved maintainability**: Related code is grouped together, making it easier to maintain and extend.

4. **Robust asset handling**: Assets can now be found regardless of how the module is imported or used.

5. **Optional dependency**: The GUI can be imported separately, avoiding tkinter dependencies when not needed.

## Usage

The GUI can now be used in two ways:

1. **Using the launcher script**:
   ```
   python run_gui.py
   ```

2. **From Python code**:
   ```python
   import tkinter as tk
   from synapse import SynapseGUI
   
   root = tk.Tk()
   app = SynapseGUI(root)
   root.mainloop()
   ``` 