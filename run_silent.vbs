' Starts the controller with NO window and no console flash (run run.bat once first so dependencies are installed).
' Stop it with F10, stop.bat, or the tray icon if enabled.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
here = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = here
py = here & "\.venv\Scripts\pythonw.exe"
If Not fso.FileExists(py) Then py = "pythonw"
sh.Run """" & py & """ main.py --no-preview", 0, False
