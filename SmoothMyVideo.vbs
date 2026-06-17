' Double-click launcher for SmoothMyVideo.
' Builds (tsc) and opens the Electron app with no console window.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")
sh.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)
sh.Run "cmd /c npm start", 0, False
