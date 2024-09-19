Run(".\MetaTrader5\Terminal64.exe /portable /skipupdate")

; Wait for the MT5 terminal to fully load (you might need to adjust the wait time)
Sleep(2000)

; Open the "Navigator" window
Send("{Esc}{ALTDOWN}f{ALTUP}fC{ENTER}")

Sleep(30000)

; Run the script
Send("{F5}")

; Run the script
Send("{ALTDOWN}{F4}")
Send("{ALTUP}")